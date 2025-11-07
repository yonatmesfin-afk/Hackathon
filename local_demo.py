"""Local demo harness using python-can virtual backend.

Runs probe, receiver, and optional attacker in threads sharing a single in-process
virtual CAN bus. Preserves the receiver's HTTP /status endpoint so the Streamlit
Dashboard can connect normally.

Usage:
  python src/local_demo.py --probe_id 1 --period_ms 100 --dummy_count 2 \
    --run_attacker can_only --attack_rate 20 --duration 30

In another terminal:
  streamlit run src/dashboard.py

Notes:
- This bypasses OS-level SocketCAN and works on macOS.
- micro_events.json is still used for micro-gap plots.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Optional

import can

# Import modules and reuse their internals
from probe_node import _simulate_sensor_value, _build_real_payload, _build_dummy_payload  # type: ignore
from common import (
    PERIOD_MS,
    DUMMY_COUNT,
    make_tag,
    get_micro_pattern,
    pack_message,
    now_ns,
)
from receiver_node import (
    StatusServer,
    start_http_server,
    ProbeState,
    align_slot_start,
    compute_slot_index,
    micro_pattern_matches,
)
from micro_sim import write_micro_events_file, read_micro_events_file
import numpy as np
from pathlib import Path
from dataclasses import asdict, dataclass


def _sleep_until_epoch(target_epoch_ns: int, base_epoch_ns: int, base_perf_ns: int) -> None:
    while True:
        now_perf = time.perf_counter_ns()
        est_epoch = base_epoch_ns + (now_perf - base_perf_ns)
        remaining_ns = target_epoch_ns - est_epoch
        if remaining_ns <= 0:
            return
        time.sleep(max(0.0, (remaining_ns - 200_000) / 1e9))


def run_probe(bus: can.BusABC, probe_id: int, period_ms: int, dummy_count: int, key: bytes, micro_events_path: str) -> None:
    period_ns = period_ms * 1_000_000
    next_slot_epoch_ns = ((now_ns() // period_ns) + 1) * period_ns
    base_epoch_ns = time.time_ns()
    base_perf_ns = time.perf_counter_ns()
    counter = 0
    slot_index = 0

    try:
        while True:
            _sleep_until_epoch(next_slot_epoch_ns, base_epoch_ns, base_perf_ns)
            slot_perf_start = time.perf_counter_ns()
            cur_slot_start_ns = next_slot_epoch_ns

            counter = (counter + 1) & 0xFF
            sensor_val = _simulate_sensor_value()
            payload = _build_real_payload(probe_id, sensor_val)
            tag_data = bytes(payload) + counter.to_bytes(1, 'big') + int(slot_index).to_bytes(4, 'big')
            tag = make_tag(key, tag_data)

            micro_offsets_us = get_micro_pattern(probe_id, counter)
            for off_us in micro_offsets_us:
                target_perf = slot_perf_start + off_us * 1_000
                while True:
                    nowp = time.perf_counter_ns()
                    if nowp >= target_perf:
                        break
                    time.sleep(max(0.0, (target_perf - nowp) / 1e9))
                print(f"[probe] ts_ns={time.time_ns()} type=micro_gap offset_us={off_us}")

            write_micro_events_file(
                path=micro_events_path,
                probe_id=probe_id,
                counter=counter,
                slot_start_ns=cur_slot_start_ns,
                offsets_us=micro_offsets_us,
                mode=os.environ.get("MICRO_MODE", "noisy"),
                sigma_us=float(os.environ.get("MICRO_SIGMA_US", "30.0")),
            )

            # Send real frame (pack payload||counter||tag)
            data_bytes = pack_message(payload, counter, tag)
            msg = can.Message(arbitration_id=0x123, data=data_bytes, is_extended_id=False)
            bus.send(msg)

            # Send dummies
            period_us = period_ms * 1000
            # spread dummies
            for i in range(max(0, dummy_count)):
                off_us = int((i + 1) * period_us / (dummy_count + 1))
                target_perf = slot_perf_start + off_us * 1_000
                while True:
                    nowp = time.perf_counter_ns()
                    if nowp >= target_perf:
                        break
                    time.sleep(max(0.0, (target_perf - nowp) / 1e9))
                d_payload = _build_dummy_payload()
                d_tag = make_tag(b"dummy-key", bytes(d_payload))
                d_bytes = pack_message(d_payload, counter, d_tag)
                dmsg = can.Message(arbitration_id=0x123, data=d_bytes, is_extended_id=False)
                bus.send(dmsg)
                print(f"[probe] ts_ns={time.time_ns()} type=dummy payload={d_bytes.hex()} counter={counter}")

            next_slot_epoch_ns += period_ns
            slot_index += 1
    except KeyboardInterrupt:
        print("[probe] Stopped")


def run_receiver(bus: can.BusABC, period_ms: int, micro_events_path: str, key: bytes, host: str, port: int, micro_threshold: float) -> None:
    period_ns = period_ms * 1_000_000
    per_probe = {}
    results = StatusServer.results_ref  # shared deque
    start_http_server(host, port, results)
    print(f"[receiver] HTTP status on http://{host}:{port}/status (virtual)")

    try:
        while True:
            msg = bus.recv(timeout=0.2)
            if msg is None:
                continue
            ts_ns = time.time_ns()
            from common import unpack_message, within_slot
            try:
                payload, counter, tag = unpack_message(bytes(msg.data))
            except Exception as e:
                print(f"[receiver] unpack error: {e}")
                continue
            if len(payload) < 3:
                continue
            probe_id = payload[0] | (payload[1] << 8)
            sensor = payload[2]

            state = per_probe.get(probe_id)
            if state is None:
                state = ProbeState(slot_start_ns=-1, last_counter=-1, last_sensor=-1)
                per_probe[probe_id] = state
            if state.slot_start_ns < 0:
                state.slot_start_ns = align_slot_start(ts_ns, period_ns)

            slot_idx = compute_slot_index(state.slot_start_ns, ts_ns, period_ns)
            cur_slot_start = state.slot_start_ns + slot_idx * period_ns

            within = within_slot(ts_ns, cur_slot_start, 2)
            counter_ok = (state.last_counter == -1) or (((state.last_counter + 1) & 0xFF) == counter)
            sensor_ok = (state.last_sensor == -1) or (abs(sensor - state.last_sensor) <= 8)

            counter_bytes = counter.to_bytes(1, "big")
            slot_bytes = int(slot_idx).to_bytes(4, "big")
            tag_data = bytes(payload) + counter_bytes + slot_bytes
            tag_ok = (make_tag(key, tag_data) == tag)

            observed_ts = read_micro_events_file(
                path=micro_events_path,
                slot_start_ns=cur_slot_start,
                slot_end_ns=cur_slot_start + period_ns,
                probe_id=probe_id,
                counter=counter,
            )
            from common import SLOT_WINDOW_MS
            micro_score = micro_pattern_matches(probe_id, counter, observed_ts, cur_slot_start, SLOT_WINDOW_MS, micro_threshold)
            micro_ok = micro_score >= micro_threshold

            violations = [
                ("within", not within),
                ("counter", not counter_ok),
                ("sensor", not sensor_ok),
                ("tag", not tag_ok),
                ("micro", not micro_ok),
            ]
            bad = [name for name, is_bad in violations if is_bad]
            if not bad:
                verdict = "ACCEPT"
            elif (not tag_ok) or (not micro_ok):
                verdict = "SOFT-CHALLENGE"
            else:
                verdict = "REJECT"

            rec = {
                "ts_ns": ts_ns,
                "probe_id": probe_id,
                "counter": counter,
                "sensor": sensor,
                "within": within,
                "counter_ok": counter_ok,
                "sensor_ok": sensor_ok,
                "tag_ok": tag_ok,
                "micro_score": round(float(micro_score), 3),
                "verdict": verdict,
            }
            results.append(rec)
            if verdict != "ACCEPT":
                print(f"[receiver] {verdict} probe=0x{probe_id:X} ctr={counter} tag={tag_ok} micro={rec['micro_score']} bad={bad}")

            state.last_counter = counter
            state.last_sensor = sensor
    except KeyboardInterrupt:
        print("[receiver] Stopped")


def run_attacker(bus: can.BusABC, mode: str, target_id: int, rate_hz: float, duration_s: float) -> None:
    end_time = time.time() + duration_s
    period = 1.0 / max(rate_hz, 0.1)
    try:
        while time.time() < end_time:
            payload = bytes(os.urandom(8))
            msg = can.Message(arbitration_id=target_id, data=payload, is_extended_id=False)
            bus.send(msg)
            print(f"[attacker] {mode} send ts_ns={time.time_ns()} id=0x{target_id:X} data={payload.hex()}")
            time.sleep(period)
    except KeyboardInterrupt:
        pass


@dataclass
class EnrollmentRecord:
    probe_id: int
    window_ms: int
    period_ms: int
    template: list[float]
    std: list[float]
    correlation_thresh: float


def _to_vector(observed_ts_ns: list[int], slot_start_ns: int, window_ms: int) -> np.ndarray:
    win_us = window_ms * 1000
    v = np.zeros(win_us, dtype=np.float32)
    for ts in observed_ts_ns:
        off = int((ts - slot_start_ns) / 1000)
        if 0 <= off < win_us:
            v[off] = 1.0
    return v


def enrollment_worker(
    probe_id: int,
    period_ms: int,
    micro_events_path: str,
    N: int,
    out_dir: str,
    corr_thresh: float,
):
    period_ns = period_ms * 1_000_000
    win_ms = 2  # SLOT_WINDOW_MS (avoid import loop)
    vectors: list[np.ndarray] = []
    rel_lists: list[list[int]] = []

    print(f"[enroll] (local) collecting {N} samples for probe_id=0x{probe_id:X}")
    while len(vectors) < N:
        # Pull latest snapshot from status
        recs = list(StatusServer.results_ref)
        if not recs:
            time.sleep(0.1)
            continue
        # Find newest matching record not yet used
        used = len(vectors)
        r = recs[-1]
        if int(r.get("probe_id", -1)) != probe_id:
            time.sleep(0.1)
            continue
        if not (r.get("within") and r.get("tag_ok")):
            time.sleep(0.1)
            continue

        ts_ns = int(r.get("ts_ns", 0))
        counter = int(r.get("counter", 0))
        slot_start_ns = ts_ns - (ts_ns % period_ns)
        observed_ts = read_micro_events_file(
            path=micro_events_path,
            slot_start_ns=slot_start_ns,
            slot_end_ns=slot_start_ns + period_ns,
            probe_id=probe_id,
            counter=counter,
        )
        vec = _to_vector(observed_ts, slot_start_ns, win_ms)
        if vec.sum() == 0:
            time.sleep(0.05)
            continue
        vectors.append(vec)
        rel_lists.append([int((t - slot_start_ns) // 1000) for t in observed_ts])
        print(f"[enroll] (local) sample {len(vectors)}/{N} ctr={counter} impulses={int(vec.sum())}")

    mean = np.mean(np.vstack(vectors), axis=0).astype(np.float32)
    std = np.std(np.vstack(vectors), axis=0).astype(np.float32)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp / f"{probe_id}.npz", vectors=np.vstack(vectors), rel_lists=np.array(rel_lists, dtype=object))
    rec = EnrollmentRecord(
        probe_id=probe_id,
        window_ms=win_ms,
        period_ms=period_ms,
        template=mean.tolist(),
        std=std.tolist(),
        correlation_thresh=float(corr_thresh),
    )
    (outp / f"{probe_id}.json").write_text(__import__("json").dumps(asdict(rec), indent=2))
    print(f"[enroll] (local) saved {out_dir}/{probe_id}.npz and {out_dir}/{probe_id}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_id", type=int, default=1)
    ap.add_argument("--period_ms", type=int, default=PERIOD_MS)
    ap.add_argument("--dummy_count", type=int, default=DUMMY_COUNT)
    ap.add_argument("--micro_events", type=str, default="micro_events.json")
    ap.add_argument("--micro_threshold", type=float, default=0.5)
    ap.add_argument("--key", type=str, default=os.environ.get("PROBE_KEY", "probe-key"))
    ap.add_argument("--run_attacker", type=str, choices=["none", "can_only"], default="none")
    ap.add_argument("--attack_rate", type=float, default=20.0)
    ap.add_argument("--attack_id", type=lambda x: int(x, 0), default=0x123)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--enroll", type=int, default=0, help="Collect N samples on the fly and export")
    ap.add_argument("--enroll_out", type=str, default="enroll_data", help="Output directory for NPZ/JSON")
    ap.add_argument("--enroll_thresh", type=float, default=0.8)
    ap.add_argument("--asr_soft", type=float, default=0.10, help="Soft success threshold (e.g., 0.05 or 0.10)")
    ap.add_argument("--asr_hard", type=float, default=0.50, help="Hard success threshold (e.g., 0.50 or 0.90)")
    args = ap.parse_args()

    key = args.key.encode() if isinstance(args.key, str) else bytes(args.key)

    # Single shared in-process virtual bus
    bus = can.interface.Bus(bustype="virtual")

    # Receiver HTTP server must start in receiver thread via start_http_server
    t_receiver = threading.Thread(target=run_receiver, args=(bus, args.period_ms, args.micro_events, key, args.host, args.port, args.micro_threshold), daemon=True)
    t_probe = threading.Thread(target=run_probe, args=(bus, args.probe_id, args.period_ms, args.dummy_count, key, args.micro_events), daemon=True)

    t_receiver.start()
    t_probe.start()

    t_attacker = None
    if args.run_attacker != "none":
        t_attacker = threading.Thread(target=run_attacker, args=(bus, args.run_attacker, args.attack_id, args.attack_rate, args.duration), daemon=True)
        t_attacker.start()

    # Optional enrollment in parallel
    t_enroll = None
    if args.enroll > 0:
        t_enroll = threading.Thread(target=enrollment_worker, args=(args.probe_id, args.period_ms, args.micro_events, args.enroll, args.enroll_out, args.enroll_thresh), daemon=True)
        t_enroll.start()

    try:
        # Run until duration expires
        t_end = time.time() + args.duration
        while time.time() < t_end:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    # End-of-run summary
    recs = list(StatusServer.results_ref)
    def _metrics(records):
        tp = sum(1 for r in records if r.get("tag_ok") and r.get("verdict") == "ACCEPT")
        tn = sum(1 for r in records if (not r.get("tag_ok")) and r.get("verdict") == "REJECT")
        fp = sum(1 for r in records if (not r.get("tag_ok")) and r.get("verdict") == "ACCEPT")
        fn = sum(1 for r in records if r.get("tag_ok") and r.get("verdict") == "REJECT")
        total = max(1, len(records))
        det = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        # overhead: fraction of frames that are dummy (heuristic via verdict != ACCEPT or tag_ok False)
        dummy = sum(1 for r in records if not (r.get("verdict") == "ACCEPT" and r.get("tag_ok") and r.get("within")))
        overhead = dummy / total
        return tp, tn, fp, fn, det, fpr, fnr, overhead, total
    tp, tn, fp, fn, det, fpr, fnr, overhead, total = _metrics(recs)
    # Attack Success Rate (ASR): accepted_attack_frames / total_attack_frames
    attack_id = args.attack_id
    atk_total = sum(1 for r in recs if int(r.get("arb_id", r.get("id", -1))) == attack_id)
    atk_accept = sum(
        1
        for r in recs
        if int(r.get("arb_id", r.get("id", -1))) == attack_id and r.get("verdict") == "ACCEPT"
    )
    asr = (atk_accept / atk_total) if atk_total > 0 else 0.0
    if asr >= args.asr_hard:
        asr_label = "HIGHLY SUCCESSFUL / CATASTROPHIC"
    elif asr >= args.asr_soft:
        asr_label = "PARTIALLY SUCCESSFUL"
    else:
        asr_label = "UNSUCCESSFUL"
    print("\n[local_demo] Summary:")
    print(f"- count={total} TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"- detection_rate={det:.3f} FPR={fpr:.3f} FNR={fnr:.3f} overhead={overhead:.3f}")
    print(f"- attack_id=0x{attack_id:X} atk_total={atk_total} atk_accept={atk_accept} ASR={asr:.3f} -> {asr_label}")
    print("[local_demo] Done. Receiver continues serving /status until process exits.")


if __name__ == "__main__":
    main()
