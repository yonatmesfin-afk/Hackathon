"""Receiver node with slotting, tag verification, micro-gap correlation, and HTTP status.

- Listens on vcan0 using python-can.
- Tracks per-probe slot start and last seen counter/sensor value.
- Unpacks frames using common.unpack_message and validates:
  within-slot timing, HMAC tag (payload||counter||slot_index), counter monotonicity,
  and sensor plausibility (max delta per period).
- Reads micro-gap events from a shared micro_events.json file and matches against
  expected patterns via normalized cross-correlation.
- Classifies frames as ACCEPT / SOFT-CHALLENGE / REJECT and exposes a small HTTP
  /status endpoint to retrieve recent results.
"""

import argparse
import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Deque, Dict, List, Tuple

import can
import numpy as np

from common import (
    get_bus,
    PERIOD_MS,
    SLOT_WINDOW_MS,
    make_tag,
    unpack_message,
    within_slot,
    get_micro_pattern,
)
from micro_sim import read_micro_events_file


@dataclass
class ProbeState:
    slot_start_ns: int = -1
    last_counter: int = -1
    last_sensor: int = -1


def align_slot_start(ts_ns: int, period_ns: int) -> int:
    return ts_ns - (ts_ns % period_ns)


def compute_slot_index(slot_start_ns: int, ts_ns: int, period_ns: int) -> int:
    if ts_ns < slot_start_ns:
        return 0
    return (ts_ns - slot_start_ns) // period_ns


 


def micro_pattern_matches(
    probe_id: int,
    counter: int,
    observed_ts_ns: List[int],
    slot_start_ns: int,
    window_ms: int,
    threshold: float,
) -> float:
    """Return normalized cross-correlation score between expected and observed micro-gaps.

    We build binary impulse trains over the slot window at 1µs resolution and compute
    cosine similarity (equivalent to normalized dot product) as the score.
    """
    win_us = window_ms * 1000
    if win_us <= 0:
        return 0.0

    # Expected offsets (us)
    expected_offsets = get_micro_pattern(probe_id, counter)
    exp_vec = np.zeros(win_us, dtype=np.float32)
    for off in expected_offsets:
        if 0 <= off < win_us:
            exp_vec[off] = 1.0

    # Observed offsets (us) relative to slot_start
    obs_vec = np.zeros(win_us, dtype=np.float32)
    for ts in observed_ts_ns:
        off_us = int((ts - slot_start_ns) / 1000)
        if 0 <= off_us < win_us:
            obs_vec[off_us] = 1.0

    # Normalized cross-correlation (dot since equal-length aligned vectors)
    dot = float(np.dot(exp_vec, obs_vec))
    denom = float(np.linalg.norm(exp_vec) * np.linalg.norm(obs_vec))
    score = (dot / denom) if denom > 0 else 0.0
    return score


class StatusServer(BaseHTTPRequestHandler):
    results_ref: Deque[dict] = deque(maxlen=200)

    def do_GET(self):
        if self.path.startswith("/status"):
            body = json.dumps(list(StatusServer.results_ref), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def start_http_server(host: str, port: int, results_ref: Deque[dict]) -> threading.Thread:
    StatusServer.results_ref = results_ref
    server = HTTPServer((host, port), StatusServer)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=0.5, help="CAN receive timeout (s)")
    parser.add_argument("--period_ms", type=int, default=PERIOD_MS)
    parser.add_argument("--key", type=str, default=os.environ.get("PROBE_KEY", "probe-key"))
    parser.add_argument("--micro_events", type=str, default="micro_events.json")
    parser.add_argument("--micro_threshold", type=float, default=0.5)
    parser.add_argument("--max_sensor_delta", type=int, default=8, help="Max allowed delta per period")
    parser.add_argument("--enroll_json", type=str, default="", help="Enrollment JSON for template-based correlation")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    key = args.key.encode() if isinstance(args.key, str) else bytes(args.key)
    period_ns = int(args.period_ms * 1_000_000)

    per_probe: Dict[int, ProbeState] = defaultdict(ProbeState)
    results: Deque[dict] = deque(maxlen=200)
    start_http_server(args.host, args.port, results)
    print(f"[receiver] HTTP status on http://{args.host}:{args.port}/status")

    # Optional enrollment template loading
    enroll_template: Dict[int, np.ndarray] = {}
    enroll_thresh: Dict[int, float] = {}
    if args.enroll_json and os.path.exists(args.enroll_json):
        with open(args.enroll_json, "r") as f:
            rec = json.load(f)
        pid = int(rec.get("probe_id", 0))
        template = np.array(rec.get("template", []), dtype=np.float32)
        if template.size > 0:
            enroll_template[pid] = template
            enroll_thresh[pid] = float(rec.get("correlation_thresh", args.micro_threshold))

    with get_bus() as bus:
        print(f"[receiver] Listening on {bus.channel_info} (Ctrl+C to stop)...")
        try:
            while True:
                msg = bus.recv(timeout=args.timeout)
                if msg is None:
                    continue

                ts_ns = time.time_ns()
                try:
                    payload, counter, tag = unpack_message(bytes(msg.data))
                except Exception as e:
                    print(f"[receiver] unpack error: {e}")
                    continue

                if len(payload) < 3:
                    print("[receiver] payload too short")
                    continue

                probe_id = payload[0] | (payload[1] << 8)
                sensor = payload[2]

                state = per_probe[probe_id]
                if state.slot_start_ns < 0:
                    state.slot_start_ns = align_slot_start(ts_ns, period_ns)

                # Determine current slot
                slot_idx = compute_slot_index(state.slot_start_ns, ts_ns, period_ns)
                cur_slot_start = state.slot_start_ns + slot_idx * period_ns

                # Checks
                within = within_slot(ts_ns, cur_slot_start, SLOT_WINDOW_MS)

                counter_ok = (state.last_counter == -1) or (((state.last_counter + 1) & 0xFF) == counter)

                sensor_ok = (state.last_sensor == -1) or (abs(sensor - state.last_sensor) <= args.max_sensor_delta)

                # Tag: H(payload||counter||slot_index)
                counter_bytes = counter.to_bytes(1, "big")
                slot_bytes = int(slot_idx).to_bytes(4, "big")
                tag_data = bytes(payload) + counter_bytes + slot_bytes
                tag_ok = (len(tag) > 0) and (make_tag(key, tag_data) == tag)

                # Micro-gap correlation
                observed_ts = read_micro_events_file(
                    path=args.micro_events,
                    slot_start_ns=cur_slot_start,
                    slot_end_ns=cur_slot_start + period_ns,
                    probe_id=probe_id,
                    counter=counter,
                )
                # If enrollment template exists for this probe, compute correlation to template
                if probe_id in enroll_template:
                    win_us = SLOT_WINDOW_MS * 1000
                    vec = np.zeros(win_us, dtype=np.float32)
                    for ts in observed_ts:
                        off_us = int((ts - cur_slot_start) / 1000)
                        if 0 <= off_us < win_us:
                            vec[off_us] = 1.0
                    t = enroll_template[probe_id]
                    dot = float(np.dot(vec, t))
                    denom = float(np.linalg.norm(vec) * np.linalg.norm(t))
                    micro_score = (dot / denom) if denom > 0 else 0.0
                    micro_ok = micro_score >= enroll_thresh.get(probe_id, args.micro_threshold)
                else:
                    micro_score = micro_pattern_matches(probe_id, counter, observed_ts, cur_slot_start, SLOT_WINDOW_MS, args.micro_threshold)
                    micro_ok = micro_score >= args.micro_threshold

                # Classification
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
                    "arb_id": int(msg.arbitration_id),
                    "probe_id": probe_id,
                    "counter": counter,
                    "sensor": sensor,
                    "within": within,
                    "counter_ok": counter_ok,
                    "sensor_ok": sensor_ok,
                    "tag_ok": tag_ok,
                    "micro_score": round(micro_score, 3),
                    "verdict": verdict,
                }
                results.append(rec)

                if args.verbose or verdict != "ACCEPT":
                    print(f"[receiver] {verdict} probe=0x{probe_id:X} ctr={counter} sensor={sensor} within={within} tag={tag_ok} micro={rec['micro_score']} bad={bad}")

                # Update state
                state.last_counter = counter
                state.last_sensor = sensor

        except KeyboardInterrupt:
            print("[receiver] Stopped")


if __name__ == "__main__":
    main()
