"""Enrollment tool for Combined Temporal + Dummy hybrid.

Collects N real-frame signatures for a given probe_id, builds a mean micro-pattern
template (1µs resolution over SLOT_WINDOW_MS), computes per-bin stddev, and saves:
- NPZ samples at enroll_data/{probe_id}.npz
- JSON enrollment record at enroll_data/{probe_id}.json with template, std, and thresholds.

Samples use micro-gap timestamps relative to the slot start for jitter robustness.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import can

from common import (
    get_bus,
    PERIOD_MS,
    SLOT_WINDOW_MS,
    unpack_message,
    within_slot,
)
from micro_sim import read_micro_events_file


def align_slot_start(ts_ns: int, period_ns: int) -> int:
    return ts_ns - (ts_ns % period_ns)


def compute_slot_index(slot_start_ns: int, ts_ns: int, period_ns: int) -> int:
    if ts_ns < slot_start_ns:
        return 0
    return (ts_ns - slot_start_ns) // period_ns


def to_vector(observed_ts_ns: List[int], slot_start_ns: int, window_ms: int) -> np.ndarray:
    """Convert observed timestamps to a binary impulse vector at 1µs resolution."""
    win_us = window_ms * 1000
    v = np.zeros(win_us, dtype=np.float32)
    for ts in observed_ts_ns:
        off_us = int((ts - slot_start_ns) / 1000)
        if 0 <= off_us < win_us:
            v[off_us] = 1.0
    return v


@dataclass
class EnrollmentRecord:
    probe_id: int
    window_ms: int
    period_ms: int
    template: List[float]
    std: List[float]
    correlation_thresh: float


def collect_samples(
    probe_id: int,
    N: int,
    period_ms: int,
    micro_events_path: str,
    timeout_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect N vectors and relative-offset sample lists for the given probe_id.

    Returns:
        vectors: (N, win_us) binary vectors
        rel_lists: (N,) dtype=object arrays with lists of us offsets (for debugging)
    """
    period_ns = period_ms * 1_000_000
    vectors: List[np.ndarray] = []
    rel_lists: List[List[int]] = []
    # Track per-probe slot start based on first observed frame
    slot_base = -1

    print(f"[enroll] Collecting {N} samples for probe_id=0x{probe_id:X}")
    with get_bus() as bus:
        end_time = time.time() + timeout_s
        while len(vectors) < N and time.time() < end_time:
            msg = bus.recv(timeout=0.5)
            if msg is None:
                continue
            ts_ns = time.time_ns()
            try:
                payload, counter, tag = unpack_message(bytes(msg.data))
            except Exception:
                continue
            # Extract probe_id from payload[0:2]
            if len(payload) < 2:
                continue
            pid = payload[0] | (payload[1] << 8)
            if pid != probe_id:
                continue

            if slot_base < 0:
                slot_base = align_slot_start(ts_ns, period_ns)

            slot_idx = compute_slot_index(slot_base, ts_ns, period_ns)
            cur_slot_start = slot_base + slot_idx * period_ns

            if not within_slot(ts_ns, cur_slot_start, SLOT_WINDOW_MS):
                continue

            observed_ts = read_micro_events_file(
                path=micro_events_path,
                slot_start_ns=cur_slot_start,
                slot_end_ns=cur_slot_start + period_ns,
                probe_id=probe_id,
                counter=counter,
            )
            vec = to_vector(observed_ts, cur_slot_start, SLOT_WINDOW_MS)
            if vec.sum() == 0:
                # No micro-events found; skip
                continue
            vectors.append(vec)
            rel_lists.append([int((t - cur_slot_start) // 1000) for t in observed_ts])
            print(f"[enroll] sample {len(vectors)}/{N} ctr={counter} impulses={int(vec.sum())}")

    if len(vectors) < N:
        print(f"[enroll] Warning: collected {len(vectors)} < {N}")

    return np.vstack(vectors), np.array(rel_lists, dtype=object)


def compute_template(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean template and stddev across samples."""
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def save_npz(out_dir: Path, probe_id: int, vectors: np.ndarray, rel_lists: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{probe_id}.npz"
    np.savez_compressed(npz_path, vectors=vectors, rel_lists=rel_lists)
    print(f"[enroll] Saved samples to {npz_path}")
    return npz_path


def save_json(out_dir: Path, rec: EnrollmentRecord) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{rec.probe_id}.json"
    json_path.write_text(json.dumps(asdict(rec), indent=2))
    print(f"[enroll] Saved enrollment to {json_path}")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_id", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--period_ms", type=int, default=PERIOD_MS)
    parser.add_argument("--micro_events", type=str, default="micro_events.json")
    parser.add_argument("--timeout", type=float, default=120.0, help="Max seconds to collect N samples")
    parser.add_argument("--out_dir", type=str, default="enroll_data")
    parser.add_argument("--correlation_thresh", type=float, default=0.8)
    parser.add_argument("--export_only", type=str, default="", help="Path to NPZ to export as JSON")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.export_only:
        npz_path = Path(args.export_only)
        if not npz_path.exists():
            raise SystemExit(f"NPZ not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        vectors = data["vectors"]
        mean, std = compute_template(vectors)
        rec = EnrollmentRecord(
            probe_id=int(npz_path.stem),
            window_ms=SLOT_WINDOW_MS,
            period_ms=args.period_ms,
            template=mean.tolist(),
            std=std.tolist(),
            correlation_thresh=float(args.correlation_thresh),
        )
        save_json(out_dir, rec)
        return

    vectors, rel_lists = collect_samples(
        probe_id=args.probe_id,
        N=args.N,
        period_ms=args.period_ms,
        micro_events_path=args.micro_events,
        timeout_s=args.timeout,
    )

    npz_path = save_npz(out_dir, args.probe_id, vectors, rel_lists)
    mean, std = compute_template(vectors)
    rec = EnrollmentRecord(
        probe_id=args.probe_id,
        window_ms=SLOT_WINDOW_MS,
        period_ms=args.period_ms,
        template=mean.tolist(),
        std=std.tolist(),
        correlation_thresh=float(args.correlation_thresh),
    )
    save_json(out_dir, rec)


if __name__ == "__main__":
    main()
