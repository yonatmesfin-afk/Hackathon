"""Metrics collector and experiment orchestrator.

Runs three experiments (baseline, can_only attacker, replay attacker) for a fixed
duration T and computes:
- true positives (ACCEPT with tag_ok)
- true negatives (REJECT with not tag_ok)
- false positives (ACCEPT with not tag_ok)
- false negatives (REJECT with tag_ok)
- average processing time per message (µs) [not available in status -> reported as NaN]
- bus overhead (% frames classified as dummy by heuristic)

Also includes a helper to sweep micro_correlation_threshold and compute ROC-like curves.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests


def fetch_status(url: str) -> List[Dict]:
    try:
        r = requests.get(url, timeout=2.0)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def dedup_records(records: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in records:
        key = (r.get("ts_ns"), r.get("probe_id"), r.get("counter"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def classify_type(rec: Dict) -> str:
    # Heuristic: ACCEPT+tag_ok+within => real, else dummy
    if rec.get("verdict") == "ACCEPT" and rec.get("tag_ok") and rec.get("within"):
        return "real"
    return "dummy"


def compute_metrics(records: List[Dict]) -> Dict[str, float]:
    # Ground truth proxy: tag_ok indicates legitimate origin (probe has key)
    tp = sum(1 for r in records if r.get("tag_ok") and r.get("verdict") == "ACCEPT")
    tn = sum(1 for r in records if not r.get("tag_ok") and r.get("verdict") == "REJECT")
    fp = sum(1 for r in records if not r.get("tag_ok") and r.get("verdict") == "ACCEPT")
    fn = sum(1 for r in records if r.get("tag_ok") and r.get("verdict") == "REJECT")

    total = max(1, len(records))
    det_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    # Overhead: fraction of frames that are dummy (heuristic)
    dummy = sum(1 for r in records if classify_type(r) == "dummy")
    overhead = dummy / total

    # Receiver doesn't expose processing time; report NaN
    avg_proc_us = math.nan

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "detection_rate": det_rate,
        "fpr": fpr,
        "fnr": fnr,
        "avg_proc_us": avg_proc_us,
        "overhead": overhead,
        "count": len(records),
    }


def run_attacker(mode: str, target_id: str, rate: float, duration: float, replay_path: str, replay_format: str) -> subprocess.Popen | None:
    if mode == "baseline":
        return None
    cmd = [sys.executable, "src/attacker_node.py", "--mode", mode, "--target_id", target_id, "--duration", str(duration)]
    if mode == "can_only":
        cmd += ["--injection_rate", str(rate)]
    elif mode == "replay":
        if replay_path:
            cmd += ["--replay_path", replay_path, "--replay_format", replay_format]
        else:
            return None
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def collect_window(receiver_url: str, start_ns: int, end_ns: int) -> List[Dict]:
    recs = fetch_status(receiver_url)
    recs = [r for r in recs if isinstance(r.get("ts_ns"), int) and start_ns <= r["ts_ns"] <= end_ns]
    return dedup_records(recs)


def run_experiment(name: str, receiver_url: str, duration: float, target_id: str, rate: float, replay_path: str, replay_format: str) -> Tuple[str, Dict[str, float]]:
    print(f"[metrics] Running experiment: {name}")
    t_start_ns = time.time_ns()
    proc = run_attacker(name, target_id, rate, duration, replay_path, replay_format)
    time.sleep(duration)
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
    t_end_ns = time.time_ns()
    # Allow receiver to flush
    time.sleep(0.5)
    records = collect_window(receiver_url, t_start_ns, t_end_ns)
    metrics = compute_metrics(records)
    print(f"[metrics] {name}: count={metrics['count']} det={metrics['detection_rate']:.2f} fpr={metrics['fpr']:.2f} fnr={metrics['fnr']:.2f} overhead={metrics['overhead']:.2f}")
    return name, metrics


def write_csv(path: str, rows: List[Tuple[str, Dict[str, float]]]) -> None:
    header = [
        "experiment","count","tp","tn","fp","fn","detection_rate","fpr","fnr","avg_proc_us","overhead"
    ]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        for name, m in rows:
            w.writerow([name, m["count"], m["tp"], m["tn"], m["fp"], m["fn"], f"{m['detection_rate']:.4f}", f"{m['fpr']:.4f}", f"{m['fnr']:.4f}", m["avg_proc_us"], f"{m['overhead']:.4f}"])


def sweep_threshold(records: List[Dict], thresholds: Iterable[float]) -> List[Tuple[float, float, float]]:
    """Return list of (threshold, TPR, FPR). Ground truth by tag_ok.

    We recompute acceptance as if receiver used the provided micro threshold:
    Accept iff original checks except micro were OK and micro_score >= thr.
    """
    points: List[Tuple[float, float, float]] = []
    for thr in thresholds:
        tp=tn=fp=fn=0
        for r in records:
            gt = bool(r.get("tag_ok"))  # ground truth proxy
            within = bool(r.get("within"))
            counter_ok = bool(r.get("counter_ok", True))
            sensor_ok = bool(r.get("sensor_ok", True))
            tag_ok = bool(r.get("tag_ok"))
            micro_score = float(r.get("micro_score", 0.0))
            base_ok = within and counter_ok and sensor_ok and tag_ok
            accept = base_ok and (micro_score >= thr)
            if accept and gt:
                tp += 1
            elif accept and not gt:
                fp += 1
            elif (not accept) and gt:
                fn += 1
            else:
                tn += 1
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        points.append((float(thr), float(tpr), float(fpr)))
    return points


def try_plot_roc(points: List[Tuple[float, float, float]], out_png: str | None) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("[metrics] matplotlib not available; skipping ROC plot. You can pip install matplotlib.")
        return
    xs = [p[2] for p in points]
    ys = [p[1] for p in points]
    plt.figure(figsize=(5,4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC-like (micro threshold sweep)')
    plt.grid(True)
    if out_png:
        plt.savefig(out_png, bbox_inches='tight')
        print(f"[metrics] Saved ROC plot to {out_png}")
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--receiver_url', default='http://127.0.0.1:8081/status')
    p.add_argument('--duration', type=float, default=15.0)
    p.add_argument('--target_id', default='0x123')
    p.add_argument('--rate', type=float, default=25.0)
    p.add_argument('--replay_path', default='')
    p.add_argument('--replay_format', choices=['json','candump','log'], default='json')
    p.add_argument('--out_csv', default='metrics_summary.csv')
    p.add_argument('--roc_png', default='')
    p.add_argument('--sweep_points', type=int, default=11, help='Number of thresholds from 0.0..1.0')
    args = p.parse_args()

    rows: List[Tuple[str, Dict[str, float]]] = []

    for name in ['baseline','can_only','replay']:
        if name == 'replay' and not args.replay_path:
            print('[metrics] Skipping replay (no replay_path provided)')
            continue
        rows.append(run_experiment(name, args.receiver_url, args.duration, args.target_id, args.rate, args.replay_path, args.replay_format))

    write_csv(args.out_csv, rows)

    # Summarize
    print('\n[metrics] Summary:')
    for name, m in rows:
        print(f"- {name}: N={m['count']} det={m['detection_rate']:.3f} FPR={m['fpr']:.3f} FNR={m['fnr']:.3f} overhead={m['overhead']:.3f}")

    # ROC-like sweep using the union of records from all experiments
    start_ns = time.time_ns() - int(args.duration * 1e9)
    all_records = fetch_status(args.receiver_url)
    all_records = dedup_records(all_records)
    thresholds = [i/(args.sweep_points-1) for i in range(args.sweep_points)] if args.sweep_points > 1 else [0.5]
    points = sweep_threshold(all_records, thresholds)
    if args.roc_png:
        try_plot_roc(points, args.roc_png)


if __name__ == '__main__':
    main()
