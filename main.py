#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable or "python"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run local_demo and optionally print only the ASR percent")
    ap.add_argument("--probe_id", type=int, default=1)
    ap.add_argument("--period_ms", type=int, default=100)
    ap.add_argument("--dummy_count", type=int, default=2)
    ap.add_argument("--attack_mode", choices=["can_only", "replay", "precise_guess", "none"], default="can_only")
    ap.add_argument("--attack_id", type=lambda x: int(x, 0), default=0x123)
    ap.add_argument("--injection_rate", type=float, default=20.0)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--percent_only", action="store_true", help="Print only ASR percent at the end")
    ap.add_argument("--label", action="store_true", help="When used with --percent_only, print 'ASR=XX.X%' instead of just the number")
    args = ap.parse_args()

    # Build local_demo command (single process, no UDP dependency)
    demo_cmd = [
        PY,
        os.path.join(ROOT, "local_demo.py"),
        "--probe_id",
        str(args.probe_id),
        "--period_ms",
        str(args.period_ms),
        "--dummy_count",
        str(args.dummy_count),
        "--duration",
        str(args.duration),
        "--attack_id",
        hex(args.attack_id),
    ]
    if args.attack_mode == "can_only":
        demo_cmd += ["--run_attacker", "can_only", "--attack_rate", str(args.injection_rate)]

    # Run and capture output, then print/parse
    proc = subprocess.run(
        demo_cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout

    if args.percent_only:
        # Parse a line containing 'ASR=XX.X%'
        percent = None
        for line in output.splitlines():
            if "ASR=" in line and "%" in line:
                try:
                    frag = line.split("ASR=", 1)[1]
                    num = frag.split("%", 1)[0]
                    percent = float(num)
                    break
                except Exception:
                    continue
        val = percent if percent is not None else 0.0
        if args.label:
            print(f"ASR={val:.1f}%")
        else:
            print(f"{val:.1f}")
    else:
        # Print the demo's full output
        print(output, end="")


if __name__ == "__main__":
    main()
