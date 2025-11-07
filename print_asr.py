#!/usr/bin/env python3
"""Print attack success rate (percent) from receiver /status.

ASR = accepted_attack_frames / total_attack_frames * 100

Usage examples:
  python src/print_asr.py --id 0x123
  python src/print_asr.py --host 127.0.0.1 --port 8081 --id 0x123 --timeout 5

Prints only the numeric percentage (e.g., "12.3").
"""
from __future__ import annotations

import argparse
import json
from urllib import request, error as urlerror


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--id", dest="attack_id", type=lambda x: int(x, 0), required=True)
    ap.add_argument("--timeout", type=float, default=2.0)
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/status"
    try:
        with request.urlopen(url, timeout=float(args.timeout)) as resp:
            body = resp.read()
        records = json.loads(body.decode("utf-8"))
    except Exception:
        # Could not fetch -> print 0.0 so caller gets a number
        print("0.0")
        return

    atk_total = 0
    atk_accept = 0
    for r in records:
        rid = int(r.get("arb_id", r.get("id", -1)))
        if rid != args.attack_id:
            continue
        atk_total += 1
        if r.get("verdict") == "ACCEPT":
            atk_accept += 1

    asr = (atk_accept / atk_total) * 100.0 if atk_total > 0 else 0.0
    # Print only numeric percentage
    print(f"{asr:.1f}")


if __name__ == "__main__":
    main()
