# CAN Prototype: Combined Temporal + Dummy (Hybrid)

This folder contains a runnable skeleton for experimenting with a hybrid Combined Temporal + Dummy approach over a virtual CAN bus (`vcan0`).

## Repository skeleton

- src/common.py — Shared constants and a helper to open a CAN bus (env-driven `CHANNEL`/`BUSTYPE`).
- src/probe_node.py — Probing node that periodically transmits frames at a fixed rate to stimulate the bus.
- src/receiver_node.py — Receiver node that continuously listens to the bus and prints messages.
- src/attacker_node.py — Attacker node that injects frames using a Dummy payload pattern with optional Temporal jitter.
- src/metrics_collector.py — Simple CSV logger that records received frames and metadata for later analysis.
- src/train_enroll.py — Placeholder training/enrollment step that writes a hybrid model artifact from collected data.
- src/dashboard.py — Streamlit dashboard to visualize recent frames and basic bus activity.

## Prerequisites

- Set up `vcan0` (see README_CAN.md) and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run components concurrently (open separate terminals)

All terminals should have the virtualenv activated and `vcan0` up.

Terminal 1 — Receiver:
```bash
python src/receiver_node.py
```

Terminal 2 — Probe node (10 Hz, default ID 0x123):
```bash
python src/probe_node.py --rate_hz 10 --payload 0102030405060708
```

Terminal 3 — Attacker node (Dummy mode at 25 Hz):
```bash
python src/attacker_node.py --mode dummy --rate_hz 25
```

Terminal 4 — Metrics collector (CSV):
```bash
python src/metrics_collector.py --out metrics.csv
```

Terminal 5 — Dashboard:
```bash
streamlit run src/dashboard.py
```

Optional — Train/enroll hybrid artifact from collected metrics:
```bash
python src/train_enroll.py --data metrics.csv --out models/hybrid_model.json
```

Notes:
- Modify interface via env vars: `CAN_CHANNEL=vcan0 CAN_BUSTYPE=socketcan` when running.
- Use different arbitration IDs with `--id 0x123` if you want to segment traffic.
