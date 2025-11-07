# Quick-Run Checklist (end-to-end demo)

Follow these steps in separate terminals. Assumes Linux with SocketCAN vcan; on macOS, use a Linux VM/container or a CAN adapter.

- **Terminal 0 — Enable vcan0 (once per boot)**
```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
ip -details -statistics link show vcan0
```
Expected:
- Link shows vcan0 UP, qdisc noqueue.

- **Terminal 1 — Probe (emit real + dummy frames, and micro-events)**
```bash
export MICRO_EVENTS_PATH=micro_events.json
export MICRO_MODE=noisy
export MICRO_SIGMA_US=30.0
python src/probe_node.py --probe_id 1 --period_ms 100 --dummy_count 2
```
Expected logs:
- `[probe] ts_ns=... type=micro_gap offset_us=...`
- `[probe] ts_ns=... type=real payload=... counter=... tag=...`

- **Terminal 2 — Enrollment (collect N=100 samples for probe_id=1)**
```bash
python src/train_enroll.py --probe_id 1 --N 100 --period_ms 100 \
  --micro_events micro_events.json --timeout 120 --out_dir enroll_data \
  --correlation_thresh 0.8
```
Expected:
- `[enroll] sample 1/100 ctr=... impulses=2`
- Outputs: `enroll_data/1.npz`, `enroll_data/1.json`

- **Terminal 3 — Receiver (with enrollment loaded)**
```bash
python src/receiver_node.py --period_ms 100 \
  --micro_events micro_events.json \
  --enroll_json enroll_data/1.json \
  --verbose
```
Expected:
- Startup: status URL and listening notice
- Normal frames: `ACCEPT ... tag=True micro≈0.8–1.0 ...`

- **Terminal 4 — Dashboard (Streamlit UI)**
```bash
streamlit run src/dashboard.py
```
Sidebar:
- Receiver URL: `http://127.0.0.1:8081/status`
- Micro events file: `micro_events.json`
- Enrollment JSON: `enroll_data/1.json`

- **Terminal 5 — Attacker (can_only mode)**
```bash
python src/attacker_node.py --mode can_only --target_id 0x123 \
  --injection_rate 20 --duration 30
```
Expected:
- Attacker logs sends; receiver shows `SOFT-CHALLENGE`/`REJECT` for attacker frames.
- Dashboard reject_rate rises; micro_match becomes `n` for attacker frames.

---

# 3‑Minute Demo Guide (README-ready)

## Problem (30s)
Modern CAN buses lack strong authentication and are vulnerable to spoofing and replay. This prototype adds a lightweight hybrid defense:
- Temporal micro‑gap patterns (PUF‑derived timing “watermark”).
- Tiny per‑frame tags (HMAC, 2 bytes).
Together they improve detection of forged and replayed frames while keeping overhead low.

## Demo sequence (enroll → normal → can_only attack → replay attack → soft‑challenge)
1) Enable vcan0 (see checklist).
2) Start probe (normal traffic; emits micro-events).
3) Run enrollment (N=100) to create `enroll_data/1.json`.
4) Start receiver with `--enroll_json enroll_data/1.json`.
5) Start dashboard and observe metrics and waveforms.
6) Launch `attacker_node.py --mode can_only` to demonstrate detection.
7) Optional: `--mode replay` to replay frames without micro-events.
8) Optional: implement soft‑challenge (nonce + response micro‑pattern + ACK) using provided stubs.

## Screens/slides to show (3)
- **Slide 1 — Problem & Approach**: 3 bullets + diagram of micro‑gaps + tag.
- **Slide 2 — Live Dashboard**: table of last frames, metrics panel, waveform (template vs observed).
- **Slide 3 — Metrics**: show `metrics_summary.csv` key rows and `roc.png` (FPR vs TPR when sweeping threshold).

## Produce CSV metrics and ROC
```bash
python src/metrics_collector.py \
  --receiver_url http://127.0.0.1:8081/status \
  --duration 15 \
  --target_id 0x123 \
  --rate 25 \
  --replay_path traces/replay.jsonl \
  --replay_format json \
  --out_csv metrics_summary.csv \
  --roc_png roc.png \
  --sweep_points 21
```
Outputs:
- `metrics_summary.csv` with TP/TN/FP/FN, detection_rate, FPR/FNR, overhead.
- `roc.png` (ROC‑like curve by sweeping micro_correlation_threshold).

## Limitations & next steps
- File‑based micro‑events are a prototype; real systems should use hardware timers/GPIO capture.
- 2‑byte tags are for demo; increase length for stronger integrity.
- Soft‑challenge protocol stubs provided; integrate to complete challenge‑response.
- Next: apply controls.json live in receiver, add processing time instrumentation, support CAN FD.
