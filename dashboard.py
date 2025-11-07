"""Streamlit dashboard for CAN prototype (Combined Temporal + Dummy).

Features:
- Live CAN decisions (last 50): time, probe_id, type (derived), payload (n/a), counter, tag_valid, micro_match, decision.
- Micro-gap waveform: template vs observed for selected probe/counter.
- Metrics: accept/reject rates, tag valid rate, micro avg, throughput.
- Controls: set global dummy_count, micro_correlation_threshold, trigger soft-challenge; writes controls.json.

Notes:
- Data source: receiver_node /status endpoint.
- Observed waveform: read from micro_events.json to reconstruct per-slot impulses.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
import streamlit as st

from common import SLOT_WINDOW_MS
from micro_sim import read_micro_events_file


# Sidebar configuration
st.set_page_config(page_title="CAN Dashboard", layout="wide")
st.title("CAN Dashboard")

receiver_url = st.sidebar.text_input("Receiver status URL", value="http://127.0.0.1:8081/status")
poll_interval = st.sidebar.slider("Poll interval (s)", min_value=0.2, max_value=5.0, value=1.0, step=0.1)
max_rows = st.sidebar.slider("Rows to show", min_value=10, max_value=200, value=50, step=10)

micro_events_path = st.sidebar.text_input("Micro events file", value="micro_events.json")
enroll_json_path = st.sidebar.text_input("Enrollment JSON (optional)", value="")

st.sidebar.markdown("---")
st.sidebar.subheader("Controls (writes controls.json)")
global_dummy = st.sidebar.number_input("dummy_count (global)", min_value=0, max_value=10, value=2, step=1)
micro_thresh = st.sidebar.slider("micro_correlation_threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
soft_probe = st.sidebar.text_input("Trigger soft-challenge for probe_id (hex or int)", value="")
apply_controls = st.sidebar.button("Apply Controls")

controls_path = Path("controls.json")
if apply_controls:
    ctrl = {
        "dummy_count": int(global_dummy),
        "micro_correlation_threshold": float(micro_thresh),
        "soft_challenge_probe_id": soft_probe,
        "ts": time.time_ns(),
    }
    controls_path.write_text(json.dumps(ctrl, indent=2))
    st.sidebar.success(f"Wrote {controls_path}")


def fetch_status(url: str) -> List[Dict]:
    try:
        r = requests.get(url, timeout=2.0)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        st.warning(f"Status fetch failed: {e}")
        return []


def derive_type(rec: Dict) -> str:
    # Heuristic: if all checks pass (will be ACCEPT), likely real; else dummy.
    if rec.get("verdict") == "ACCEPT" and rec.get("tag_ok") and rec.get("within"):
        return "real"
    return "dummy"


def to_table_rows(records: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for r in records[-max_rows:]:
        ts_ns = r.get("ts_ns", 0)
        ts = datetime.fromtimestamp(ts_ns / 1e9).strftime("%H:%M:%S.%f")[:-3]
        rows.append({
            "time": ts,
            "probe_id": f"0x{int(r.get('probe_id', 0)):X}",
            "type": derive_type(r),
            "payload": "n/a",  # receiver does not provide payload; can be added later
            "counter": r.get("counter", ""),
            "tag_valid": "y" if r.get("tag_ok") else "n",
            "micro_match": "y" if float(r.get("micro_score", 0.0)) >= 0.5 else "n",
            "decision": r.get("verdict", ""),
        })
    return rows


def plot_waveforms(probe_id: int, counter: int, slot_start_ns: int, period_ns: int, enroll_json: str):
    st.subheader("Micro-gap waveform: Template vs Observed")
    observed_ts = read_micro_events_file(
        path=micro_events_path,
        slot_start_ns=slot_start_ns,
        slot_end_ns=slot_start_ns + period_ns,
        probe_id=probe_id,
        counter=counter,
    )
    win_us = SLOT_WINDOW_MS * 1000
    obs = np.zeros(win_us, dtype=np.float32)
    for t in observed_ts:
        off = int((t - slot_start_ns) / 1000)
        if 0 <= off < win_us:
            obs[off] = 1.0

    template = None
    if enroll_json and os.path.exists(enroll_json):
        try:
            rec = json.loads(Path(enroll_json).read_text())
            if int(rec.get("probe_id", 0)) == probe_id:
                template = np.array(rec.get("template", []), dtype=np.float32)
        except Exception:
            template = None

    cols = st.columns(2)
    with cols[0]:
        st.write("Observed impulses (µs binning)")
        st.line_chart(obs)
    with cols[1]:
        if template is not None and template.size == obs.size:
            st.write("Enrollment template")
            st.line_chart(template)
        else:
            st.info("No enrollment template loaded or mismatch.")


placeholder_table = st.empty()
placeholder_metrics = st.empty()
placeholder_plot = st.empty()

last_records: List[Dict] = []
last_poll = 0.0

while True:
    now = time.time()
    if now - last_poll >= poll_interval:
        last_records = fetch_status(receiver_url)
        last_poll = now

        # Live table
        table_rows = to_table_rows(last_records)
        placeholder_table.dataframe(table_rows, use_container_width=True)

        # Metrics
        if last_records:
            total = len(last_records)
            accept = sum(1 for r in last_records if r.get("verdict") == "ACCEPT")
            reject = sum(1 for r in last_records if r.get("verdict") == "REJECT")
            soft = total - accept - reject
            tag_valid = sum(1 for r in last_records if r.get("tag_ok"))
            avg_micro = sum(float(r.get("micro_score", 0.0)) for r in last_records) / total
            # Throughput estimate based on time span of recent records
            ts_sorted = sorted([r.get("ts_ns", 0) for r in last_records if isinstance(r.get("ts_ns"), int)])
            fps = 0.0
            if len(ts_sorted) >= 2:
                span_s = max(1e-6, (ts_sorted[-1] - ts_sorted[0]) / 1e9)
                fps = total / span_s

            placeholder_metrics.markdown(
                f"""
                - **accept_rate**: {accept/total:.2f}
                - **reject_rate**: {reject/total:.2f}
                - **soft_challenge_rate**: {soft/total:.2f}
                - **tag_valid_rate**: {tag_valid/total:.2f}
                - **avg_micro_score**: {avg_micro:.2f}
                - **throughput_fps**: {fps:.1f}
                - **avg_tag_compute_ms**: n/a
                """
            )

            # Plot for most recent record
            last = last_records[-1]
            probe_id = int(last.get("probe_id", 0))
            counter = int(last.get("counter", 0))
            # Reconstruct slot start from nearest lower multiple of period; need period_ms input from user
            period_ms = st.session_state.get("period_ms", 100)
            period_ms = st.sidebar.number_input("Period (ms)", min_value=10, max_value=1000, value=period_ms, step=10)
            st.session_state["period_ms"] = period_ms
            period_ns = period_ms * 1_000_000
            ts_ns = int(last.get("ts_ns", 0))
            slot_start_ns = ts_ns - (ts_ns % period_ns)

            with placeholder_plot.container():
                plot_waveforms(probe_id, counter, slot_start_ns, period_ns, enroll_json_path)

    time.sleep(0.05)
