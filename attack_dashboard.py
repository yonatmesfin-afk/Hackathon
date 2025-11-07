from __future__ import annotations

import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


@dataclass
class Rec:
    ts_ns: int
    arb_id: int
    verdict: str


def fetch_status(url: str, timeout: float = 2.0) -> List[Rec]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        out: List[Rec] = []
        if isinstance(data, list):
            for d in data:
                try:
                    ts_ns = int(d.get("ts_ns", 0))
                    # id may be under 'arb_id' or 'id'
                    arb_id = int(d.get("arb_id", d.get("id", -1)))
                    verdict = str(d.get("verdict", ""))
                    out.append(Rec(ts_ns=ts_ns, arb_id=arb_id, verdict=verdict))
                except Exception:
                    continue
        return out
    except Exception:
        return []


def compute_cumulative_asr(records: List[Rec]) -> Dict[int, Tuple[int, int, float]]:
    by_id: Dict[int, Tuple[int, int]] = defaultdict(lambda: (0, 0))  # total, accept
    for r in records:
        if r.verdict:
            total, acc = by_id[r.arb_id]
            total += 1
            if r.verdict == "ACCEPT":
                acc += 1
            by_id[r.arb_id] = (total, acc)
    return {k: (v[0], v[1], (v[1] / v[0]) if v[0] else 0.0) for k, v in by_id.items()}


def compute_windowed_asr(records: List[Rec], window_s: float) -> Dict[int, float]:
    if not records:
        return {}
    last_ts = max(r.ts_ns for r in records) / 1e9
    cutoff = last_ts - window_s
    by_id: Dict[int, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in records:
        ts_s = r.ts_ns / 1e9
        if ts_s >= cutoff:
            total, acc = by_id[r.arb_id]
            total += 1
            if r.verdict == "ACCEPT":
                acc += 1
            by_id[r.arb_id] = (total, acc)
    return {k: ((acc / total) if total else 0.0) for k, (total, acc) in by_id.items()}


st.set_page_config(page_title="CAN Attack Dashboard", layout="wide")
st.title("CAN Attack Dashboard")

receiver_url = st.sidebar.text_input("Receiver status URL", value="http://127.0.0.1:8081/status")
refresh_ms = int(st.sidebar.slider("Refresh (ms)", min_value=250, max_value=5000, value=1000, step=250))
window_s = float(st.sidebar.slider("Rolling window (s)", min_value=5, max_value=120, value=30, step=5))
max_points = int(st.sidebar.slider("History points per series", min_value=20, max_value=500, value=120, step=10))

st.sidebar.markdown("---")
run = st.sidebar.button("Start / refresh once")

# History buffers per attack id
history_asr: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=max_points))

placeholder_top = st.empty()
placeholder_charts = st.empty()
placeholder_table = st.empty()

while True:
    if not run and refresh_ms <= 0:
        break

    records = fetch_status(receiver_url)
    cum = compute_cumulative_asr(records)
    win = compute_windowed_asr(records, window_s=window_s)

    ids_sorted = sorted(cum.keys())

    with placeholder_top.container():
        cols = st.columns(4)
        if ids_sorted:
            # Show first 4 summary tiles
            for i, attack_id in enumerate(ids_sorted[:4]):
                total, acc, asr = cum[attack_id]
                cols[i].metric(label=f"ID 0x{attack_id:X} ASR%", value=f"{asr*100:.1f}", delta=f"win {win.get(attack_id, 0.0)*100:.1f}")
        else:
            st.info("No attack records yet. Generate traffic with main.py or attacker_node.py.")

    now = time.time()
    for attack_id in ids_sorted:
        history_asr[attack_id].append((now, win.get(attack_id, 0.0) * 100.0))

    with placeholder_charts.container():
        tabs = st.tabs([f"0x{aid:X}" for aid in ids_sorted] or ["ASR"])
        if ids_sorted:
            for tab, aid in zip(tabs, ids_sorted):
                with tab:
                    # rolling ASR timeseries
                    data = list(history_asr[aid])
                    if data:
                        t = np.array([p[0] for p in data])
                        v = np.array([p[1] for p in data])
                        # normalize time to start at 0
                        t0 = t[0]
                        df = pd.DataFrame({"time_s": t - t0, "ASR_%": v})
                        st.line_chart(df.set_index("time_s"))
                    total, acc, asr = cum.get(aid, (0, 0, 0.0))
                    st.caption(f"Cumulative: total={total} accept={acc} ASR={asr*100:.1f}%  |  Window({window_s:.0f}s): {win.get(aid, 0.0)*100:.1f}%")
        else:
            st.empty()

    with placeholder_table.container():
        # Show the last N raw decisions for context
        N = 100
        rows: List[Dict] = []
        for r in records[-N:]:
            rows.append({
                "time": time.strftime("%H:%M:%S", time.localtime(r.ts_ns/1e9)),
                "id": f"0x{r.arb_id:X}",
                "verdict": r.verdict,
            })
        if rows:
            st.subheader("Recent decisions")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)

    if not run:
        st.experimental_rerun()
    else:
        break
