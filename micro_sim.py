"""Micro-gap simulation helpers for probe and receiver.

Features:
- File-backed micro-event emission: append JSON lines keyed by probe_id and counter.
- In-memory queue (best-effort, same-process) for tests.
- Precise and noisy modes (Gaussian jitter with configurable sigma).
"""

from __future__ import annotations

import json
import os
import random
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple


_mem_bus: Dict[Tuple[int, int], Deque[int]] = defaultdict(lambda: deque(maxlen=256))
_mem_lock = threading.Lock()


def write_micro_events_file(
    path: str,
    probe_id: int,
    counter: int,
    slot_start_ns: int,
    offsets_us: Iterable[int],
    mode: str = "precise",
    sigma_us: float = 50.0,
) -> List[int]:
    """Append micro-gap events to a JSON-lines file.

    Each line: {"ts_ns": <int>, "type": "micro_gap", "probe_id": <int>, "counter": <int>}

    Args:
        path: JSON-lines file path.
        probe_id: Logical probe identifier.
        counter: 8-bit counter value for this slot.
        slot_start_ns: Epoch ns for slot start.
        offsets_us: Sequence of intra-slot offsets in microseconds.
        mode: 'precise' for exact timestamps, 'noisy' for Gaussian jitter.
        sigma_us: Stddev of jitter in microseconds when mode=='noisy'.

    Returns:
        List of timestamps (ns) written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out_ts: List[int] = []
    with open(path, "a") as f:
        for off_us in offsets_us:
            jitter_us = random.gauss(0.0, sigma_us) if mode == "noisy" else 0.0
            ts_ns = int(slot_start_ns + (off_us + jitter_us) * 1_000)
            rec = {
                "ts_ns": ts_ns,
                "type": "micro_gap",
                "probe_id": int(probe_id),
                "counter": int(counter & 0xFF),
            }
            f.write(json.dumps(rec) + "\n")
            out_ts.append(ts_ns)
    return out_ts


def read_micro_events_file(
    path: str,
    slot_start_ns: int,
    slot_end_ns: int,
    probe_id: Optional[int] = None,
    counter: Optional[int] = None,
) -> List[int]:
    """Read micro-gap timestamps from file filtered by slot window and optional keys."""
    if not os.path.exists(path):
        return []
    ts_list: List[int] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "micro_gap":
                continue
            if probe_id is not None and int(obj.get("probe_id", -1)) != int(probe_id):
                continue
            if counter is not None and int(obj.get("counter", -1)) != int(counter & 0xFF):
                continue
            ts = obj.get("ts_ns")
            if not isinstance(ts, int):
                continue
            if slot_start_ns <= ts < slot_end_ns:
                ts_list.append(ts)
    return ts_list


def push_micro_events_mem(
    probe_id: int,
    counter: int,
    slot_start_ns: int,
    offsets_us: Iterable[int],
    mode: str = "precise",
    sigma_us: float = 50.0,
) -> List[int]:
    """Push micro-gap events into an in-memory queue keyed by (probe_id, counter)."""
    out_ts: List[int] = []
    with _mem_lock:
        q = _mem_bus[(int(probe_id), int(counter & 0xFF))]
        for off_us in offsets_us:
            jitter_us = random.gauss(0.0, sigma_us) if mode == "noisy" else 0.0
            ts_ns = int(slot_start_ns + (off_us + jitter_us) * 1_000)
            q.append(ts_ns)
            out_ts.append(ts_ns)
    return out_ts


def read_micro_events_mem(
    probe_id: int,
    counter: int,
    slot_start_ns: int,
    slot_end_ns: int,
) -> List[int]:
    """Read and drain in-memory micro-gap events for the given key within the slot window."""
    out: List[int] = []
    with _mem_lock:
        q = _mem_bus.get((int(probe_id), int(counter & 0xFF)))
        if not q:
            return out
        # Non-destructive read: copy those in window
        for ts in list(q):
            if slot_start_ns <= ts < slot_end_ns:
                out.append(ts)
    return out


# Example usage snippets
EXAMPLE_PROBE_SNIPPET = """
from common import get_micro_pattern
from micro_sim import write_micro_events_file

# During probe slot handling
offsets_us = get_micro_pattern(probe_id, counter)
# After computing cur_slot_start_ns for this period
write_micro_events_file(
    path="micro_events.json",
    probe_id=probe_id,
    counter=counter,
    slot_start_ns=cur_slot_start_ns,
    offsets_us=offsets_us,
    mode="noisy",      # or "precise"
    sigma_us=30.0,      # jitter standard deviation in microseconds
)
"""

EXAMPLE_RECEIVER_SNIPPET = """
from micro_sim import read_micro_events_file

observed_ts = read_micro_events_file(
    path="micro_events.json",
    slot_start_ns=cur_slot_start_ns,
    slot_end_ns=cur_slot_start_ns + period_ns,
    probe_id=probe_id,
    counter=counter,
)
# Now pass observed_ts to micro_pattern_matches(...)
"""
