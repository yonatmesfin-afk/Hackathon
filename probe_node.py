"""Probe node that emits real frames per slot and dummy frames between slots.

This implements a Combined Temporal + Dummy hybrid:
- For each period, emit micro-gap events (software timestamps; future GPIO toggles can be added)
  within the slot window, then send one authenticated "real" CAN frame.
- Additionally, send a configured number of dummy frames at random offsets in the period
  with plausible payloads and invalid tags.
"""

import argparse
import os
import random
import time
from typing import List
import can

from common import (
    get_bus,
    DEFAULT_ID,
    PERIOD_MS,
    DUMMY_COUNT,
    make_tag,
    get_micro_pattern,
    pack_message,
    now_ns,
)
from micro_sim import write_micro_events_file


def _align_next_slot(start_epoch_ns: int, period_ns: int) -> int:
    """Return the next slot epoch (ns) >= now aligned to period."""
    n = now_ns()
    if start_epoch_ns <= 0:
        # Align to next multiple of period from now
        return ((n // period_ns) + 1) * period_ns
    # If provided start is in the past, advance to the next multiple
    if n > start_epoch_ns:
        delta = (n - start_epoch_ns)
        k = (delta + period_ns - 1) // period_ns
        return start_epoch_ns + k * period_ns
    return start_epoch_ns


def _sleep_until_epoch(target_epoch_ns: int, base_epoch_ns: int, base_perf_ns: int) -> None:
    """Sleep until an absolute epoch target using perf_counter for precision."""
    while True:
        now_perf = time.perf_counter_ns()
        est_epoch = base_epoch_ns + (now_perf - base_perf_ns)
        remaining_ns = target_epoch_ns - est_epoch
        if remaining_ns <= 0:
            return
        # Sleep most of the remaining time; loop tight for final few 100us
        time.sleep(max(0.0, (remaining_ns - 200_000) / 1e9))


def _simulate_sensor_value() -> int:
    """Return a plausible sensor reading (e.g., temperature * 10 in 0..255)."""
    temp_c = random.uniform(20.0, 30.0)  # 20.0°C .. 30.0°C
    return int(temp_c * 10) & 0xFF


def _build_real_payload(probe_id: int, sensor_val: int) -> List[int]:
    """Build a 5-byte payload for real frames (fits 8-byte frame with counter+tag)."""
    return [
        probe_id & 0xFF,
        (probe_id >> 8) & 0xFF,
        sensor_val & 0xFF,
        random.randint(0, 255),  # noise/jitter byte
        0x00,  # reserved
    ]


def _build_dummy_payload() -> List[int]:
    """Build a plausible dummy 5-byte payload."""
    return [
        random.randint(0, 255),
        random.randint(0, 255),
        _simulate_sensor_value(),
        random.randint(0, 255),
        0xFF,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=lambda x: int(x, 0), default=DEFAULT_ID, help="CAN arbitration ID")
    parser.add_argument("--probe_id", type=int, default=0x1234, help="Logical probe identifier")
    parser.add_argument("--period_ms", type=int, default=PERIOD_MS, help="Slot period in ms")
    parser.add_argument("--dummy_count", type=int, default=DUMMY_COUNT, help="Dummy frames per period")
    parser.add_argument("--slot_start_epoch_ns", type=int, default=-1, help="Epoch ns for first slot (align if omitted)")
    parser.add_argument("--key", type=str, default=os.environ.get("PROBE_KEY", "probe-key"), help="Auth key (bytes string)")
    args = parser.parse_args()

    key = args.key.encode() if isinstance(args.key, str) else bytes(args.key)
    dummy_key = b"dummy-key"

    period_ns = int(args.period_ms * 1_000_000)
    next_slot_epoch_ns = _align_next_slot(args.slot_start_epoch_ns, period_ns)

    base_epoch_ns = time.time_ns()
    base_perf_ns = time.perf_counter_ns()

    counter = 0  # 8-bit counter
    slot_index = 0  # increments each period

    with get_bus() as bus:
        print(f"[probe] id=0x{args.id:X} probe_id=0x{args.probe_id:X} period={args.period_ms}ms dummy_count={args.dummy_count} on {bus.channel_info}")
        try:
            while True:
                # Sleep until the exact start of the slot
                _sleep_until_epoch(next_slot_epoch_ns, base_epoch_ns, base_perf_ns)

                # Reference perf time for in-slot precise scheduling
                slot_perf_start = time.perf_counter_ns()
                cur_slot_start_ns = next_slot_epoch_ns

                # Real frame content
                counter = (counter + 1) & 0xFF
                sensor_val = _simulate_sensor_value()
                payload = _build_real_payload(args.probe_id, sensor_val)
                # Tag over payload || counter || slot_index for receiver compatibility
                tag_data = bytes(payload) + counter.to_bytes(1, 'big') + int(slot_index).to_bytes(4, 'big')
                tag = make_tag(key, tag_data)

                # Micro-gap pattern (software timestamp logging; placeholder for GPIO toggles)
                micro_offsets_us = get_micro_pattern(args.probe_id, counter)
                for off_us in micro_offsets_us:
                    target_perf = slot_perf_start + off_us * 1_000
                    # Precise sleep until target within the slot
                    while True:
                        nowp = time.perf_counter_ns()
                        if nowp >= target_perf:
                            break
                        time.sleep(max(0.0, (target_perf - nowp) / 1e9))
                    event_epoch_ns = time.time_ns()
                    print(f"[probe] ts_ns={event_epoch_ns} type=micro_gap offset_us={off_us}")
                    # TODO: Hardware micro-gap could toggle a GPIO here for oscilloscope capture.

                # Also emit micro-gap events to shared file for receiver consumption
                write_micro_events_file(
                    path=os.environ.get("MICRO_EVENTS_PATH", "micro_events.json"),
                    probe_id=args.probe_id,
                    counter=counter,
                    slot_start_ns=cur_slot_start_ns,
                    offsets_us=micro_offsets_us,
                    mode=os.environ.get("MICRO_MODE", "noisy"),
                    sigma_us=float(os.environ.get("MICRO_SIGMA_US", "30.0")),
                )

                # Schedule dummy frames at random offsets (µs) within the period after slot start
                period_us = args.period_ms * 1000
                dummy_offsets = sorted(random.sample(range(0, max(1, period_us - 1)), k=max(0, args.dummy_count)))

                # Real frame offset: after last micro-gap, keep within small window
                real_offset_us = (micro_offsets_us[-1] + 100) if micro_offsets_us else 0
                real_offset_us = min(real_offset_us, period_us - 500)  # leave a tail margin

                # Build event schedule (offset_us, kind)
                events = [(off, "dummy") for off in dummy_offsets]
                events.append((real_offset_us, "real"))
                events.sort()

                # Execute events in-time
                for off_us, kind in events:
                    target_perf = slot_perf_start + off_us * 1_000
                    while True:
                        nowp = time.perf_counter_ns()
                        if nowp >= target_perf:
                            break
                        time.sleep(max(0.0, (target_perf - nowp) / 1e9))

                    if kind == "real":
                        data_bytes = pack_message(payload, counter, tag)
                        msg = can.Message(arbitration_id=args.id, data=data_bytes, is_extended_id=False)
                        bus.send(msg)
                        print(
                            f"[probe] ts_ns={time.time_ns()} type=real payload={data_bytes.hex()} counter={counter} tag={tag.hex()}"
                        )
                    else:
                        # Dummy frame: plausible payload and invalid tag (different key)
                        d_payload = _build_dummy_payload()
                        d_tag = make_tag(dummy_key, bytes(d_payload))
                        data_bytes = pack_message(d_payload, counter, d_tag)
                        msg = can.Message(arbitration_id=args.id, data=data_bytes, is_extended_id=False)
                        bus.send(msg)
                        print(
                            f"[probe] ts_ns={time.time_ns()} type=dummy payload={data_bytes.hex()} counter={counter} tag={d_tag.hex()}"
                        )

                # Advance to next slot
                next_slot_epoch_ns += period_ns
                slot_index += 1

        except KeyboardInterrupt:
            print("[probe] Stopped")


if __name__ == "__main__":
    main()
