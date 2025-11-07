import os
import time
import hmac
import hashlib
from typing import List, Tuple
import can
import sys

# Runtime configuration for CAN interface (preserved for existing scripts)
CHANNEL: str = os.environ.get("CAN_CHANNEL", "vcan0")
BUSTYPE: str = os.environ.get("CAN_BUSTYPE", "socketcan")
DEFAULT_ID: int = int(os.environ.get("CAN_DEFAULT_ID", "0x123"), 16)


def get_bus() -> can.BusABC:
    """Return a CAN bus instance using environment-driven channel and bustype.

    On platforms without SocketCAN (e.g., macOS/Windows), falls back to the
    'virtual' backend if BUSTYPE is 'socketcan' and creation fails. To explicitly
    choose a backend, set CAN_BUSTYPE (e.g., 'virtual').
    """
    try:
        return can.interface.Bus(channel=CHANNEL, bustype=BUSTYPE)
    except OSError as e:
        # Common on macOS: socketcan not supported -> fall back to in-process virtual bus
        if BUSTYPE == "socketcan":
            try:
                return can.interface.Bus(bustype="virtual")
            except Exception:
                raise e
        raise


# Timing and message layout constants
PERIOD_MS: int = 100
SLOT_WINDOW_MS: int = 2
TAG_BYTES: int = 2
COUNTER_BYTES: int = 1
DUMMY_COUNT: int = 2


def make_tag(key: bytes, payload_bytes: bytes) -> bytes:
    """Compute HMAC-SHA256 over the payload and return a truncated tag.

    Args:
        key: Secret key bytes used for HMAC.
        payload_bytes: The message payload bytes to authenticate.

    Returns:
        Truncated tag of length TAG_BYTES.
    """
    mac = hmac.new(key, payload_bytes, hashlib.sha256).digest()
    return mac[:TAG_BYTES]


def verify_tag(key: bytes, payload_bytes: bytes, tag: bytes) -> bool:
    """Verify a truncated HMAC tag for the given payload.

    Uses constant-time comparison to avoid timing side-channels.

    Args:
        key: Secret key bytes.
        payload_bytes: The original payload bytes.
        tag: The provided truncated tag to verify.

    Returns:
        True if valid, False otherwise.
    """
    expected = make_tag(key, payload_bytes)
    return hmac.compare_digest(expected, tag[:TAG_BYTES])


def pack_message(payload: List[int], counter: int, tag: bytes) -> bytes:
    """Pack a CAN message payload with counter and tag into bytes.

    The packed byte layout fits in a classic 8-byte CAN frame:
        [payload ..] [counter (COUNTER_BYTES)] [tag (TAG_BYTES)]

    Args:
        payload: List of byte values (0-255). Length must be <= 8 - COUNTER_BYTES - TAG_BYTES.
        counter: Monotonic counter value to encode in COUNTER_BYTES (0-255 when COUNTER_BYTES==1).
        tag: Authentication tag bytes; only TAG_BYTES are used.

    Returns:
        Byte string representing the message data field.

    Raises:
        ValueError: If lengths/values are out of bounds.
    """
    if any(not (0 <= b <= 0xFF) for b in payload):
        raise ValueError("payload elements must be 0..255")

    max_payload = 8 - COUNTER_BYTES - TAG_BYTES
    if len(payload) > max_payload:
        raise ValueError(f"payload too long: max {max_payload} bytes")

    max_counter = (1 << (8 * COUNTER_BYTES)) - 1
    if not (0 <= counter <= max_counter):
        raise ValueError(f"counter out of range for {COUNTER_BYTES} bytes")

    counter_bytes = counter.to_bytes(COUNTER_BYTES, byteorder="big", signed=False)
    tag_trunc = (tag or b"")[:TAG_BYTES]
    if len(tag_trunc) != TAG_BYTES:
        raise ValueError(f"tag must be at least {TAG_BYTES} bytes")

    packed = bytes(payload) + counter_bytes + tag_trunc
    return packed


def unpack_message(msg_bytes: bytes) -> Tuple[List[int], int, bytes]:
    """Unpack bytes into (payload, counter, tag) according to the packing layout.

    Args:
        msg_bytes: Raw message bytes (usually up to 8 bytes for classic CAN).

    Returns:
        A tuple (payload, counter, tag) where:
          - payload is a list of ints 0..255
          - counter is an int decoded from COUNTER_BYTES
          - tag is TAG_BYTES of bytes

    Raises:
        ValueError: If msg_bytes is too short to contain counter and tag.
    """
    if len(msg_bytes) < COUNTER_BYTES + TAG_BYTES:
        raise ValueError("message too short to contain counter and tag")

    payload_len = len(msg_bytes) - COUNTER_BYTES - TAG_BYTES
    payload = list(msg_bytes[:payload_len])
    counter_start = payload_len
    counter_end = counter_start + COUNTER_BYTES
    counter = int.from_bytes(msg_bytes[counter_start:counter_end], byteorder="big", signed=False)
    tag = msg_bytes[counter_end:counter_end + TAG_BYTES]
    return payload, counter, tag


def now_ns() -> int:
    """Return the current UNIX time in nanoseconds."""
    return time.time_ns()


def within_slot(timestamp_ns: int, slot_start_ns: int, window_ms: int) -> bool:
    """Check if timestamp is within [slot_start_ns, slot_start_ns + window_ms) in ns.

    Args:
        timestamp_ns: Time to test (ns).
        slot_start_ns: Slot start time (ns).
        window_ms: Window size in milliseconds.

    Returns:
        True if timestamp is within the window starting at slot_start_ns.
    """
    window_ns = int(window_ms * 1_000_000)
    return slot_start_ns <= timestamp_ns < (slot_start_ns + window_ns)


def get_micro_pattern(probe_id: int, counter: int) -> List[int]:
    """Deterministically derive micro-gap offsets (microseconds) for a probe/counter.

    This pseudo-PUF uses SHA-256(probe_id||counter) to derive DUMMY_COUNT offsets
    within the slot window. It is deterministic (no secrets) and bounded to the
    configured time window (SLOT_WINDOW_MS).

    Args:
        probe_id: Integer identifier of the probing sequence or arbitration ID.
        counter: Monotonic counter value associated with the probe.

    Returns:
        List of DUMMY_COUNT non-negative integers (microseconds) within the slot window.
    """
    seed = f"{probe_id}:{counter}".encode()
    digest = hashlib.sha256(seed).digest()

    # Maximum microseconds inside the slot window
    max_us = max(0, SLOT_WINDOW_MS * 1000 - 1)

    offsets: List[int] = []
    # Use digest bytes to construct DUMMY_COUNT offsets
    for i in range(DUMMY_COUNT):
        # Take 2 bytes per offset for enough spread
        hi = digest[2 * i]
        lo = digest[2 * i + 1]
        val = ((hi << 8) | lo)
        offsets.append(int(val % (max_us + 1)))

    offsets.sort()
    return offsets
