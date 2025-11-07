# CAN Attack Overview and Viability

This repository includes simple CAN-bus attack modes for experimentation and for measuring a Detection/ASR (Attack Success Rate) pipeline. These attacks are intentionally basic; they are meant to exercise the receiver’s checks and the metrics reporting, not to be state‑of‑the‑art.

## What the receiver defends against
The receiver validates each frame with multiple checks:
- Timing window: frame must arrive within the configured slot window.
- Counter monotonicity: counter must increment properly.
- Sensor plausibility: sensor deltas must stay within a configured bound.
- Authentication tag (HMAC, truncated): payload||counter||slot_index must match.
- Micro‑gap correlation: observed in‑slot micro events must correlate with expected pattern (or an enrolled template).

A frame is classified:
- ACCEPT if all checks pass.
- SOFT‑CHALLENGE if some authenticity signals fail (e.g., tag or micro), but not enough for hard reject.
- REJECT otherwise.

## Implemented attack modes
1) can_only (random forgeries)
   - Sends random payloads for a chosen arbitration ID at a chosen rate.
   - No valid tag, no micro‑pattern knowledge, no counter alignment.
   - Goal: stress naive receivers; realistic receivers should reject these.
   - Viability: Very low vs. a receiver with cryptographic tags and micro timing.

2) replay (payload/tag replay)
   - Replays recorded frames (from JSON or candump) to mimic legitimate traffic.
   - Without synchronizing counters/slots or defeating freshness, strong receivers detect and reject.
   - Viability: Low to moderate depending on target system. Effective only if the target lacks freshness (counters/slots) or uses weak authentication.

3) precise_guess (slot timing guess)
   - Attempts to inject in guessed time slots with mid‑window offsets.
   - Does not have the correct tag or micro signature.
   - Viability: Low on systems with tags/micro checks; timing alone is insufficient.

## Are these attacks “advanced”?
- No. They are purposely simple baselines to validate the receiver’s multi‑signal defenses and end‑to‑end metrics (ASR, FP/FN, etc.).
- Realistic, advanced attacks would require some combination of:
  - Key compromise or side‑channel to forge valid tags.
  - Learning/mimicking per‑probe micro‑patterns or defeating micro correlation.
  - Synchronizing slot index and counter state to satisfy freshness.
  - Payload semantics awareness to keep sensor deltas plausible.
- These baselines are viable in weakly protected environments (no tags/freshness/micro), but are not expected to succeed against the provided receiver configuration.

## How we quantify “success”
Attack Success Rate (ASR) = accepted_attack_frames / total_attack_frames.
- We compute ASR either inside the single‑process demo or by querying the receiver’s /status.
- “Success” means the receiver classified a forged frame as ACCEPT.
- You can choose thresholds: e.g., soft=10%, hard=50% to label the campaign as partially or highly successful.

## How to run attacks and measure ASR
Single‑process (recommended):
- Full summary (includes ASR line):
  ```bash
  python src/main.py --attack_mode can_only --attack_id 0x123 \
    --probe_id 1 --period_ms 100 --dummy_count 2 \
    --injection_rate 20 --duration 15
  ```
- Percentage only (labeled):
  ```bash
  python src/main.py --attack_mode can_only --attack_id 0x123 \
    --probe_id 1 --period_ms 100 --dummy_count 2 \
    --injection_rate 20 --duration 15 --percent_only --label
  ```

Separate processes (expert use):
- Start receiver with /status:
  ```bash
  python src/receiver_node.py --host 127.0.0.1 --port 8081 --verbose
  ```
- Run attacker and compute ASR from /status at the end:
  ```bash
  python src/attacker_node.py --mode can_only --target_id 0x123 \
    --injection_rate 20 --duration 15 \
    --compute_asr --asr_only --quiet \
    --status_host 127.0.0.1 --status_port 8081 --wait_status 30
  ```
- Or print later via the helper:
  ```bash
  python src/print_asr.py --id 0x123
  ```

## Why you often see ASR = 0%
- Random forgeries don’t have a valid tag or micro signature, and they ignore counter freshness and sensor bounds.
- The receiver is designed to reject them, so accepted_this_run stays 0.
- This is expected and demonstrates the multi‑signal defenses are working.

## Viability summary
- can_only: Viable only against weak receivers; expected 0% ASR here.
- replay: Viable if the target lacks freshness/authentication; expected near 0% ASR here.
- precise_guess: Timing alone is insufficient; expected near 0% ASR here.

## Future work ideas (to study stronger adversaries)
- Add a tag‑forgery oracle (simulating key compromise) to evaluate impact on ASR.
- Train a micro‑pattern mimic to raise micro correlation under attack.
- Incorporate payload semantics models to keep sensor deltas plausible.
- Explore time‑synchronized counter spoofing to satisfy freshness.
