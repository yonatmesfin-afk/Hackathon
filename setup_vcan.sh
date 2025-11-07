#!/usr/bin/env bash
set -euo pipefail

# Usage: sudo bash scripts/setup_vcan.sh
# Creates and enables a virtual CAN interface named vcan0

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Error: $1 is required but not installed" >&2; exit 1; }
}

require_cmd ip
require_cmd modprobe

# Install can-utils if available (Debian/Ubuntu)
if command -v apt >/dev/null 2>&1; then
  echo "[info] Installing can-utils (if not present)"
  sudo apt update -y && sudo apt install -y can-utils || true
fi

# Load kernel modules (idempotent)
echo "[info] Loading kernel modules: can, can_raw, vcan"
sudo modprobe can || true
sudo modprobe can_raw || true
sudo modprobe vcan || true

# Create vcan0 if it doesn't exist
if ip link show vcan0 >/dev/null 2>&1; then
  echo "[info] vcan0 already exists"
else
  echo "[info] Creating vcan0"
  sudo ip link add dev vcan0 type vcan
fi

# Bring it up
echo "[info] Bringing vcan0 up"
sudo ip link set up vcan0

# Show interface
ip -details link show vcan0

echo "[done] vcan0 is ready"
