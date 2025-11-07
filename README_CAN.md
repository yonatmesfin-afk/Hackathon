# CAN Bus Prototype (Linux/WSL)

This project sets up a Python dev environment and a virtual CAN interface `vcan0` for local testing.

## 1) Create and enable virtual CAN interface `vcan0`

Native Linux or WSL2 Ubuntu (run in the Linux shell):

```bash
# Make script executable and run it with sudo
chmod +x scripts/setup_vcan.sh
sudo bash scripts/setup_vcan.sh
```

Manual commands (equivalent):
```bash
sudo apt update && sudo apt install -y can-utils
sudo modprobe can
sudo modprobe can_raw
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
ip link show vcan0
```

WSL2 notes:
- Requires WSL2 (not WSL1). Check with: `wsl.exe -l -v`
- Use an Ubuntu/Debian distro. Run the commands above inside WSL.
- Kernel modules are supported in WSL2; if `modprobe vcan` fails after an update, restart Windows.

## 2) Python environment

Create a virtualenv and install packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Contents of `requirements.txt` (already provided):
```
python-can==4.3.1
numpy==1.26.4
scipy==1.11.4
streamlit==1.36.0
scikit-learn==1.3.2
```

## 3) Test send/receive on vcan0

Open two terminals (with the venv activated):

Terminal A (receiver):
```bash
python src/receiver.py
```

Terminal B (sender):
```bash
python src/sender.py
```

You should see a message appear in Terminal A.

Alternative quick test with can-utils:
```bash
candump vcan0
# in another terminal
cansend vcan0 123#0102030405060708
```

## Common pitfalls

- Permissions: If you see permission errors, run the setup script with `sudo` and ensure the interface exists: `ip link show vcan0`.
- WSL2 vs WSL1: Must be WSL2. Convert with: `wsl --set-version <YourDistro> 2` (run in Windows PowerShell as admin).
- Missing modules: Re-run `sudo modprobe vcan` or reboot if modules fail after updates.
- Python version: These pins work well with Python 3.9–3.12. Verify your Python with `python3 --version`.
- Multiple CAN interfaces: If using a different name, change `channel='vcan0'` in the scripts.
