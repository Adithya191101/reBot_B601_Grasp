#!/usr/bin/env bash
# M0 host setup — the only steps that need root on this machine.
# Everything else (docker, nvidia-container-toolkit, group membership,
# workspace) is already in place and verified. Run me with:
#   sudo bash scripts/m0_host_setup.sh
set -euo pipefail

# 1. NVIDIA Isaac ROS apt repository, pinned to release-4.5 (Jazzy / noble)
k="/usr/share/keyrings/nvidia-isaac-ros.gpg"
if [ ! -f "$k" ]; then
  curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key \
    | gpg --dearmor > "$k"
fi
f="/etc/apt/sources.list.d/nvidia-isaac-ros.list"
s="deb [signed-by=$k] https://isaac.download.nvidia.com/isaac-ros/release-4.5 noble main"
touch "$f"
grep -qxF "$s" "$f" || echo "$s" >> "$f"
apt-get update

# 2. The Isaac ROS CLI (the 4.5 replacement for run_dev.sh)
apt-get install -y isaac-ros-cli

# 3. Initialize the dockerized dev environment
isaac-ros init docker

echo "M0 host setup complete. Next (no sudo): isaac-ros activate"
