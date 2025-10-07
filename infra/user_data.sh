#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

############## HELPERS ##############
# Keep auto-apt from waking up during provisioning
quiesce_auto_apt() {
  systemctl stop unattended-upgrades apt-daily.service apt-daily-upgrade.service 2>/dev/null || true
  systemctl mask unattended-upgrades apt-daily.service apt-daily-upgrade.service 2>/dev/null || true
}

restore_auto_apt() {
  systemctl unmask unattended-upgrades apt-daily.service apt-daily-upgrade.service 2>/dev/null || true
  systemctl start unattended-upgrades 2>/dev/null || true
}

wait_for_apt() {
  quiesce_auto_apt
  # Wait for dpkg frontend lock and any unattended-upgr process
  for _ in $(seq 1 180); do
    if ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
       && ! pgrep -x unattended-upgr >/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "apt still busy after timeout"; exit 1
}

retry() {  # retry with backoff: retry <cmd...>
  local n=0 max=6
  until "$@"; do
    n=$((n+1)); [ $n -ge $max ] && return 1
    sleep $((2**n))
  done
}

##############        LOG        ##############
set -euxo pipefail
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1
echo "== vLLM setup start $(date) =="


##############    CHECK NVIDIA    ##############
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi missing. Use a GPU DLAMI or preinstall drivers." ; exit 1
fi

if systemctl is-active --quiet docker; then systemctl stop docker; fi

############## RUN APT GET UPDATE ##############
wait_for_apt
retry apt-get -o DPkg::Lock::Timeout=600 -yq update
for p in curl jq gnupg; do
  if ! dpkg -s "$p" >/dev/null 2>&1; then
    wait_for_apt
    retry apt-get -o Dpkg::Use-Pty=0 -o DPkg::Lock::Timeout=600 -yq install "$p"
  fi
done

# NVIDIA Container Toolkit
if [ -r /etc/os-release ]; then
  . /etc/os-release
  distribution="$ID$VERSION_ID"   # e.g., ubuntu20.04 or ubuntu22.04
else
  distribution="ubuntu20.04"
fi

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

wait_for_apt
retry apt-get -o DPkg::Lock::Timeout=600 -yq update

NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
# Installing toolkit:
wait_for_apt
retry apt-get -o Dpkg::Use-Pty=0 -o DPkg::Lock::Timeout=600 -yq install \
  nvidia-container-toolkit=$NVIDIA_CONTAINER_TOOLKIT_VERSION \
  nvidia-container-toolkit-base=$NVIDIA_CONTAINER_TOOLKIT_VERSION \
  libnvidia-container-tools=$NVIDIA_CONTAINER_TOOLKIT_VERSION \
  libnvidia-container1=$NVIDIA_CONTAINER_TOOLKIT_VERSION

wait_for_apt
retry apt-get -o DPkg::Lock::Timeout=600 -yq update

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Ensure daemon.json exists
[ -s /etc/docker/daemon.json ] || echo '{}' | sudo tee /etc/docker/daemon.json >/dev/null

# Merge: add nvidia runtime + default + log opts atomically
tmpfile="$(mktemp)"
sudo jq '
  .runtimes = (.runtimes // {}) |
  .runtimes.nvidia = {"path":"nvidia-container-runtime","runtimeArgs":[]} |
  ."default-runtime" = "nvidia" |
  ."log-driver" = "json-file" |
  ."log-opts" = (.["log-opts"] // {}) |
  ."log-opts"."max-size" = "50m" |
  ."log-opts"."max-file" = "3"
' /etc/docker/daemon.json > "$tmpfile" && sudo mv "$tmpfile" /etc/docker/daemon.json

sudo systemctl restart docker

# HF cache persists weights
mkdir -p /opt/hf-cache
chmod 777 /opt/hf-cache

restore_auto_apt || true
