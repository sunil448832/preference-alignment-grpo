#!/usr/bin/env bash
# GPU setup for a SageMaker ml.g5.2xlarge (1x NVIDIA A10G, 24GB) -- or any single-GPU EC2-style box.
#
# What it does, in order (each phase is idempotent, rerun freely):
#   1. Diagnose: is there an NVIDIA device on the PCI bus? are we inside a container? is the kernel
#      module already on disk but not loaded? (the usual cause of "nvidia-smi: command not found /
#      couldn't communicate with the NVIDIA driver" on a box that used to work)
#   2. Install the NVIDIA driver + CUDA toolkit from NVIDIA's own repo (Ubuntu 22.04/24.04, Amazon Linux 2023)
#   3. Load the module; if the kernel refuses (nouveau still bound, headers mismatch) tell you to reboot and rerun
#   4. Build a Python venv from requirements.txt and verify torch/bitsandbytes see the GPU
#
# Usage:  bash setup_gpu.sh [--no-vllm] [--no-toolkit] [--no-python] [--venv DIR]
#   --no-vllm     skip vLLM (default installs it first: rollouts and eval use it; big download, pins its own torch)
#   --no-toolkit  skip the ~3GB CUDA toolkit (nvcc). The pinned torch/bitsandbytes wheels bundle their own
#                 CUDA runtime, so the toolkit is only needed if you compile something (flash-attn etc).
#   --no-python   stop after the driver is working
#   --venv DIR    where to create the venv (default: ~/venvs/grpo)
set -euo pipefail

CUDA_TOOLKIT_PKG="cuda-toolkit-12-8"
WITH_VLLM=1
INSTALL_TOOLKIT=1
SETUP_PYTHON=1
VENV="${HOME}/venvs/grpo"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-vllm)    WITH_VLLM=0 ;;
    --no-toolkit) INSTALL_TOOLKIT=0 ;;
    --no-python)  SETUP_PYTHON=0 ;;
    --venv)       VENV="$2"; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mXX  %s\033[0m\n' "$*" >&2; exit 1; }

SUDO=""
if [[ $EUID -ne 0 ]]; then
  command -v sudo >/dev/null || die "not root and no sudo available"
  SUDO="sudo"
fi

gpu_works() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 1. Diagnose
# ---------------------------------------------------------------------------
log "Diagnosing"
. /etc/os-release
echo "OS: ${PRETTY_NAME}   kernel: $(uname -r)   arch: $(uname -m)"

if command -v lspci >/dev/null 2>&1; then
  if ! lspci | grep -qi nvidia; then
    die "No NVIDIA device on the PCI bus. This is not a GPU instance (check the instance type is ml.g5.*), or the GPU is not passed through to this environment."
  fi
  echo "PCI: $(lspci | grep -i nvidia | head -1)"
else
  # lspci is missing on minimal images; /sys is always there
  if ! grep -qil '^0x10de$' /sys/bus/pci/devices/*/vendor 2>/dev/null; then
    die "No NVIDIA device found under /sys/bus/pci. Wrong instance type, or GPU not passed through."
  fi
  echo "PCI: NVIDIA vendor id 0x10de present"
fi

if [[ -f /.dockerenv ]] || grep -qaE 'docker|containerd|kubepods' /proc/1/cgroup 2>/dev/null; then
  if ! gpu_works; then
    die "This is a container (SageMaker Studio space / kernel image) and the GPU is not exposed to it. Kernel drivers cannot be installed from inside a container. Fix on the SageMaker side: pick a GPU instance type for the space, and a GPU-enabled image. Then nvidia-smi works without any install."
  fi
fi

if gpu_works; then
  log "Driver already working"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  # Driver files on disk but module not loaded? Common after a kernel update on a preinstalled image.
  if ls /lib/modules/"$(uname -r)"/kernel/drivers/video/nvidia*.ko* /lib/modules/"$(uname -r)"/updates/dkms/nvidia*.ko* >/dev/null 2>&1 \
     || modinfo nvidia >/dev/null 2>&1; then
    log "NVIDIA module exists on disk but is not loaded; trying to load it"
    $SUDO modprobe nvidia 2>/dev/null || true
    if ! gpu_works && command -v dkms >/dev/null 2>&1; then
      log "modprobe failed; rebuilding via dkms for kernel $(uname -r)"
      $SUDO dkms autoinstall -k "$(uname -r)" || warn "dkms autoinstall failed (see above)"
      $SUDO modprobe nvidia 2>/dev/null || true
    fi
    if gpu_works; then
      log "Driver repaired without reinstall"; nvidia-smi --query-gpu=name,driver_version --format=csv
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 2. Install driver (+ toolkit) if still not working
# ---------------------------------------------------------------------------
if ! gpu_works; then
  log "Installing NVIDIA driver from NVIDIA's CUDA repo"
  case "${ID}-${VERSION_ID}" in
    ubuntu-22.04|ubuntu-24.04)
      REPO_DISTRO="ubuntu${VERSION_ID//./}"
      export DEBIAN_FRONTEND=noninteractive
      $SUDO apt-get update -qq
      $SUDO apt-get install -y -qq wget gnupg build-essential dkms "linux-headers-$(uname -r)" pciutils
      if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
        wget -qO /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${REPO_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
        $SUDO dpkg -i /tmp/cuda-keyring.deb
        $SUDO apt-get update -qq
      fi
      # Ubuntu's stock image sometimes ships nouveau; make sure it never grabs the GPU again after reboot.
      printf 'blacklist nouveau\noptions nouveau modeset=0\n' | $SUDO tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null
      $SUDO update-initramfs -u
      # nvidia-open = open kernel modules, the NVIDIA-recommended path for Turing and newer (A10G is Ampere).
      $SUDO apt-get install -y nvidia-open || $SUDO apt-get install -y cuda-drivers
      if [[ $INSTALL_TOOLKIT -eq 1 ]]; then $SUDO apt-get install -y "$CUDA_TOOLKIT_PKG"; fi
      ;;
    amzn-2023)
      $SUDO dnf install -y dkms "kernel-devel-$(uname -r)" "kernel-modules-extra-$(uname -r)" gcc make pciutils
      $SUDO dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo"
      $SUDO dnf clean expire-cache
      $SUDO dnf module enable -y nvidia-driver:open-dkms 2>/dev/null || true
      $SUDO dnf install -y nvidia-open || $SUDO dnf install -y nvidia-driver
      if [[ $INSTALL_TOOLKIT -eq 1 ]]; then $SUDO dnf install -y "$CUDA_TOOLKIT_PKG"; fi
      ;;
    amzn-2)
      die "Amazon Linux 2 with no working NVIDIA driver. AL2 is end-of-life and NVIDIA's current repos no longer target it. SageMaker notebook instances ship the driver preinstalled on AL2 (the repair step above should have loaded it); if it is genuinely missing, recreate the notebook instance with the 'Amazon Linux 2023' platform and rerun this script."
      ;;
    *)
      die "Unsupported OS '${ID} ${VERSION_ID}'. Supported: Ubuntu 22.04/24.04, Amazon Linux 2023."
      ;;
  esac

  # ---------------------------------------------------------------------------
  # 3. Load it
  # ---------------------------------------------------------------------------
  log "Loading the driver"
  if lsmod | grep -q '^nouveau'; then
    $SUDO rmmod nouveau 2>/dev/null || warn "nouveau is bound to the GPU and cannot be unloaded live"
  fi
  $SUDO modprobe nvidia 2>/dev/null || true
  # nvidia-smi may have landed in /usr/bin only after install; refresh the shell's lookup table
  hash -r
  if ! gpu_works; then
    warn "Driver installed but the kernel module is not loaded yet. Reboot, then rerun this script; it will skip straight to the Python setup."
    echo "    sudo reboot"
    exit 0
  fi
fi

log "GPU is up"
nvidia-smi

# ---------------------------------------------------------------------------
# 4. Python environment
# ---------------------------------------------------------------------------
if [[ $SETUP_PYTHON -eq 0 ]]; then exit 0; fi

log "Python environment at ${VENV}"
# transformers 5.x / trl 1.x need Python >= 3.10. AL2023's default python3 is 3.9, so pick the newest present.
PY=""
for cand in python3.12 python3.11 python3.10; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [[ -z "$PY" ]]; then
  case "$ID" in
    amzn)   $SUDO dnf install -y python3.11 python3.11-pip && PY=python3.11 ;;
    ubuntu) $SUDO apt-get install -y python3-venv python3-pip && PY=python3 ;;
  esac
fi
[[ -n "$PY" ]] || die "no Python >= 3.10 found"
echo "using $($PY --version) at $(command -v $PY)"

if [[ ! -x "${VENV}/bin/python" ]]; then
  mkdir -p "$(dirname "$VENV")"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -q --upgrade pip wheel

if [[ $WITH_VLLM -eq 1 ]]; then
  # vLLM pins an exact torch; install it first so it picks the torch, then the rest resolves around it.
  log "Installing vLLM (this pulls its own torch build)"
  pip install "$(grep -E '^vllm' "${PROJECT_DIR}/requirements.txt" | cut -d'#' -f1 | tr -d ' ')"
fi

log "Installing project requirements"
# requirements.txt lists vllm too; it was installed above (or skipped with --no-vllm), so drop that line here.
grep -vE '^\s*(#|$)' "${PROJECT_DIR}/requirements.txt" | grep -vE '^vllm' > /tmp/reqs_no_vllm.txt
pip install -r /tmp/reqs_no_vllm.txt

# The 4B model is ~8GB of safetensors. SageMaker notebook instances have a small root volume and a
# large EBS volume at ~/SageMaker; point the HF cache there if it exists.
if [[ -d "${HOME}/SageMaker" ]]; then
  HF_HOME_DIR="${HOME}/SageMaker/hf_cache"
  mkdir -p "$HF_HOME_DIR"
  if ! grep -q 'HF_HOME' "${VENV}/bin/activate"; then
    echo "export HF_HOME=${HF_HOME_DIR}" >> "${VENV}/bin/activate"
  fi
  export HF_HOME="$HF_HOME_DIR"
  echo "HF_HOME -> ${HF_HOME_DIR} (added to the venv's activate script)"
fi

log "Verifying from Python"
python - <<'PYEOF'
import torch, transformers, trl, peft
print(f"torch {torch.__version__}  cuda build {torch.version.cuda}  cuda available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "torch cannot see the GPU"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}  {p.total_memory/2**30:.1f} GiB  capability {p.major}.{p.minor}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", (x @ x).isfinite().all().item())
import bitsandbytes as bnb
print(f"bitsandbytes {bnb.__version__}  transformers {transformers.__version__}  trl {trl.__version__}  peft {peft.__version__}")
try:
    import vllm; print(f"vllm {vllm.__version__}")
except ImportError:
    print("vllm: not installed (rollout.backend=\"hf\" and eval --backend hf only)")
PYEOF

log "Done"
cat <<MSG
Activate with:   source ${VENV}/bin/activate
Sanity run:      cd ${PROJECT_DIR} && python -c "import train_grpo_pytorch as t; t.MAX_STEPS=3; t.WARMUP_STEPS=1; t.PROMPTS_PER_STEP=2; t.NUM_GENERATIONS=4; t.MAX_COMPLETION_LENGTH=256; t.train()"
Note: an A10G has 24GB, so the defaults (64 concurrent 2048-token rollouts) will not fit on this box; keep the reduced knobs here.
MSG
