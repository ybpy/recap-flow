#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export RECAP_RUNTIME=${RECAP_RUNTIME:-/mnt/recap-flow-runtime}
if [[ "${E0_PLATFORM:-}" == pro5000 ]]; then source "$ROOT/e0/environment/pro5000.sh" "${1:-}"; fi
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
export UV_CACHE_DIR="$RECAP_RUNTIME/uv-cache"
export UV_LINK_MODE=copy
export HF_HOME="$RECAP_RUNTIME/cache/huggingface"
export HF_LEROBOT_HOME="$RECAP_RUNTIME/cache/lerobot"
export OPENPI_DATA_HOME=/mnt/models/openpi-cache
export XDG_CACHE_HOME="$RECAP_RUNTIME/cache"
export JAX_COMPILATION_CACHE_DIR="$RECAP_RUNTIME/cache/jax/${E0_PLATFORM:-unspecified}/${CUDA_VISIBLE_DEVICES:-all}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=3
# Fix kernel selection across process restarts; preserve model dtype and optimizer math.
autotune_level=0
if [[ "${E0_PLATFORM:-}" == pro5000 ]]; then autotune_level=1; fi
for flag in --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=$autotune_level --xla_gpu_enable_triton_gemm=false; do
  key=${flag%%=*}
  for existing in ${XLA_FLAGS:-}; do
    if [[ "${existing%%=*}" == "$key" && "$existing" != "$flag" ]]; then
      echo "Conflicting frozen XLA option: $existing (expected $flag)" >&2
      exit 2
    fi
  done
  if [[ " ${XLA_FLAGS:-} " != *" $flag "* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:-} $flag"
  fi
done
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export LIBERO_CONFIG_PATH="$RECAP_RUNTIME/libero-config"
# Use only graphics userspace libraries matching the running kernel driver.
driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
graphics="$RECAP_RUNTIME/assets/nvidia-egl/$driver_version"
if [[ -f "$graphics/10_nvidia.json" ]]; then
  export LD_LIBRARY_PATH="$graphics/lib:${LD_LIBRARY_PATH:-}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$graphics/10_nvidia.json"
fi
export WANDB_MODE=disabled
if [[ "${E0_PLATFORM:-}" == cyy_dev ]]; then
  export NCCL_PROTO=LL
  export NCCL_P2P_DISABLE=1
  export JAX_ENABLE_COMPILATION_CACHE=false
  nccl="$RECAP_RUNTIME/assets/nccl/2.31.2/libnccl.so.2"
  if [[ ! -f "$nccl" ]]; then
    echo "cyy_dev requires the verified NCCL 2.31.2 runtime: $nccl" >&2
    exit 2
  fi
  if [[ ":${LD_PRELOAD:-}:" != *":$nccl:"* ]]; then
    export LD_PRELOAD="$nccl${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi
export TMPDIR="$RECAP_RUNTIME/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$LIBERO_CONFIG_PATH"
cd "$ROOT"
command=${1:-help}
if [[ "$command" == env ]]; then exec bash "$ROOT/e0/environment/bootstrap.sh"; fi
if [[ "$command" == rollout || "$command" == egl || "$command" == schema-libero || "$command" == replay-check || "$command" == research-branch ]]; then
  python="$ROOT/.venv-libero/bin/python"
else
  python="$ROOT/.venv-train/bin/python"
fi
case "$command" in
  run|resume|serve|predict|preflight|research-conditions|research-loss) export JAX_PLATFORMS=cuda ;;
  *) export JAX_PLATFORMS=cpu ;;
esac
exec "$python" "$ROOT/scripts/e0/main.py" "$@"
