#!/usr/bin/env bash
# Sourced only by the project entry point; never change the global environment.
export NCCL_PCI_SEARCH_FIX=0
export NCCL_IB_TC=160 NCCL_SOCKET_IFNAME=eth0 NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_LEVEL=SYS NCCL_IB_GID_INDEX=3 NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TIMEOUT=22 NCCL_IB_DISABLE=0
unset NCCL_P2P_DISABLE
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export JAX_ENABLE_COMPILATION_CACHE=false
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2
expected_nccl=37947ee6a4885076b964ffca7cd37c4ebbe0f04600ca6dca86980228a6570ebb
actual_nccl=$(sha256sum "$LD_PRELOAD" | cut -d ' ' -f 1)
if [[ "$actual_nccl" != "$expected_nccl" ]]; then
  echo "PRO5000 HCCL library differs from the verified release" >&2
  exit 2
fi
# Exclude image Python 3.12 Torch/TensorRT and CUDA 13 compatibility libraries.
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
if [[ "${1:-}" != env ]]; then
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi
