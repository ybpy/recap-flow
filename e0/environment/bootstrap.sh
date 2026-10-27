#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export RECAP_RUNTIME=${RECAP_RUNTIME:-/mnt/recap-flow-runtime}
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
export UV_CACHE_DIR="$RECAP_RUNTIME/uv-cache"
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1
export PATH="$ROOT/e0/environment/build-bin:$PATH"
export UV_PYTHON_BIN_DIR="$ROOT/.uv-python/bin"
export TMPDIR="$RECAP_RUNTIME/tmp"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
cd "$ROOT"
git -c http.proxy="${https_proxy:-}" submodule update --init third_party/libero
uv python install 3.11.9
uv venv --allow-existing --python 3.11.9 .venv-train
uv venv --allow-existing --python /usr/bin/python3.10 .venv-libero
for name in train libero; do
  uv pip sync --python "$ROOT/.venv-$name/bin/python" --link-mode copy "e0/environment/$name.lock"
  "$ROOT/.venv-$name/bin/python" "$ROOT/e0/environment/repair_decord_tag.py"
  if [[ "$name" == libero ]]; then
    printf "%s\n" "$ROOT/third_party/libero" > "$ROOT/.venv-libero/lib/python3.10/site-packages/recap_libero.pth"
  fi
  uv pip check --python "$ROOT/.venv-$name/bin/python"
done
