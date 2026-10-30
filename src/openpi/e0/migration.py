"""Revalidate existing audited data and capture portable environment identities."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from openpi_client.e0.artifacts import content_id, sha256_file, write_json


def audit(output):
    root = Path(__file__).resolve().parents[3]
    runtime = Path(os.environ.get("RECAP_RUNTIME", "/mnt/recap-flow-runtime"))
    data = runtime / "datasets/sft-v1"
    manifest = json.loads((data / "e0-manifest.json").read_text())
    if manifest["dataset_id"] != content_id({k: v for k, v in manifest.items() if k != "dataset_id"}):
        raise ValueError("dataset manifest checksum mismatch")
    def verify(episode):
        path = data / episode["path"]
        if sha256_file(path) != episode["sha256"]:
            raise ValueError(f"corrupt or changed snapshot: {path}")
        return path.stat().st_size
    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(verify, manifest["episodes"]))
    norm = runtime / "assets/libero/norm_stats.json"
    norm_hash = sha256_file(norm)
    if norm_hash != "ccf0c0d5ca98df82a7b2933afb5441141a5d2f06b494d0f9106f5aad8492d05e":
        raise ValueError("frozen LIBERO normalization changed")
    interpreters = {}
    for name in ("train", "libero"):
        python = root / f".venv-{name}/bin/python"
        real = python.resolve()
        if not real.is_relative_to(root / ".uv-python"):
            raise ValueError(f"interpreter escapes project: {real}")
        details = subprocess.check_output([str(python), "-c",
            "import json,sys,sysconfig; print(json.dumps({'version':sys.version,'prefix':sys.prefix,'build':sysconfig.get_config_var('CONFIG_ARGS')}))"], text=True)
        interpreters[name] = dict(json.loads(details), executable=str(real), sha256=sha256_file(real))
    system = subprocess.check_output(["dpkg-query", "-W", "-f=${Package}=${Version}\n",
                    "libegl1", "libopengl0", "libglvnd0", "libegl-mesa0", "libc6"], text=True).splitlines()
    write_json(output, {"complete": True, "dataset_id": manifest["dataset_id"],
               "episodes": len(sizes), "frames": sum(e["length"] for e in manifest["episodes"]),
               "bytes_verified": sum(sizes), "norm_sha256": norm_hash, "norm_recomputed": False,
               "interpreters": interpreters, "system_packages": system,
               "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
               "locks": {n: sha256_file(root / "e0/environment" / n) for n in ("train.lock", "libero.lock")}})
