"""Inventory only selected immutable assets, excluding temporary downloads."""

from pathlib import Path

from openpi_client.e0.artifacts import content_id
from openpi_client.e0.artifacts import sha256_file
from openpi_client.e0.artifacts import write_json


def inventory(output="/mnt/recap-flow-runtime/reports/asset-inventory.json"):
    root = Path(__file__).resolve().parents[3]
    runtime = Path("/mnt/recap-flow-runtime")
    locations = {
        "actor_base": Path("/mnt/models/pi05_base/params"),
        "actor_tokenizer": Path("/mnt/models/openpi-cache/big_vision"),
        "critic_siglip": runtime / "assets/critic/siglip-resumed.npz",
        "critic_gemma": runtime / "assets/critic/gemma-pretrained",
        "critic_tokenizer": runtime / "assets/critic/tokenizer.model",
        "libero_norm_stats": runtime / "assets/libero/norm_stats.json",
        "cyy_dev_nccl": runtime / "assets/nccl/2.31.2/libnccl.so.2",
        "cyy_dev_egl": runtime / "assets/nvidia-egl/580.126.20/lib",
        "libero_bddl": root / "third_party/libero/libero/libero/bddl_files",
        "libero_initial_states": root / "third_party/libero/libero/libero/init_files",
        "libero_assets": root / "third_party/libero/libero/libero/assets",
    }
    assets = {}
    for name, location in locations.items():
        if not location.exists():
            raise FileNotFoundError(location)
        directory = location if location.is_dir() else location.parent
        candidates = sorted(location.rglob("*")) if location.is_dir() else [location]
        files = []
        for p in candidates:
            if not p.is_file() or any(part.endswith((".parts", ".partial")) for part in p.parts):
                continue
            if p.suffix in (".partial", ".assembling"):
                continue
            files.append({"path": str(p.relative_to(directory)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        if not files:
            raise ValueError(f"empty asset location: {location}")
        assets[name] = {"root": str(directory), "files": files, "asset_id": content_id(files)}
        print(name, len(files), flush=True)
    result = {"assets": assets, "asset_id": content_id(assets)}
    write_json(output, result)
    return result
