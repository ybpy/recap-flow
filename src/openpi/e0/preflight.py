import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from openpi_client.e0.artifacts import sha256_file
from openpi_client.e0.artifacts import write_json


def train_environment(output):
    import jax
    import torch

    jax.config.update("jax_enable_compilation_cache", val=False)
    # Check GPU explicitly even when the calling command defaults JAX to CPU.
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("no JAX GPUs")
    from openpi.e0.platform import validate_devices
    import os
    validate_devices(os.environ["E0_PLATFORM"], [d.device_kind for d in devices],
                     process_count=jax.process_count(), training=True)
    gpu_results = []
    for device in devices:
        import numpy as np

        placement = jax.sharding.SingleDeviceSharding(device)
        matrix = jax.device_put(np.ones((256, 256), np.float32), placement)
        multiply = jax.jit(lambda x: x @ x, in_shardings=placement, out_shardings=placement)
        value = multiply(matrix)
        value.block_until_ready()
        assert float(np.asarray(value)[0, 0]) == 256.0
        gpu_results.append(str(device))
    for i in range(torch.cuda.device_count()):
        value = torch.ones((256, 256), device=f"cuda:{i}") @ torch.ones((256, 256), device=f"cuda:{i}")
        assert value[0, 0].item() == 256.0
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[3] / "e0/environment/gpu_collective_probe.py"), run_name="__main__")
    from openpi.e0.provenance import validate_loaded_libraries
    libraries = validate_loaded_libraries()
    import lerobot
    import openpi_client

    import gemma
    import openpi

    root = Path(__file__).resolve().parents[3]
    paths = {
        module.__name__: str(Path(module.__file__).resolve()) for module in (openpi, openpi_client, gemma, lerobot)
    }
    forbidden = []
    for p in Path(sys.prefix).glob("lib/python*/site-packages/*.pth"):
        forbidden.extend(str(p) for line in p.read_text().splitlines() if "pistar-world" in line)
    if forbidden:
        raise RuntimeError(f"cross-project environment dependency: {forbidden}")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("jax", "jaxlib", "flax", "optax", "orbax-checkpoint", "torch", "numpy", "lerobot", "datasets")
    }
    expected = {
        "jax": "0.5.3",
        "jaxlib": "0.5.3",
        "flax": "0.10.2",
        "optax": "0.2.4",
        "orbax-checkpoint": "0.11.13",
        "torch": "2.7.1+cu128",
        "numpy": "1.26.4",
        "lerobot": "0.1.0",
        "datasets": "3.6.0",
    }
    if sys.version_info[:3] != (3, 11, 9) or versions != expected:
        raise RuntimeError(f"environment differs from the pinned E0 train environment: {versions}")
    if any(not Path(p).is_relative_to(root) for p in paths.values()):
        raise RuntimeError(f"imports escape the owned project/environment: {paths}")
    result = {
        "complete": True,
        "python": sys.version,
        "executable": sys.executable,
        "real_interpreter": str(Path(sys.executable).resolve()),
        "imports": paths,
        "versions": versions,
        "devices": gpu_results,
        "loaded_libraries": libraries,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "environment_lock_sha256": sha256_file(root / "e0/environment/train.lock"),
    }
    write_json(output, result)
    print(json.dumps(result, indent=2))
