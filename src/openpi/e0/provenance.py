"""Record code, environment and hardware without recording credentials."""

import importlib.metadata
import os
from pathlib import Path
import subprocess

from openpi_client.e0.artifacts import content_id
from openpi_client.e0.artifacts import sha256_file


def code_record(root):
    root = Path(root)
    names = (
        subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    files = {}
    for name in names:
        p = root / name
        if p.is_file() and p.suffix in (".py", ".sh", ".json", ".toml", ".lock", ".in"):
            files[name] = sha256_file(p)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    lines = subprocess.check_output(["git", "submodule", "status"], cwd=root, text=True).splitlines()
    # Branch/tag descriptions vary across independent clones; pin content and initialization state.
    submodule = [{"state": line[0], "commit": line[1:].split()[0], "path": line[1:].split()[1]} for line in lines]
    return {
        "commit": commit,
        "source_id": content_id({"files": files, "submodules": submodule}),
        "files": files,
        "submodules": submodule,
    }


def environment_record():
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_GL",
        "MUJOCO_EGL_DEVICE_ID",
        "JAX_PLATFORMS",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "OMP_NUM_THREADS",
        "LIBERO_CONFIG_PATH",
        "NCCL_PROTO",
        "NCCL_P2P_DISABLE",
        "JAX_ENABLE_COMPILATION_CACHE",
        "E0_PLATFORM",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "NCCL_PCI_SEARCH_FIX",
        "NCCL_ALGO",
        "CUBLAS_WORKSPACE_CONFIG",
        "NCCL_P2P_LEVEL",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
    )
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"], text=True
    ).splitlines()
    libraries = []
    maps = Path("/proc/self/maps")
    if maps.exists():
        libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines() if "libnccl.so" in line})
    return {
        "nccl_libraries": libraries,
        "packages": packages,
        "environment": {k: os.environ[k] for k in keys if k in os.environ},
        "gpus": gpu,
    }


def runtime_identity():
    """Portable across identical nodes; excludes GPU UUIDs and transient process IDs."""
    import sys

    root = Path(__file__).resolve().parents[3]
    record = environment_record()
    keys = {k: v for k, v in os.environ.items() if k.startswith("NCCL_") and k != "NCCL_DEBUG"}
    identity = {
        "python": sys.version,
        "python_binary_sha256": sha256_file(Path(sys.executable).resolve()),
        "packages": record["packages"],
        "gpu_models_drivers_memory": [line.split(",", 1)[1].strip() for line in record["gpus"]],
        "environment": record["environment"],
        "communication_variables": keys,
        "preloaded_libraries": {p: sha256_file(Path(p).resolve())
                                for p in os.environ.get("LD_PRELOAD", "").split(":") if p},
        "host_driver_libraries": {name: sha256_file(Path("/usr/lib/x86_64-linux-gnu") / name)
                                  for name in ("libcuda.so.1", "libnvidia-ptxjitcompiler.so.1")},
        "locks": {name: sha256_file(root / "e0/environment" / name)
                  for name in ("train.lock", "libero.lock", "runtime-lock.json")},
    }
    identity["runtime_id"] = content_id(identity)
    return identity


def validate_loaded_libraries():
    libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if any(name in line for name in ("libcuda.so", "libcudart.so", "libcublas.so", "libcudnn.so", "libnccl.so"))})
    if os.environ.get("E0_PLATFORM") == "pro5000":
        if any("/compat/" in name or "/python3.12/" in name for name in libraries):
            raise RuntimeError(f"unexpected CUDA compatibility/image Python library loaded: {libraries}")
        driver = [name for name in libraries if "libcuda.so" in name]
        if not driver or any(not name.endswith("libcuda.so.580.126.20") for name in driver):
            raise RuntimeError(f"PRO5000 driver library differs from frozen driver: {driver}")
        nccl = [name for name in libraries if "libnccl.so" in name]
        if nccl != ["/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.4"]:
            raise RuntimeError(f"unexpected communication libraries: {nccl}")
    return libraries


def archive_code(root, record, destination):
    import hashlib
    import io
    import tarfile

    root = Path(root)
    with tarfile.open(destination, "w:gz") as archive:
        for relative in sorted(record["files"]):
            data = (root / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != record["files"][relative]:
                raise RuntimeError(f"source changed while recording run: {relative}; use an immutable release")
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = (root / relative).stat().st_mode & 0o777
            archive.addfile(info, io.BytesIO(data))
