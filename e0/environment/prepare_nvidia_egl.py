"""Extract matching NVIDIA graphics userspace libraries; never installs a kernel driver."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--version", required=True)
args = parser.parse_args()
runtime = Path("/mnt/recap-flow-runtime")
source = json.loads(Path(__file__).with_name(f"nvidia-{args.version}.lock.json").read_text())
entry = source["nvidia_driver"]["linux-x86_64"]
target = runtime / "assets/nvidia-egl" / args.version
target.mkdir(parents=True, exist_ok=True)
archive = target / "driver-userspace.tar.xz"
if not archive.exists():
    partial = archive.with_suffix(".partial")
    urls = [
        "https://developer.download.nvidia.cn/compute/nvidia-driver/redist/" + entry["relative_path"],
        "https://developer.download.nvidia.com/compute/nvidia-driver/redist/" + entry["relative_path"],
    ]
    for attempt in range(4):
        url = urls[min(attempt, 1)]
        start = partial.stat().st_size if partial.exists() else 0
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}) if attempt == 0 else urllib.request.ProxyHandler()
        )
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        try:
            with opener.open(request, timeout=60) as response:
                append = (
                    start > 0
                    and response.status == 206
                    and response.headers.get("Content-Range", "").startswith(f"bytes {start}-")
                )
                with partial.open("ab" if append else "wb") as output:
                    while block := response.read(4 << 20):
                        output.write(block)
            if partial.stat().st_size != int(entry["size"]):
                raise OSError("incomplete graphics archive")
            partial.rename(archive)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)
digest = hashlib.sha256()
with archive.open("rb") as f:
    while block := f.read(8 << 20):
        digest.update(block)
if digest.hexdigest() != entry["sha256"]:
    raise OSError("NVIDIA archive checksum mismatch")
library = target / "lib"
library.mkdir(exist_ok=True)
prefixes = (
    "libEGL_nvidia",
    "libGLX_nvidia",
    "libGLESv1_CM_nvidia",
    "libGLESv2_nvidia",
    "libnvidia-eglcore",
    "libnvidia-glcore",
    "libnvidia-glsi",
    "libnvidia-tls",
    "libnvidia-egl-gbm",
    "libnvidia-egl-wayland",
)
files = []
with tarfile.open(archive, "r:xz") as contents:
    for member in contents:
        name = Path(member.name).name
        if member.isfile() and name.startswith(prefixes) and ".so" in name:
            data = contents.extractfile(member)
            destination = library / name
            with destination.open("wb") as f:
                while block := data.read(4 << 20):
                    f.write(block)
            files.append(name)
for alias, prefix in [
    ("libEGL_nvidia.so.0", "libEGL_nvidia.so."),
    ("libGLX_nvidia.so.0", "libGLX_nvidia.so."),
    ("libGLESv1_CM_nvidia.so.1", "libGLESv1_CM_nvidia.so."),
    ("libGLESv2_nvidia.so.2", "libGLESv2_nvidia.so."),
]:
    candidate = library / (prefix + args.version)
    if candidate.exists() and not (library / alias).exists():
        (library / alias).symlink_to(candidate.name)
if not (library / "libEGL_nvidia.so.0").exists():
    raise FileNotFoundError("graphics library not in NVIDIA archive")
icd = {"file_format_version": "1.0.0", "ICD": {"library_path": str(library / "libEGL_nvidia.so.0")}}
(target / "10_nvidia.json").write_text(json.dumps(icd, indent=2) + "\n")
(target / "manifest.json").write_text(
    json.dumps({"version": args.version, "source": entry, "files": files}, indent=2) + "\n"
)
print(json.dumps({"complete": True, "version": args.version, "library": str(library), "files": files}), flush=True)
