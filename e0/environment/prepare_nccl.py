"""Extract pinned NCCL 2.31.2 for an explicit Blackwell runtime profile."""

import json
import os
from pathlib import Path
import shutil
import zipfile

from resume_fetch import fetch
from resume_fetch import hash_file

runtime = Path(os.environ.get("RECAP_RUNTIME", "/mnt/recap-flow-runtime"))
version = "2.31.2"
meta = json.loads(Path(__file__).with_name("nccl-source.lock.json").read_text())
assert meta["version"] == version
entry = meta["artifact"]
target = runtime / "assets/nccl" / version
target.mkdir(parents=True, exist_ok=True)
wheel = target / entry["filename"]
if wheel.exists() and wheel.stat().st_size < entry["size"]:
    partial = wheel.with_suffix(".truncated")
    wheel.rename(partial)
    parts = wheel.with_name(wheel.name + ".parts")
    parts.mkdir(exist_ok=True)
    with partial.open("rb") as source:
        start = 0
        while block := source.read(64 << 20):
            (parts / str(start)).write_bytes(block)
            start += len(block)
fetch(entry["url"], wheel, entry["size"], entry["digests"]["sha256"])
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        if name.endswith("/libnccl.so.2"):
            with archive.open(name) as source, (target / "libnccl.so.2").open("wb") as output:
                shutil.copyfileobj(source, output, 8 << 20)
(target / "manifest.json").write_text(
    json.dumps(
        {
            "version": version,
            "url": entry["url"],
            "sha256": entry["digests"]["sha256"],
            "library_sha256": hash_file(target / "libnccl.so.2"),
        },
        indent=2,
    )
    + "\n"
)
print("verified", target, flush=True)
