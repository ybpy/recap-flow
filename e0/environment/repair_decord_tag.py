"""Repair decord 0.6.0's upstream wheel tag; library files are unchanged."""

import base64
import csv
import hashlib
import importlib.metadata
import io
from pathlib import Path
import platform

try:
    dist = importlib.metadata.distribution("decord")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(0) from None
if dist.version != "0.6.0" or platform.system() != "Linux" or platform.machine() != "x86_64":
    raise RuntimeError("unexpected decord version/platform; do not apply this repair")

directory = Path(dist.locate_file(next(file for file in dist.files if str(file).endswith(".dist-info/WHEEL")))).parent
wheel = directory / "WHEEL"
original = wheel.read_text()
old = "Tag: cp36-cp36m-manylinux2010_x86_64"
new = "Tag: py3-none-manylinux2010_x86_64"
if old not in original:
    if new not in original:
        raise RuntimeError("unknown decord wheel metadata")
else:
    wheel.write_text(original.replace(old, new))
    record = directory / "RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text())))
    name = directory.name + "/WHEEL"
    payload = wheel.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
    for row in rows:
        if row[0] == name:
            row[1] = "sha256=" + digest
            row[2] = str(len(payload))
    with record.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("decord 0.6.0: corrected CPython-3.6 tag to upstream wheel filename's py3-none tag")
