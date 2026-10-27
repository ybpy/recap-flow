"""Resume verified public artifacts using bounded byte ranges and final SHA256."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request


def fetch(url, target, total, digest, seed_parts=(), workers=8):
    target = Path(target)
    if target.exists():
        if hash_file(target) == digest:
            return target
        raise ValueError(f"existing target checksum mismatch: {target}")
    parts = target.with_name(target.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    block_size = 64 << 20
    for other in map(Path, seed_parts):
        if not other.exists():
            continue
        for source in other.iterdir():
            if not source.name.isdigit():
                continue
            destination = parts / source.name
            expected = min(block_size, total - int(source.name))
            size = source.stat().st_size
            if size <= 0 or size > expected:
                continue
            if destination.exists() and destination.stat().st_size >= size:
                continue
            # Copy a consistent prefix; another downloader may still append to its file.
            with source.open("rb") as src, destination.open("wb") as dst:
                remaining = size
                while remaining:
                    data = src.read(min(1 << 20, remaining))
                    if not data:
                        break
                    dst.write(data)
                    remaining -= len(data)

    def get(start):
        part = parts / str(start)
        expected = min(block_size, total - start)
        failures = 0
        while not part.exists() or part.stat().st_size < expected:
            size = part.stat().st_size if part.exists() else 0
            first = start + size
            last = min(start + expected, first + (4 << 20)) - 1
            request = urllib.request.Request(
                url + ("&" if "?" in url else "?") + f"e0range={first}-{last}",
                headers={"Range": f"bytes={first}-{last}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != f"bytes {first}-{last}/{total}"
                    ):
                        raise OSError("server did not honor requested range")
                    with part.open("ab") as output:
                        remaining = last - first + 1
                        while remaining:
                            data = response.read(min(256 << 10, remaining))
                            if not data:
                                raise OSError("short range; will resume prefix")
                            output.write(data)
                            remaining -= len(data)
                failures = 0
            except Exception as exc:
                failures += 1
                if failures >= 12:
                    raise
                print("retry", first, str(exc), flush=True)
                time.sleep(min(failures, 5))
        if part.stat().st_size != expected:
            raise OSError("invalid part size")
        print("part", start, expected, flush=True)
        return part

    with ThreadPoolExecutor(max_workers=workers) as pool:
        completed = list(pool.map(get, range(0, total, block_size)))
    tmp = target.with_name(target.name + ".assembling")
    with tmp.open("wb") as out:
        for p in completed:
            with p.open("rb") as src:
                shutil.copyfileobj(src, out, 8 << 20)
    if hash_file(tmp) != digest:
        raise OSError("assembled artifact SHA256 mismatch")
    tmp.rename(target)
    print("verified", str(target), digest, flush=True)
    return target


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while data := f.read(8 << 20):
            h.update(data)
    return h.hexdigest()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--seed-parts", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    fetch(a.url, a.target, a.size, a.sha256, a.seed_parts, a.workers)
