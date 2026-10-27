"""Optionally copy already-downloaded pinned wheel caches; never links runtime packages."""

import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--target", required=True)
args = parser.parse_args()
SOURCE = Path(args.source)
TARGET = Path(args.target)
for package in ("torch",):
    for record in SOURCE.glob(f"wheels-v5/url/*/{package}"):
        links = [p for p in record.iterdir() if p.is_symlink()]
        for link in links:
            archive = link.resolve()
            target_archive = TARGET / "archive-v0" / archive.name
            if not target_archive.exists():
                shutil.copytree(archive, target_archive)
            dst_record = TARGET / record.relative_to(SOURCE)
            dst_record.mkdir(parents=True, exist_ok=True)
            for item in record.iterdir():
                destination = dst_record / item.name
                if item.is_symlink():
                    if not destination.exists():
                        destination.symlink_to(target_archive)
                elif item.is_file():
                    shutil.copy2(item, destination)
            print(f"Copied cached wheel {link.name}; installation still uses --link-mode copy", flush=True)
