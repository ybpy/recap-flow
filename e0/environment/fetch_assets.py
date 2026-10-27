"""Fetch only the fixed, checksummed critic release used by E0."""

import json
import os
from pathlib import Path

from resume_fetch import fetch


def main():
    lock = json.loads(Path(__file__).with_name("critic-assets.lock.json").read_text())
    root = Path(os.environ.get("RECAP_RUNTIME", "/mnt/recap-flow-runtime")) / "assets/critic"
    records = []
    for item in lock["files"]:
        target = root / item["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{lock['repo']}/resolve/{lock['revision']}/{item['publisher_path']}"
        fetch(url, target, item["bytes"], item["sha256"])
        records.append(dict(item, path=str(target)))
    (root / "selected-manifest.json").write_text(
        json.dumps({"repo": lock["repo"], "revision": lock["revision"], "verified": True, "files": records}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
