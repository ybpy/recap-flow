"""Content-addressed metadata and atomic output helpers."""

from pathlib import Path
import hashlib
import json
import os


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_id(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("w") as f:
        for row in rows:
            f.write(canonical(row) + "\n")
    tmp.replace(path)


def rewrite_columns(table, replacements):
    # Preserve HF image features and all unrelated Arrow field/schema metadata.
    import pyarrow as pa

    for name, values in replacements.items():
        index = table.schema.get_field_index(name)
        if index < 0:
            raise KeyError(f"label column missing from source schema: {name}")
        field = table.schema.field(index)
        if pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type):
            values = [[v] for v in values]
        table = table.set_column(index, field, pa.array(values, type=field.type))
    return table
