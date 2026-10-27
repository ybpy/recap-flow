import io

import numpy as np
from openpi_client.e0.artifacts import rewrite_columns
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


def test_relabel_preserves_images_and_schema(tmp_path):
    buffer = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), 37, np.uint8)).save(buffer, format="PNG")
    schema = pa.schema(
        [
            pa.field("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            pa.field("adv_ind", pa.string()),
            pa.field("reward_label", pa.list_(pa.float32(), 1)),
        ],
        metadata={b"huggingface": b'{"info":{"features":{"image":{"_type":"Image"}}}}'},
    )
    table = pa.Table.from_pylist(
        [{"image": {"bytes": buffer.getvalue(), "path": None}, "adv_ind": "none", "reward_label": [-1.0]}],
        schema=schema,
    )
    result = rewrite_columns(table, {"adv_ind": ["positive"], "reward_label": [0.0]})
    path = tmp_path / "episode.parquet"
    pq.write_table(result, path)
    restored = pq.read_table(path)
    assert restored.schema.metadata == table.schema.metadata
    assert restored["image"].to_pylist() == table["image"].to_pylist()
    assert restored["reward_label"].to_pylist() == [[0.0]]
