"""Exercise the actual LeRobot reader and writer in either E0 environment."""

import io
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from openpi_client.e0.artifacts import write_json
from openpi_client.e0.protocol import trajectory_labels


def run(data_dir="/mnt/recap-flow-runtime/datasets/sft-v1", output=None):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = Path(data_dir)
    info = json.loads((root / "meta/info.json").read_text())
    source = root / info["data_path"].format(episode_chunk=0, episode_index=0)
    table = pq.read_table(source)
    dataset = LeRobotDataset(
        str(root), root=root, episodes=[0], delta_timestamps={"actions": [i / 10 for i in range(10)]}
    )
    first, last = dataset[0], dataset[len(dataset) - 1]
    actions = np.asarray(table["actions"].to_pylist(), np.float32)
    np.testing.assert_array_equal(first["actions"].numpy(), actions[:10])
    np.testing.assert_array_equal(last["actions"].numpy(), np.repeat(actions[-1:], 10, axis=0))
    images = {}
    for key in ("image", "wrist_image"):
        with Image.open(io.BytesIO(table[key][0].as_py()["bytes"])) as image:
            images[key] = np.asarray(image)
        np.testing.assert_allclose(first[key].numpy().transpose(1, 2, 0), images[key] / 255.0, atol=1e-7)
    target = Path(tempfile.mkdtemp(prefix="schema-", dir="/mnt/recap-flow-runtime/tmp")) / "dataset"
    defaults = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    features = {k: dict(v, shape=tuple(v["shape"])) for k, v in info["features"].items() if k not in defaults}
    writer = LeRobotDataset.create(
        repo_id="recap-flow/schema-probe", root=target, fps=10, robot_type="panda", features=features, use_videos=False
    )
    frame = dict(
        images,
        state=np.asarray(table["state"][0].as_py(), np.float32),
        actions=actions[0],
        intervention=np.asarray([0], np.int64),
        adv_ind="positive",
        task="schema roundtrip",
    )
    frame.update({k: v[:1] for k, v in trajectory_labels(1, True).items()})
    writer.add_frame(frame)
    writer.save_episode()
    writer.stop_image_writer()
    restored = LeRobotDataset("recap-flow/schema-probe", root=target)[0]
    np.testing.assert_array_equal(restored["actions"].numpy(), actions[0])
    for key in images:
        np.testing.assert_allclose(restored[key].numpy().transpose(1, 2, 0), images[key] / 255.0, atol=1e-7)
    result = {
        "complete": True,
        "interpreter": sys.executable,
        "source": str(root),
        "written_fixture": str(target),
        "action_horizon": 10,
        "terminal_padding": "repeat final action",
        "image_roundtrip": True,
    }
    output = output or "/mnt/recap-flow-runtime/reports/schema-" + Path(sys.prefix).name + ".json"
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
