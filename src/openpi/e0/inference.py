import json
from pathlib import Path

import numpy as np
from openpi_client.e0.artifacts import content_id
from openpi_client.e0.artifacts import write_json

from openpi.e0.config import actor_config
from openpi.e0.config import read_manifest


def serve(manifest, checkpoint, port=8000):
    import jax

    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared.normalize import load

    if jax.default_backend() != "gpu":
        raise RuntimeError("policy serving requires GPU")
    spec = read_manifest(manifest)
    config = actor_config(spec, inference=True)
    policy = create_trained_policy(
        config,
        checkpoint,
        norm_stats=load(Path(spec["norm_assets"]) / "libero"),
        sample_kwargs={"num_steps": 10, "adv_guidance_beta": 1.0},
    )
    # Compile before opening the port: compilation latency cannot become an episode timeout.
    rng = np.random.default_rng(0)
    example = {
        "observation/state": rng.random(8).astype(np.float32),
        "observation/image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }
    example.update(adv_ind="positive", sampling_seed=0)
    first = policy.infer(example)["actions"]
    policy.infer(dict(example, sampling_seed=1))
    replay = policy.infer(example)["actions"]
    np.testing.assert_array_equal(first, replay)
    checkpoint_record = json.loads((Path(checkpoint) / "e0-checkpoint.json").read_text())
    if checkpoint_record["manifest_id"] != spec["manifest_id"]:
        raise ValueError("policy checkpoint and training manifest differ")
    metadata = dict(
        config.policy_metadata,
        checkpoint=str(checkpoint),
        manifest_id=spec["manifest_id"],
        policy_id=content_id(checkpoint_record),
        replay_test="passed",
        replay_action_sha256=__import__("hashlib").sha256(np.ascontiguousarray(first).tobytes()).hexdigest(),
    )
    WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata=metadata).serve_forever()


def critic_predictions(manifest, checkpoint, data_dir, output, *, batch_size=32, autonomous_only=False):
    from collections import defaultdict
    import sys

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    from openpi.e0.sampling import collate
    from openpi.models import model as model_api
    from openpi.models.value_model import dist_to_value
    from openpi.models.value_model import normalize_value_target_shape
    from openpi.models.value_model import target_to_twohot
    from openpi.models.value_model_config import ValueModelConfig
    from openpi.training import data_loader as dl

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import train_value

    spec = read_manifest(manifest)
    checkpoint_record = json.loads((Path(checkpoint) / "e0-checkpoint.json").read_text())
    if spec["kind"] != "critic" or checkpoint_record["manifest_id"] != spec["manifest_id"]:
        raise ValueError("critic checkpoint and training manifest differ")
    dataset_manifest = json.loads((Path(data_dir) / "e0-manifest.json").read_text())
    sources = {e["source"] for e in dataset_manifest["episodes"]}
    if autonomous_only:
        if not sources <= {"demo", "rollout"}:
            raise ValueError("relabel predictions must be made on the cumulative training pool")
    else:
        if sources != {"development"}:
            raise ValueError("critic validation must use independent development trajectories")
        present = {e["task_index"] for e in dataset_manifest["episodes"]}
        if present != set(dataset_manifest["long_task_indices"]):
            raise ValueError("critic validation must cover all ten Long tasks")
    config = ValueModelConfig()
    data_config = train_value.build_value_data_config(data_dir, config, tokenizer_path=spec["tokenizer_path"])
    dataset = dl.transform_dataset(dl.create_torch_dataset(data_config, 1, config), data_config, skip_norm_stats=True)
    model = config.load(model_api.restore_params(Path(checkpoint) / "params"))
    model.eval()
    graph, params = nnx.split(model)

    @jax.jit
    def predict(params, observation, targets):
        model = nnx.merge(graph, params)
        obs = model_api.preprocess_observation(
            jax.random.key(0), observation, train=False, image_keys=list(observation.images)
        )
        logits = model(obs, train=False).astype(jnp.float32)
        targets = target_to_twohot(normalize_value_target_shape(targets))
        return dist_to_value(logits), -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)

    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    task_ce, records, offset = defaultdict(list), [], 0
    for episode in dataset_manifest["episodes"]:
        length = episode["length"]
        if autonomous_only and episode["source"] == "demo":
            offset += length
            continue
        values, losses = [], []
        for begin in range(offset, offset + length, batch_size):
            indices = list(range(begin, min(begin + batch_size, offset + length)))
            n = len(indices)
            indices += [indices[-1]] * (batch_size - n)
            raw = collate([dataset[i] for i in indices])
            v, ce = predict(params, model_api.Observation.from_dict(raw), raw["value"])
            values.extend(np.asarray(v)[:n].tolist())
            losses.extend(np.asarray(ce)[:n].tolist())
        if not np.isfinite(values).all() or not np.isfinite(losses).all():
            raise FloatingPointError("critic produced nonfinite values/CE")
        name = f"episode_{episode['episode_index']:06d}.npy"
        np.save(out / name, np.asarray(values, np.float32))
        records.append({"episode_uid": episode["episode_uid"], "path": name, "length": length})
        task_ce[str(episode["task_index"])].extend(losses)
        offset += length
    scores = {key: float(np.mean(value)) for key, value in task_ce.items()}
    write_json(
        out / "predictions.json",
        {
            "dataset_id": dataset_manifest["dataset_id"],
            "checkpoint": str(checkpoint),
            "manifest_id": spec["manifest_id"],
            "episodes": records,
            "task_ce": scores,
            "macro_ce": float(np.mean(list(scores.values()))),
        },
    )
