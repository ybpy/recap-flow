"""E0 actor/critic runner with complete state and draw-indexed recovery."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from openpi_client.e0.artifacts import sha256_file
from openpi_client.e0.artifacts import write_json

from openpi.e0.config import actor_config
from openpi.e0.config import read_manifest
from openpi.e0.config import validate_lineage
from openpi.e0.sampling import FrameStream
from openpi.e0.sampling import batches

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))


class CheckpointAssets:
    def __init__(self, data_config, spec, train_rng):
        import jax

        self.config, self.spec = data_config, spec
        self.train_key = np.asarray(jax.random.key_data(train_rng)).tolist()
        self.prng = {
            "impl": str(jax.config.jax_default_prng_impl),
            "threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        }
        self.completed = 0

    def data_config(self):
        return self.config

    def state_dict(self):
        return {
            "manifest_id": self.spec["manifest_id"],
            "completed_updates": self.completed,
            "next_draw": self.completed * self.spec["batch_size"],
            "seed": self.spec["seed"],
            "train_key": self.train_key,
            "prng": self.prng,
            "rng_algorithm": "jax key(seed), split train/init; fold_in(train_key, completed_updates)",
            "transform_rng": "sha256(seed, transform, draw); numpy+python+torch CPU states scoped per sample",
        }


def state_fingerprints(state):
    import hashlib

    import jax

    result = {}
    for key, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        array = np.ascontiguousarray(jax.device_get(value))
        result[str(key)] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return result


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_candidate(checkpoint, destination, spec, updates):
    if destination.exists():
        return
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"inspect interrupted export before retry: {temporary}")
    temporary.mkdir(parents=True)
    for name in ("params", "assets"):
        shutil.copytree(checkpoint / name, temporary / name, copy_function=_link_or_copy)
    write_json(
        temporary / "e0-checkpoint.json",
        {
            "manifest_id": spec["manifest_id"],
            "updates": updates,
            "parameter_kind": "EMA",
            "candidate": True,
            "smoke": spec["smoke"],
            "seed": spec["seed"],
            "stage": spec["stage"],
            "branch": spec.get("branch"),
            "run_id": spec["id"],
        },
    )
    temporary.rename(destination)


def run(manifest_path, *, resume=False, stop_after=None):
    import jax
    import jax.numpy as jnp
    import train as actor
    import train_value as critic

    from openpi.models import model as model_api
    from openpi.training import checkpoints as ckpts
    from openpi.training import data_loader as dl
    from openpi.training import sharding
    from openpi.training.utils import TrainState

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    spec = read_manifest(manifest_path)
    if jax.default_backend() != "gpu":
        raise RuntimeError("E0 training requires GPU; CPU fallback is forbidden")
    if not spec["smoke"] and (jax.device_count() != 8 or any("A800" not in d.device_kind for d in jax.devices())):
        raise ValueError(
            "E0 pilot/formal training requires one node with 8 A800 GPUs; use an explicit smoke manifest for debugging"
        )
    if spec["batch_size"] % jax.device_count():
        raise ValueError("global batch must be divisible by visible GPUs")
    dataset_manifest = json.loads(Path(spec["dataset_manifest"]).read_text())
    validate_lineage(spec, dataset_manifest)
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    record = run_dir / "resolved-manifest.json"
    if record.exists():
        if not resume:
            raise FileExistsError("run exists; choose a new ID or explicitly resume")
        if json.loads(record.read_text()) != spec:
            raise ValueError("resume manifest changed; new-round initialization is a separate run")
        from openpi.e0.provenance import code_record

        previous = json.loads((run_dir / "provenance.json").read_text())
        if previous["code"]["source_id"] != code_record(ROOT)["source_id"]:
            raise ValueError("resume code differs from the checkpoint run; use the recorded code snapshot")
    else:
        if resume:
            raise FileNotFoundError("cannot resume a run without provenance")
        write_json(record, spec)
        from openpi.e0.provenance import archive_code
        from openpi.e0.provenance import code_record
        from openpi.e0.provenance import environment_record

        source_record = code_record(ROOT)
        archive_code(ROOT, source_record, run_dir / "source.tar.gz")
        write_json(
            run_dir / "provenance.json",
            {
                "code": source_record,
                "environment": environment_record(),
                "source_archive_sha256": sha256_file(run_dir / "source.tar.gz"),
            },
        )
    runtime = Path(os.environ.get("RECAP_RUNTIME", "/mnt/recap-flow-runtime"))
    jax.config.update(
        "jax_compilation_cache_dir", os.environ.get("JAX_COMPILATION_CACHE_DIR", str(runtime / "cache/jax/all"))
    )
    mesh = sharding.make_mesh(spec["fsdp_devices"])
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_rng, init_rng = jax.random.split(jax.random.key(spec["seed"]))
    manager, resuming = ckpts.initialize_checkpoint_dir(
        run_dir / "recovery", keep_period=None, overwrite=False, resume=resume
    )
    if resume and not resuming:
        raise RuntimeError("no complete recovery checkpoint; do not silently restart")
    if spec["kind"] == "actor":
        config = actor_config(spec)
        data_config = config.data.create(config.assets_dirs, config.model)
        state, state_sharding = actor.init_train_state(config, init_rng, mesh, resume=resuming)
        step_fn = functools.partial(actor.train_step, config)
        model_config = config.model
        target_key = "actions"
    else:
        from openpi.models.value_model_config import ValueModelConfig

        model_config = ValueModelConfig()
        data_config = critic.build_value_data_config(
            spec["data_dir"], model_config, tokenizer_path=spec["tokenizer_path"]
        )
        initialized, tx, _ = critic.create_train_state(
            model_config,
            spec["num_train_steps"],
            init_rng,
            load_pretrained=not resuming,
            ema_decay=0.99,
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
            warmup_steps=spec["warmup_steps"],
            grad_clip_norm=1.0,
            freeze_mode=spec["freeze_mode"],
            siglip_path=spec["siglip_path"],
            gemma_checkpoint_dir=spec["gemma_checkpoint_dir"],
        )
        state = TrainState(
            step=initialized.step,
            params=initialized.params,
            model_def=initialized.model_def,
            opt_state=initialized.opt_state,
            tx=tx,
            ema_decay=0.99,
            ema_params=jax.tree.map(lambda x: jnp.array(x, copy=True), initialized.ema_params),
        )
        state_sharding = sharding.fsdp_sharding(state, mesh)
        state = jax.device_put(state, state_sharding)

        def step_fn(rng, state, batch):
            updated, metrics = critic.train_step(
                state, tx, rng, batch[0], batch[1], ema_decay=0.99, freeze_mode=spec["freeze_mode"]
            )
            return dataclasses.replace(
                state,
                step=updated.step,
                params=updated.params,
                opt_state=updated.opt_state,
                ema_params=updated.ema_params,
            ), metrics

        target_key = "value"
    effective = {
        "kind": spec["kind"],
        "model": dataclasses.asdict(model_config),
        "data": dataclasses.asdict(data_config),
        "manifest": spec,
    }
    if spec["kind"] == "actor":
        effective["train_config"] = dataclasses.asdict(config)
    else:
        effective["optimizer"] = {
            "name": "AdamW",
            "b1": 0.9,
            "b2": 0.95,
            "eps": 1e-8,
            "weight_decay": 1e-10,
            "clip": 1.0,
            "peak_lr": 2.5e-5,
            "end_lr": 2.5e-6,
            "ema": 0.99,
        }

    def serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return repr(value)

    if not resuming:
        write_json(run_dir / "effective-config.json", json.loads(json.dumps(effective, default=serializable)))
    assets = CheckpointAssets(data_config, spec, train_rng)
    if resuming:
        metadata = json.loads((run_dir / "recovery" / str(manager.latest_step()) / "assets/e0-state.json").read_text())
        if metadata["manifest_id"] != spec["manifest_id"]:
            raise ValueError("checkpoint belongs to a different manifest")
        if metadata["train_key"] != assets.train_key or metadata["prng"] != assets.prng:
            raise ValueError("checkpoint RNG implementation or seed differs from the run")
        train_rng = jax.random.wrap_key_data(jnp.asarray(metadata["train_key"], dtype=jnp.uint32))
        state = ckpts.restore_state(manager, state, assets)
        if (
            int(state.step) != metadata["completed_updates"]
            or metadata["next_draw"] != int(state.step) * spec["batch_size"]
        ):
            raise ValueError("checkpoint/data cursor mismatch")
    if spec["smoke"]:
        name = f"resumed-state-{int(state.step)}.json" if resuming else "initial-state-fingerprint.json"
        write_json(run_dir / name, state_fingerprints(state))
    dataset = dl.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    dataset = dl.transform_dataset(dataset, data_config, skip_norm_stats=spec["kind"] == "critic")
    stream = FrameStream.from_manifest(
        spec["dataset_manifest"], spec["seed"], "sft" if spec["stage"] == "sft" else "recap"
    )
    if len(dataset) != stream.total:
        raise ValueError("manifest and LeRobot frame count differ")
    start, end = int(state.step), min(spec["num_train_steps"], stop_after or spec["num_train_steps"])
    compiled = jax.jit(
        step_fn,
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    metrics_path = run_dir / "metrics.jsonl"
    if resuming and metrics_path.exists():
        # A process may crash after logging an update but before saving it.
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        committed = [row for row in rows if row["updates"] <= start]
        if len(committed) != len(rows):
            from openpi_client.e0.artifacts import write_jsonl

            write_jsonl(
                run_dir / f"rolled-back-metrics-{time.time_ns()}.jsonl", [row for row in rows if row["updates"] > start]
            )
            write_jsonl(metrics_path, committed)
        if [row["updates"] for row in committed] != list(range(1, start + 1)):
            raise ValueError("recovery metrics contain missing or duplicated updates")
    ends = np.cumsum([e["length"] for e in dataset_manifest["episodes"]])
    episode_tasks = np.asarray([e["task_index"] for e in dataset_manifest["episodes"]])
    episode_rollout = np.asarray([e["source"] == "rollout" for e in dataset_manifest["episodes"]])
    t0 = time.monotonic()
    session_id = __import__("uuid").uuid4().hex
    loader = batches(dataset, stream, spec["batch_size"], start, end, spec["num_workers"])
    for index, raw in enumerate(loader, start=start):
        raw_digest = hashlib_batch(raw)
        batch = (model_api.Observation.from_dict(raw), raw[target_key])
        batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), batch)
        with sharding.set_mesh(mesh):
            state, metrics = compiled(train_rng, state, batch)
        metrics = {key: float(x) for key, x in jax.device_get(metrics).items()}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"nonfinite training metric after update {index + 1}: {metrics}")
        completed = index + 1
        sampled_frames = [
            stream.frame(draw) for draw in range(index * spec["batch_size"], completed * spec["batch_size"])
        ]
        sampled_episodes = np.searchsorted(ends, sampled_frames, side="right")
        exposure = {
            "task_frame_counts": np.bincount(episode_tasks[sampled_episodes], minlength=40).tolist(),
            "rollout_frames": int(episode_rollout[sampled_episodes].sum()),
            "global_batch": spec["batch_size"],
            "gpu_count": jax.device_count(),
        }
        with metrics_path.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        updates=completed,
                        session_id=session_id,
                        batch_sha256=raw_digest,
                        rng_key=np.asarray(jax.random.key_data(jax.random.fold_in(train_rng, index))).tolist(),
                        elapsed_seconds=time.monotonic() - t0,
                        exposure=exposure,
                        **metrics,
                    )
                )
                + "\n"
            )
        if completed % 10 == 0 or completed == start + 1:
            logging.info("update=%s %s", completed, metrics)
        candidate = completed in spec["checkpoint_updates"]
        if candidate or completed % spec["checkpoint_interval"] == 0 or completed == end:
            assets.completed = completed
            ckpts.save_state(manager, state, assets, completed)
            manager.wait_until_finished()
            if candidate:
                export_candidate(
                    run_dir / "recovery" / str(completed), run_dir / "candidates" / str(completed), spec, completed
                )
    manager.wait_until_finished()
    manager.close()
    if spec["smoke"]:
        write_json(run_dir / "state-fingerprint.json", state_fingerprints(state))
    write_json(
        run_dir / "status.json",
        {
            "completed_updates": int(state.step),
            "planned_updates": spec["num_train_steps"],
            "complete": int(state.step) == spec["num_train_steps"],
            "smoke": spec["smoke"],
        },
    )
    return state


def hashlib_batch(batch):
    import hashlib

    import jax

    h = hashlib.sha256()
    for key, value in jax.tree_util.tree_flatten_with_path(batch)[0]:
        x = np.ascontiguousarray(value)
        h.update(str(key).encode())
        h.update(str(x.dtype).encode())
        h.update(x.tobytes())
    return h.hexdigest()
