"""Manifest-driven LIBERO rollout, shared by cyy_dev and TIONE."""

from __future__ import annotations
from collections import deque
import json
from pathlib import Path
import random
import time
import traceback
import numpy as np
from openpi_client import image_tools, msgpack_numpy
from openpi_client.e0.artifacts import content_id, read_jsonl, write_json
from openpi_client.e0.protocol import H, K, WAIT_STEPS, stable_seed


class InfrastructureError(RuntimeError):
    pass


class PolicyFailure(RuntimeError):
    pass


class Client:
    def __init__(self, host, port, timeout):
        import websockets.sync.client
        from websockets.exceptions import ConnectionClosed

        self.closed = ConnectionClosed
        self.timeout = timeout
        self.packer = msgpack_numpy.Packer()
        try:
            self.ws = websockets.sync.client.connect(
                f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=30
            )
            self.metadata = msgpack_numpy.unpackb(self.ws.recv(timeout=30))
        except (OSError, TimeoutError, ConnectionClosed) as e:
            raise InfrastructureError(str(e)) from e
        if self.metadata.get("H") != H or self.metadata.get("K") != K:
            raise ValueError(f"unexpected policy protocol: {self.metadata}")

    def infer(self, observation):
        try:
            self.ws.send(self.packer.pack(observation))
            response = self.ws.recv(timeout=self.timeout)
        except TimeoutError as e:
            raise PolicyFailure("inference timeout") from e
        except (OSError, self.closed) as e:
            raise InfrastructureError(str(e)) from e
        if isinstance(response, str):
            raise RuntimeError("unclassified policy server exception: " + response)
        actions = np.asarray(msgpack_numpy.unpackb(response)["actions"], np.float32)
        if actions.shape != (H, 7) or not np.isfinite(actions).all():
            raise PolicyFailure("invalid action shape or nonfinite action chunk")
        return actions

    def close(self):
        self.ws.close()


def quat2axisangle(quat):
    q = np.asarray(quat, np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(denominator, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / denominator


def one_episode(spec, host, port, timeout, expected_policy_id=None):
    from openpi_client.e0.preflight import configure_libero, official_init_states

    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    random.seed(spec["env_seed"])
    np.random.seed(spec["env_seed"])
    suite = benchmark.get_benchmark_dict()[spec["suite"]]()
    task = suite.get_task(spec["task_id"])
    states = official_init_states(suite, spec["task_id"])
    if len(states) != 50:
        raise ValueError("official initial-state inventory must contain 50 entries")
    env, client = None, None
    steps, success, error = [], False, None
    try:
        client = Client(host, port, timeout)
        if expected_policy_id and client.metadata.get("policy_id") != expected_policy_id:
            raise ValueError("policy changed during this episode manifest")
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        try:
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(spec["env_seed"])
            env.reset()
            observation = env.set_init_state(states[spec["initial_state_id"]])
        except Exception as e:
            render_failure = any(
                word in str(e) for word in ("EGL", "eglInitialize", "GLFW", "OpenGL", "DISPLAY", "Framebuffer")
            )
            if isinstance(e, (OSError, ImportError)) or render_failure:
                raise InfrastructureError(f"render/environment initialization: {e}") from e
            raise
        for _ in range(WAIT_STEPS):
            observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
        queue, decision = deque(), 0
        while len(steps) < spec["max_policy_steps"]:
            image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
            state = np.concatenate(
                (
                    observation["robot0_eef_pos"],
                    quat2axisangle(observation["robot0_eef_quat"]),
                    observation["robot0_gripper_qpos"],
                )
            ).astype(np.float32)
            if not np.isfinite(state).all():
                raise RuntimeError("unclassified nonfinite simulator observation")
            if not queue:
                request = {
                    "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224)),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist, 224, 224)
                    ),
                    "observation/state": state,
                    "prompt": str(task.language),
                    "adv_ind": "positive",
                    "sampling_seed": stable_seed(spec["sampling_seed"], decision),
                }
                queue.extend(client.infer(request)[:K])
                decision += 1
            action = queue.popleft()
            next_obs, _, done, _ = env.step(action.tolist())
            steps.append({"image": image, "wrist_image": wrist, "state": state, "actions": action})
            observation = next_obs
            if done:
                success = True
                break
    except PolicyFailure as e:
        error = str(e)
    finally:
        if env is not None:
            env.close()
        if client is not None:
            client.close()
    return steps, {
        "success": success,
        "policy_steps": len(steps),
        "status": "policy_failure" if error else "ok",
        "error": error,
        "prompt": str(task.language),
    }


def run(manifest_path, output, host="127.0.0.1", port=8000, shard=0, shards=1, timeout=120):
    if not 0 <= shard < shards:
        raise ValueError("invalid worker partition")
    rows = read_jsonl(manifest_path)
    if len({r["episode_uid"] for r in rows}) != len(rows):
        raise ValueError("duplicate episode IDs")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    client = Client(host, port, timeout)
    policy_metadata = client.metadata
    client.close()
    if not policy_metadata.get("policy_id"):
        raise ValueError("server is missing an immutable checkpoint identity")
    protocol = {
        "episode_manifest_sha256": content_id(rows),
        "H": H,
        "K": K,
        "wait_steps": WAIT_STEPS,
        "policy_id": policy_metadata["policy_id"],
    }
    from importlib.metadata import version
    import platform

    write_json(
        root / f"worker-{shard}-provenance.json",
        {
            "policy": policy_metadata,
            "python": platform.python_version(),
            "mujoco": version("mujoco"),
            "robosuite": version("robosuite"),
            "shard": shard,
            "shards": shards,
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "egl_device": __import__("os").environ.get("MUJOCO_EGL_DEVICE_ID"),
        },
    )
    old = root / "protocol.json"
    import fcntl

    with (root / ".protocol.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if old.exists() and json.loads(old.read_text()) != protocol:
            raise ValueError("output already belongs to another episode manifest or checkpoint")
        write_json(old, protocol)
    for index, spec in enumerate(rows):
        if index % shards != shard:
            continue
        key = content_id(spec["episode_uid"])
        result_path, trajectory_path = root / "attempts" / (key + ".json"), root / "trajectories" / (key + ".npz")
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["status"] != "infrastructure_error":
                continue
            # Two failed attempts already consumed. A new invocation must not silently add retries.
            raise InfrastructureError(f"episode exhausted retry budget: {spec['episode_uid']}")
        journal_path = root / "attempt-journals" / (key + ".json")
        journal = (
            json.loads(journal_path.read_text())
            if journal_path.exists()
            else {"episode_uid": spec["episode_uid"], "attempts": []}
        )
        for attempt in journal["attempts"]:
            if attempt["status"] == "started":
                attempt.update(status="interrupted", error="worker stopped before committing an episode result")
        errors = [a.get("error", a["status"]) for a in journal["attempts"]]
        start_retry = len(journal["attempts"])
        if start_retry >= 2:
            write_json(
                result_path,
                dict(spec, status="infrastructure_error", success=False, policy_steps=0, errors=errors, retry=1),
            )
            raise InfrastructureError(f"episode exhausted retry budget: {spec['episode_uid']}")
        for retry in range(start_retry, 2):
            journal["attempts"].append({"retry": retry, "status": "started"})
            write_json(journal_path, journal)
            t0 = time.monotonic()
            try:
                steps, result = one_episode(spec, host, port, timeout, protocol["policy_id"])
                result.update(
                    spec,
                    policy_id=protocol["policy_id"],
                    retry=retry,
                    seconds=time.monotonic() - t0,
                    previous_errors=errors,
                )
                if steps:
                    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = trajectory_path.with_suffix(".partial.npz")
                    np.savez_compressed(temporary, **{k: np.stack([s[k] for s in steps]) for k in steps[0]})
                    temporary.replace(trajectory_path)
                    result["trajectory"] = str(trajectory_path)
                write_json(result_path, result)
                journal["attempts"][-1].update(status="committed")
                write_json(journal_path, journal)
                print(json.dumps(result), flush=True)
                break
            except InfrastructureError as e:
                errors.append(str(e))
                journal["attempts"][-1].update(status="infrastructure_error", error=str(e))
                write_json(journal_path, journal)
                if retry == 1:
                    write_json(
                        result_path,
                        dict(
                            spec,
                            status="infrastructure_error",
                            success=False,
                            policy_steps=0,
                            errors=errors,
                            retry=retry,
                        ),
                    )
                    raise
            except Exception:
                journal["attempts"][-1].update(status="unclassified_error", error=traceback.format_exc())
                write_json(journal_path, journal)
                write_json(
                    root / "unclassified-errors" / (key + ".json"),
                    dict(spec, traceback=traceback.format_exc(), retry=retry),
                )
                raise
