"""Research-only replay and branching; outputs are never a training dataset."""

from collections import deque
import json
from pathlib import Path
import random
import time

import numpy as np
from openpi_client import image_tools
from openpi_client.e0.artifacts import content_id, write_json
from openpi_client.e0.preflight import configure_libero, official_init_states
from openpi_client.e0.protocol import K, WAIT_STEPS, stable_seed
from openpi_client.e0.rollout import Client, quat2axisangle


def observation_arrays(observation):
    return {
        "image": np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]),
        "wrist_image": np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1]),
        "state": np.concatenate((observation["robot0_eef_pos"],
                                  quat2axisangle(observation["robot0_eef_quat"]),
                                  observation["robot0_gripper_qpos"])).astype(np.float32),
    }


def policy_request(arrays, prompt, seed, condition="positive"):
    return {
        "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(arrays["image"], 224, 224)),
        "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(arrays["wrist_image"], 224, 224)),
        "observation/state": arrays["state"], "prompt": prompt,
        "adv_ind": condition, "sampling_seed": seed,
    }


def replay_environment(result, trajectory, anchor):
    """Rebuild controller history too; validate every replayed observation."""
    # NpzFile decompresses on each access; load once before replaying a long prefix.
    trajectory = {key: trajectory[key] for key in ("actions", "state", "image", "wrist_image")}
    if not 0 <= anchor < len(trajectory["actions"]):
        raise ValueError("anchor must precede a recorded action")
    paths = configure_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    random.seed(result["env_seed"])
    np.random.seed(result["env_seed"])
    suite = benchmark.get_benchmark_dict()[result["suite"]]()
    task = suite.get_task(result["task_id"])
    env = OffScreenRenderEnv(bddl_file_name=str(Path(paths["bddl_files"]) / task.problem_folder / task.bddl_file),
                             camera_heights=256, camera_widths=256)
    try:
        env.seed(result["env_seed"])
        env.reset()
        obs = env.set_init_state(official_init_states(suite, result["task_id"])[result["initial_state_id"]])
        for _ in range(WAIT_STEPS):
            obs, *_ = env.step([0.] * 6 + [-1.])
        for index in range(anchor + 1):
            arrays = observation_arrays(obs)
            np.testing.assert_allclose(arrays["state"], trajectory["state"][index], rtol=0, atol=1e-6)
            for key in ("image", "wrist_image"):
                np.testing.assert_array_equal(arrays[key], trajectory[key][index])
            if index < anchor:
                obs, _, done, _ = env.step(trajectory["actions"][index].tolist())
                if done:
                    raise AssertionError("historical replay terminated before anchor")
        return env, obs
    except BaseException:
        env.close()
        raise


def replay_check(results, output):
    rows = [json.loads(p.read_text()) for p in sorted((Path(results) / "attempts").glob("*.json"))]
    chosen = {}
    for row in sorted(rows, key=lambda x: (x["task_id"], x["initial_state_id"])):
        if row.get("trajectory") and row["task_id"] not in chosen:
            chosen[row["task_id"]] = row
    if set(chosen) != set(range(10)):
        raise ValueError("replay acceptance requires trajectories from all ten Long tasks")
    records = []
    for task, row in chosen.items():
        with np.load(row["trajectory"]) as trajectory:
            anchor = max(0, (len(trajectory["actions"]) // 2 // K) * K)
            env, obs = replay_environment(row, trajectory, anchor)
            checked = 0
            try:
                for index in range(anchor, min(anchor + K, len(trajectory["actions"]))):
                    obs, _, done, _ = env.step(trajectory["actions"][index].tolist())
                    checked += 1
                    if index + 1 < len(trajectory["actions"]):
                        arrays = observation_arrays(obs)
                        for key in arrays:
                            np.testing.assert_allclose(arrays[key], trajectory[key][index + 1], rtol=0,
                                                       atol=1e-6 if key == "state" else 0)
                    if done:
                        assert row["success"] and index + 1 == len(trajectory["actions"])
            finally:
                env.close()
        records.append({"task_id": task, "episode_uid": row["episode_uid"], "anchor": anchor,
                        "fixed_future_steps": checked})
    write_json(output, {"complete": True, "purpose": "research", "training_eligible": False,
                        "method": "initial state + dummy actions + complete controller history",
                        "records": records})


def branch_run(manifest, output, host="127.0.0.1", port=8000, timeout=120):
    plan = json.loads(Path(manifest).read_text())
    if plan["purpose"] != "research" or not plan["namespace"].startswith("research/"):
        raise ValueError("branching requires an isolated research namespace")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    binding = {"plan_id": content_id(plan), "purpose": "research", "training_eligible": False}
    path = root / "binding.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("output bound to another research plan")
    write_json(path, binding)
    client = Client(host, port, timeout)
    try:
        if client.metadata["policy_id"] != plan["continuation_policy_id"]:
            raise ValueError("fixed continuation policy mismatch")
        for item in plan["anchors"]:
            row = json.loads(Path(item["result_path"]).read_text())
            if row["purpose"] != "development":
                raise ValueError("research anchors must come from development episodes")
            with np.load(row["trajectory"]) as trajectory:
                anchor = item["step"]
                arrays = {k: trajectory[k][anchor] for k in ("image", "wrist_image", "state")}
                for candidate in range(4):
                    chunk = client.infer(policy_request(arrays, row["prompt"],
                                         stable_seed(plan["namespace"], item["anchor_id"], "candidate", candidate)))
                    for group in range(3):
                        result_path = root / f'{item["anchor_id"]}-c{candidate}-g{group}.json'
                        if result_path.exists():
                            continue
                        started = time.monotonic()
                        env, obs = replay_environment(row, trajectory, anchor)
                        setup_seconds = time.monotonic() - started
                        executed, decision, done, failure = 0, 0, False, None
                        queue = deque(chunk[:K])
                        trace = []
                        try:
                            while anchor + executed < row["max_policy_steps"]:
                                if not queue:
                                    # Paired continuation noise across all four candidates.
                                    seed = stable_seed(plan["namespace"], item["anchor_id"], "continuation", group, decision)
                                    queue.extend(client.infer(policy_request(observation_arrays(obs), row["prompt"], seed))[:K])
                                    decision += 1
                                action = queue.popleft()
                                trace.append(dict(observation_arrays(obs), actions=action))
                                obs, _, done, _ = env.step(action.tolist())
                                executed += 1
                                if done:
                                    break
                        except Exception as error:
                            from openpi_client.e0.rollout import PolicyFailure
                            if not isinstance(error, PolicyFailure):
                                raise
                            failure = str(error)
                        finally:
                            env.close()
                        trajectory_path = result_path.with_suffix(".npz")
                        if trace:
                            temporary = trajectory_path.with_suffix(".partial.npz")
                            np.savez_compressed(temporary, **{key: np.stack([s[key] for s in trace]) for key in trace[0]})
                            temporary.replace(trajectory_path)
                        write_json(result_path, dict(binding, anchor_id=item["anchor_id"], candidate=candidate,
                                   group=group, actions=chunk.tolist(), success=bool(done), error=failure,
                                   source_episode_uid=row["episode_uid"], task_id=row["task_id"], anchor_step=anchor,
                                   trajectory=str(trajectory_path) if trace else None,
                                   replay_steps=anchor, dummy_steps=WAIT_STEPS, candidate_steps=min(K, executed),
                                   continuation_steps=max(0, executed-K), setup_seconds=setup_seconds,
                                   total_seconds=time.monotonic()-started))
    finally:
        client.close()
    rows = [json.loads(p.read_text()) for p in root.glob("*-c*-g*.json")]
    expected = len(plan["anchors"]) * 4 * 3
    if len(rows) != expected:
        raise RuntimeError(f"incomplete branching denominator: {len(rows)}/{expected}")
    write_json(root / "summary.json", dict(binding, complete=True, attempts=expected,
               successful_continuations=sum(row["success"] for row in rows),
               costs={key: sum(row[key] for row in rows)
                      for key in ("replay_steps", "dummy_steps", "candidate_steps", "continuation_steps")}))
