import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.e0.platform import validate_training
from openpi.e0.sampling import FrameStream, RECIPES
from openpi_client.e0.protocol import stable_seed


def fixture_stream(name=None):
    # Each Long task has 10 demo and 90 rollout frames; non-Long tasks have 10 demos.
    lengths, tasks, sources = [], [], []
    for task in range(40):
        lengths.append(10)
        tasks.append(task)
        sources.append("demo")
        if task < 10:
            lengths.append(90)
            tasks.append(task)
            sources.append("rollout")
    return FrameStream(lengths, tasks, list(range(10)), 42, "recap",
                       recipe=None if name is None else RECIPES[name], sources=sources)


def test_existing_stream_is_bitwise_unchanged():
    stream = FrameStream([10] * 40, range(40), range(10), 42, "recap")
    explicit = FrameStream([10] * 40, range(40), range(10), 42, "recap",
                           recipe=RECIPES["e0_75_25_pooled_v1"])
    for draw in range(2000):
        rng = np.random.default_rng(stable_seed(42, "frame", draw))
        tasks = list(range(10)) if rng.random() < 0.75 else list(range(10, 40))
        expected = tasks[int(rng.integers(len(tasks)))] * 10 + int(rng.integers(10))
        assert stream.frame(draw) == explicit.frame(draw) == expected


def test_long_recipes_task_source_exposure_and_resume():
    pooled = fixture_stream("long_pooled_v1")
    balanced = fixture_stream("long_demo70_rollout30_v1")
    a = np.array([pooled.frame(d) for d in range(20000)])
    b = np.array([balanced.frame(d) for d in range(20000)])
    assert max(a) < 1000 and max(b) < 1000
    # Paired variants use the same task sequence, despite different source weights.
    np.testing.assert_array_equal(a // 100, b // 100)
    assert np.max(np.abs(np.bincount(a // 100, minlength=10) / len(a) - 0.1)) < 0.01
    assert 0.085 < np.mean(a % 100 < 10) < 0.115
    assert 0.685 < np.mean(b % 100 < 10) < 0.715
    fresh = fixture_stream("long_demo70_rollout30_v1")
    np.testing.assert_array_equal(b[19800:], [fresh.frame(d) for d in range(19800, 20000)])


def test_ratio_and_missing_source_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="parameters differ"):
        FrameStream([10] * 40, range(40), range(10), 42, "recap",
                    recipe=dict(RECIPES["long_demo70_rollout30_v1"], demo_probability=0.3))
    with pytest.raises(ValueError, match="every Long task"):
        FrameStream([10] * 40, range(40), range(10), 42, "recap",
                    recipe=RECIPES["long_demo70_rollout30_v1"], sources=["demo"] * 40)
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"episodes": [{"source": "review"}], "long_task_indices": list(range(10))}))
    with pytest.raises(ValueError, match="evaluation trajectories"):
        FrameStream.from_manifest(path, 42, "recap", recipe=RECIPES["long_pooled_v1"])


def test_two_card_debug_cannot_bypass_formal_hardware(monkeypatch):
    monkeypatch.setenv("E0_PLATFORM", "pro5000")
    jax = SimpleNamespace(devices=lambda: [SimpleNamespace(device_kind="NVIDIA RTX PRO 5000 72GB Blackwell")] * 2,
                          process_count=lambda: 1)
    spec = dict(platform="pro5000", smoke=True, debug_gpu_count=2, kind="actor", fsdp_devices=2)
    validate_training(spec, jax)
    with pytest.raises(ValueError, match="smoke=True"):
        validate_training(dict(spec, smoke=False), jax)
    with pytest.raises(ValueError, match="count/sharding"):
        validate_training(dict(spec, fsdp_devices=4), jax)
    with pytest.raises(ValueError, match="incorrect GPU count"):
        validate_training({k: v for k, v in spec.items() if k != "debug_gpu_count"}, jax)
