import numpy as np
from openpi_client.e0.protocol import advantages
from openpi_client.e0.protocol import condition_labels
from openpi_client.e0.protocol import episodes
from openpi_client.e0.protocol import select_checkpoint
from openpi_client.e0.protocol import summarize_attempts
from openpi_client.e0.protocol import training_seed_statistics
from openpi_client.e0.protocol import trajectory_labels
import pytest

from openpi.e0.sampling import DrawDataset
from openpi.e0.sampling import DrawSampler
from openpi.e0.sampling import FrameStream


def test_terminal_and_short_trajectory():
    assert trajectory_labels(1, success=True)["value_label"].tolist() == [0]
    assert trajectory_labels(1, success=False)["reward_label"].tolist() == [-1]
    for n in (1, 3, 49, 50, 51):
        for success in (True, False):
            labels = trajectory_labels(n, success)
            v = np.linspace(-0.9, -0.1, n)
            expected = np.array(
                [
                    sum(labels["reward_label"][t : min(t + 50, n)]) + (v[t + 50] if t + 50 < n else 0) - v[t]
                    for t in range(n)
                ]
            )
            np.testing.assert_allclose(advantages(labels["reward_label"], v), expected, atol=1e-7)
    with pytest.raises(ValueError, match="zero-action"):
        trajectory_labels(0, success=False)


def test_per_task_ties_and_expert_exclusion():
    labels, stats = condition_labels(
        [0] * 5 + [1] * 4, [1, 2, 3, 4, 1e9, 8, 8, 8, -1e9], [False] * 4 + [True] + [False] * 3 + [True]
    )
    assert labels.tolist() == [
        "negative",
        "negative",
        "negative",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
    ]
    assert stats["0"]["threshold"] == pytest.approx(3.1)
    assert stats["1"]["positive_fraction"] == 1


def test_prefetch_resume_and_task_sampling():
    stream = FrameStream([10] * 40, list(range(40)), list(range(10)), 42, "recap")
    all_draws = list(DrawSampler(stream, 0, 5000))
    resumed = list(DrawSampler(FrameStream([10] * 40, list(range(40)), list(range(10)), 42, "recap"), 1024, 5000))
    assert all_draws[1024:] == resumed
    fraction = np.mean([frame < 100 for _, frame in all_draws])
    assert 0.72 < fraction < 0.78
    sft = FrameStream([3, 4], [0, 1], [], 42, "sft")
    assert sorted(sft.frame(i) for i in range(7)) == list(range(7))


def test_transform_rng_is_draw_indexed():
    class Dataset:
        def __len__(self):
            return 3

        def __getitem__(self, i):
            return i, np.random.random()

    ds = DrawDataset(Dataset(), 42)
    expected = ds[(10, 2)]
    for i in range(20):
        ds[(i, 0)]
    assert ds[(10, 2)] == expected


def test_denominator_and_macro():
    specs = list(episodes("dev", ["libero_10"], attempts=1))
    rows = [dict(x, status="ok", success=i < 3, policy_steps=10) for i, x in enumerate(specs)]
    assert summarize_attempts(specs, rows)["macro_success"] == 0.3
    assert not summarize_attempts(specs, rows[:-1])["complete"]
    rows[-1]["status"] = "infrastructure_error"
    assert not summarize_attempts(specs, rows)["complete"]


def test_training_seeds_not_eval_seeds():
    stats = training_seed_statistics({0: [0.1, 0.2, 0.3], 1: [0.2, 0.3, 0.4], 2: [0.3, 0.4, 0.5]})
    assert stats["training_repeats"] == 3
    assert stats["sample_std"] == pytest.approx(0.1)
    assert select_checkpoint([{"updates": 40, "sr": 0.5}, {"updates": 30, "sr": 0.5}], "sr")["updates"] == 30


class RandomDataset:
    def __len__(self):
        return 120

    def __getitem__(self, index):
        import random

        import torch

        return {
            "frame": np.asarray(index),
            "numpy": np.asarray(np.random.rand()),
            "python": np.asarray(random.random()),
            "torch": torch.rand(()).numpy(),
        }


def test_real_dataloader_prefetch_and_worker_partition():
    from openpi.e0.sampling import batches

    def stream():
        return FrameStream([3] * 40, list(range(40)), list(range(10)), 42, "recap")

    continuous = list(batches(RandomDataset(), stream(), 8, 0, 7, workers=0))
    resumed = list(batches(RandomDataset(), stream(), 8, 3, 7, workers=2))
    for left, right in zip(continuous[3:], resumed, strict=True):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


def test_complete_grid_and_changed_seed_rejected():
    from openpi_client.e0.protocol import validate_grid

    rows = list(episodes("final/s0", ["libero_10"], groups=[0, 1, 2], purpose="final", training_seed=0))
    validate_grid(rows, ["libero_10"], [0, 1, 2], "final")
    with pytest.raises(ValueError, match="exact task"):
        validate_grid(rows[:-1], ["libero_10"], [0, 1, 2], "final")
    rows[0]["sampling_seed"] += 1
    with pytest.raises(ValueError, match="seed derivation"):
        validate_grid(rows, ["libero_10"], [0, 1, 2], "final")
