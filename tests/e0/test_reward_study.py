import numpy as np
import pytest
from openpi_client.e0.protocol import advantages, trajectory_labels

from openpi.e0.critic_report import parts, estimates
from openpi.e0.reward_study import METHODS, fit_scales, targets, save_labels, load_labels, ValueDataset


@pytest.fixture
def pool():
    return dict(
        dataset_id="train",
        long_task_indices=[0],
        episodes=[
            dict(episode_uid="demo", source="demo", task_index=0, length=100),
            dict(episode_uid="r", source="rollout", task_index=0, length=17, success=False),
            dict(episode_uid="demo2", source="demo", task_index=1, length=150),
        ],
    )


@pytest.mark.parametrize("length", [1, 4, 50, 51, 520])
@pytest.mark.parametrize("success", [False, True])
def test_reward_definitions(pool, length, success):
    scales = fit_scales(pool)
    for m in METHODS:
        reward, value = targets(m, length, success, 0, scales).T
        if m == METHODS[0]:
            old = trajectory_labels(length, success)
            np.testing.assert_array_equal(value, old["value_label"])
            np.testing.assert_array_equal(reward, old["reward_label"])
        else:
            np.testing.assert_allclose(value[:-1], reward[:-1] + value[1:], rtol=1e-6, atol=1e-6)
            assert value[-1] == reward[-1] == (0 if success else -0.5)
            assert np.min(value / scales["head_scale"]) >= -1
        w, d, a = parts(reward, value)
        np.testing.assert_allclose(a, advantages(reward, value), atol=1e-12)
        np.testing.assert_allclose(w + d, a)
        assert d[-1] == -value[-1]
    np.testing.assert_array_equal(
        targets(METHODS[1], length, success, 1, scales), targets(METHODS[2], length, success, 1, scales)
    )


def test_label_binding(tmp_path, pool):
    scales = fit_scales(pool)
    record = save_labels(tmp_path, METHODS[1], scales, {"train": pool})
    spec = dict(
        reward_protocol=str(tmp_path / "protocol.json"), reward_protocol_id=record["protocol_id"], dataset_id="train"
    )
    a, scale = load_labels(spec, pool, "train")
    dummy = [{"value": np.zeros((1,), np.float32)} for _ in a]
    overlay = ValueDataset(dummy, a, scale)
    assert overlay[101]["value"].item() == np.float32(a[101, 1] / scale)
    with pytest.raises(ValueError):
        load_labels(dict(spec, reward_protocol_id="other"), pool, "train")
    with pytest.raises(ValueError):
        load_labels(spec, dict(pool, dataset_id="other"), "train")
    with pytest.raises(ValueError):
        fit_scales(dict(pool, episodes=[dict(pool["episodes"][0], source="review")]))


def test_episode_intervals_are_paired():
    metrics = {"u": {"d": 0.0}, "v": {"d": 0.0}, "x": {"d": 1.0}}
    attempts = {"u": {"task_id": 0}, "v": {"task_id": 0}, "x": {"task_id": 1}}
    assert estimates(metrics, attempts)["macro"]["d"] == dict(mean=0.5, ci95=[0.5, 0.5], tasks=2)
