import pytest

from openpi.e0.curves import aggregate_points


def test_curve_error_bars_use_training_seeds():
    points = [
        {
            "method": "warm_start",
            "stage": "r1",
            "training_seed": seed,
            "success": score,
            "cumulative_environment_actions": 1000,
            "training_loop_gpu_hours": 2,
            "measured_execution_gpu_hours": 1,
        }
        for seed, score in enumerate((0.2, 0.3, 0.4))
    ]
    row = aggregate_points(points)[0]
    assert row["training_repeats"] == 3
    assert row["success"]["mean"] == pytest.approx(0.3)
    assert row["success"]["sample_std"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="training seeds"):
        aggregate_points(points[:-1])
    with pytest.raises(ValueError, match="duplicate training seed"):
        aggregate_points(points + points[:1])
