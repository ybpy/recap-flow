import pytest

from openpi.e0.platform import actor_fsdp, validate_devices, validate_training


def test_platform_rejects_mixed_hardware_and_wrong_size():
    kinds = ["NVIDIA RTX PRO 5000 72GB Blackwell"] * 8
    validate_devices("pro5000", kinds, training=True)
    assert actor_fsdp("pro5000") == 4
    with pytest.raises(ValueError):
        validate_devices("a800", kinds, training=True)
    with pytest.raises(ValueError):
        validate_devices("pro5000", kinds[:4], training=True)
    with pytest.raises(ValueError):
        validate_devices("pro5000", kinds, process_count=2, training=True)


def test_formal_platform_and_sharding_are_not_relaxed(monkeypatch):
    from types import SimpleNamespace

    jax = SimpleNamespace(devices=lambda: [SimpleNamespace(device_kind="NVIDIA RTX PRO 5000 72GB Blackwell")]*8,
                          process_count=lambda: 1)
    monkeypatch.setenv("E0_PLATFORM", "pro5000")
    spec = dict(platform="pro5000", smoke=False, kind="actor", fsdp_devices=4)
    validate_training(spec, jax)
    with pytest.raises(ValueError, match="FSDP=4"):
        validate_training(dict(spec, fsdp_devices=1), jax)
    with pytest.raises(ValueError, match="eight-GPU"):
        validate_training(dict(spec, platform="cyy_dev"), jax)
