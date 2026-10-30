"""Explicit hardware contracts for current and historical E0 deployments."""

import os

PLATFORMS = ("cyy_dev", "a800", "pro5000")


def actor_fsdp(platform):
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform: {platform}")
    return 4 if platform in ("cyy_dev", "pro5000") else 1


def validate_devices(platform, kinds, *, process_count=1, training=False):
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform: {platform}")
    expected = "A800" if platform == "a800" else "RTX PRO 5000"
    if process_count != 1 or not kinds or any(expected not in kind for kind in kinds):
        raise ValueError(f"{platform} requires one process/node using {expected}; observed {kinds}")
    if training and len(kinds) != (4 if platform == "cyy_dev" else 8):
        raise ValueError(f"incorrect GPU count for {platform}: {len(kinds)}")


def validate_training(spec, jax):
    platform = spec.get("platform", "a800" if not spec["smoke"] else os.environ.get("E0_PLATFORM"))
    if not spec["smoke"] and platform not in ("a800", "pro5000"):
        raise ValueError("pilot/formal training requires an explicitly supported eight-GPU platform")
    if platform != os.environ.get("E0_PLATFORM"):
        raise ValueError("manifest platform and E0_PLATFORM differ")
    validate_devices(platform, [d.device_kind for d in jax.devices()],
                     process_count=jax.process_count(), training=True)
    if platform == "pro5000" and spec["fsdp_devices"] != (4 if spec["kind"] == "actor" else 1):
        raise ValueError("PRO5000 requires actor FSDP=4 and critic FSDP=1")
