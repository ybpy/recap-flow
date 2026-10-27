import json
import os
from pathlib import Path
import sys
import numpy as np
from openpi_client.e0.artifacts import write_json


def configure_libero():
    root = Path(__file__).resolve().parents[5]
    # Editable installation resolves to repo/packages/openpi-client/src/openpi_client/e0.
    root = next(p for p in Path(__file__).resolve().parents if (p / "third_party/libero").is_dir())
    config = Path(os.environ["LIBERO_CONFIG_PATH"])
    config.mkdir(parents=True, exist_ok=True)
    base = root / "third_party/libero/libero/libero"
    values = {
        "benchmark_root": str(base),
        "bddl_files": str(base / "bddl_files"),
        "init_states": str(base / "init_files"),
        "assets": str(base / "assets"),
        "datasets": "/mnt/recap-flow-runtime/datasets",
    }
    (config / "config.yaml").write_text("".join(f"{key}: {value}\n" for key, value in values.items()))
    return values


def official_init_states(suite, task_id):
    import torch
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    directory = Path(get_libero_path("init_states")).resolve()
    path = (directory / task.problem_folder / task.init_states_file).resolve()
    path.relative_to(directory)
    # Pinned LIBERO state files contain numpy arrays, not PyTorch weight tensors.
    return torch.load(path, map_location="cpu", weights_only=False)


def simulator(output):
    paths = configure_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import mujoco

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(paths["bddl_files"]) / task.problem_folder / task.bddl_file),
        camera_heights=256,
        camera_widths=256,
    )
    try:
        env.seed(42)
        env.reset()
        obs = env.set_init_state(official_init_states(suite, 0)[0])
        for _ in range(10):
            obs, *_ = env.step([0.0] * 6 + [-1.0])
        assert obs["agentview_image"].shape == (256, 256, 3)
        assert np.isfinite(obs["agentview_image"]).all()
        from OpenGL import GL

        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        if "NVIDIA" not in vendor:
            raise RuntimeError(f"EGL is using software/non-NVIDIA rendering: {vendor}, {renderer}")
        import torch

        result = {
            "complete": True,
            "python": sys.version,
            "executable": sys.executable,
            "real_interpreter": str(Path(sys.executable).resolve()),
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "paths": paths,
            "gl_vendor": vendor,
            "gl_renderer": renderer,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "image_shape": [256, 256, 3],
        }
        write_json(output, result)
        print(json.dumps(result, indent=2))
    finally:
        env.close()
