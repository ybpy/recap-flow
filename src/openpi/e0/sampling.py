"""A draw-indexed stream: prefetch cannot change the next resumed batch."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
from openpi_client.e0.protocol import stable_seed


RECIPES = {
    "e0_75_25_pooled_v1": dict(name="e0_75_25_pooled_v1", long_probability=0.75,
                                task_sampling="uniform", source_sampling="pooled_frames"),
    "long_pooled_v1": dict(name="long_pooled_v1", long_probability=1.0,
                            task_sampling="uniform", source_sampling="pooled_frames"),
    "long_demo70_rollout30_v1": dict(name="long_demo70_rollout30_v1", long_probability=1.0,
                                     task_sampling="uniform", source_sampling="demo_rollout",
                                     demo_probability=0.7),
}


def validate_recipe(recipe, mode):
    if recipe is None:
        return None
    if mode != "recap" or not isinstance(recipe, dict) or recipe.get("name") not in RECIPES:
        raise ValueError("unknown sampling recipe, or recipe used outside RECAP")
    if recipe != RECIPES[recipe["name"]]:
        raise ValueError("sampling recipe parameters differ from its versioned definition")
    return dict(recipe)


class FrameStream:
    def __init__(self, lengths, task_ids, long_tasks, seed, mode, *, recipe=None, sources=None):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if not len(self.lengths) or np.any(self.lengths <= 0):
            raise ValueError("sampling requires nonempty, positive episode lengths")
        self.ends = np.cumsum(self.lengths)
        self.seed, self.mode = seed, mode
        self.recipe = validate_recipe(recipe, mode)
        self.total = int(self.ends[-1])
        if mode not in ("sft", "recap"):
            raise ValueError(mode)
        by_task = defaultdict(list)
        by_source = defaultdict(list)
        if sources is not None and len(sources) != len(lengths):
            raise ValueError("episode sources and lengths differ")
        offset = 0
        for index, (length, task) in enumerate(zip(lengths, task_ids, strict=True)):
            by_task[int(task)].append((offset, int(length)))
            if sources is not None:
                by_source[(int(task), sources[index])].append((offset, int(length)))
            offset += int(length)
        self.tasks = {}
        for task, spans in by_task.items():
            self.tasks[task] = (spans, np.cumsum([n for _, n in spans]))
        self.long = sorted(set(map(int, long_tasks)))
        self.other = sorted(set(self.tasks) - set(self.long))
        if mode == "recap" and (len(self.long) != 10 or len(self.other) != 30):
            raise ValueError("RECAP requires all 10 Long and 30 other tasks")
        if not set(self.long) <= set(self.tasks):
            raise ValueError("Long task mapping references absent tasks")
        self.source_tasks = {key: (spans, np.cumsum([n for _, n in spans]))
                             for key, spans in by_source.items()}
        if self.recipe and self.recipe["source_sampling"] == "demo_rollout":
            if sources is None or any((task, source) not in self.source_tasks
                                      for task in self.long for source in ("demo", "rollout")):
                raise ValueError("70/30 requires demo and rollout frames in every Long task")
            if any(task in self.long and source not in ("demo", "rollout") for task, source in by_source):
                raise ValueError("70/30 Long pool must contain only demo and rollout")
        self._epoch, self._permutation = None, None

    def frame(self, draw):
        if self.mode == "sft":
            epoch, position = divmod(draw, self.total)
            if epoch != self._epoch:
                self._permutation = np.random.default_rng(stable_seed(self.seed, "shuffle", epoch)).permutation(
                    self.total
                )
                self._epoch = epoch
            return int(self._permutation[position])
        rng = np.random.default_rng(stable_seed(self.seed, "frame", int(draw)))
        probability = self.recipe["long_probability"] if self.recipe else 0.75
        group = self.long if rng.random() < probability else self.other
        task = group[int(rng.integers(len(group)))]
        spans, ends = self.tasks[task]
        if self.recipe and self.recipe["source_sampling"] == "demo_rollout":
            source_rng = np.random.default_rng(stable_seed(self.seed, "source", int(draw)))
            source = "demo" if source_rng.random() < self.recipe["demo_probability"] else "rollout"
            spans, ends = self.source_tasks[(task, source)]
        within = int(rng.integers(ends[-1]))
        episode = int(np.searchsorted(ends, within, side="right"))
        prior = 0 if episode == 0 else ends[episode - 1]
        return int(spans[episode][0] + within - prior)

    @classmethod
    def from_manifest(cls, path, seed, mode, *, recipe=None):
        manifest = json.loads(Path(path).read_text())
        episodes = manifest["episodes"]
        if any(e["source"] not in ("demo", "rollout", "intervention") for e in episodes):
            raise ValueError("evaluation trajectories cannot enter a training sampler")
        return cls(
            [e["length"] for e in episodes],
            [e["task_index"] for e in episodes],
            manifest["long_task_indices"],
            seed,
            mode,
            recipe=recipe,
            sources=[e["source"] for e in episodes],
        )


class DrawSampler:
    def __init__(self, stream, start_draw, stop_draw):
        self.stream, self.start_draw, self.stop_draw = stream, start_draw, stop_draw

    def __iter__(self):
        for draw in range(self.start_draw, self.stop_draw):
            yield draw, self.stream.frame(draw)

    def __len__(self):
        return self.stop_draw - self.start_draw


@contextmanager
def draw_randomness(seed):
    # Tokenizer condition dropout uses numpy; keep unrelated global RNG untouched.
    import torch

    np_state, py_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
    np.random.seed(seed)
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.set_rng_state(torch_state)


class DrawDataset:
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, draw_frame):
        draw, frame = draw_frame
        with draw_randomness(stable_seed(self.seed, "transform", int(draw))):
            return self.dataset[frame]


def collate(items):
    import jax

    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *items)


def worker_init(_):
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def batches(dataset, stream, batch_size, start_step, stop_step, workers=0):
    import torch

    generator = torch.Generator().manual_seed(stable_seed(stream.seed, "worker"))
    return torch.utils.data.DataLoader(
        DrawDataset(dataset, stream.seed),
        sampler=DrawSampler(stream, start_step * batch_size, stop_step * batch_size),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=collate,
        worker_init_fn=worker_init,
        multiprocessing_context="spawn" if workers else None,
        generator=generator,
        persistent_workers=workers > 0,
        drop_last=False,
    )
