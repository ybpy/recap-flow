"""A draw-indexed stream: prefetch cannot change the next resumed batch."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
from openpi_client.e0.protocol import stable_seed


class FrameStream:
    def __init__(self, lengths, task_ids, long_tasks, seed, mode):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.ends = np.cumsum(self.lengths)
        self.seed, self.mode = seed, mode
        self.total = int(self.ends[-1])
        if mode not in ("sft", "recap"):
            raise ValueError(mode)
        by_task = defaultdict(list)
        offset = 0
        for length, task in zip(lengths, task_ids, strict=True):
            by_task[int(task)].append((offset, int(length)))
            offset += int(length)
        self.tasks = {}
        for task, spans in by_task.items():
            self.tasks[task] = (spans, np.cumsum([n for _, n in spans]))
        self.long = sorted(set(map(int, long_tasks)))
        self.other = sorted(set(self.tasks) - set(self.long))
        if mode == "recap" and (len(self.long) != 10 or len(self.other) != 30):
            raise ValueError("RECAP requires all 10 Long and 30 other tasks")
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
        group = self.long if rng.random() < 0.75 else self.other
        task = group[int(rng.integers(len(group)))]
        spans, ends = self.tasks[task]
        within = int(rng.integers(ends[-1]))
        episode = int(np.searchsorted(ends, within, side="right"))
        prior = 0 if episode == 0 else ends[episode - 1]
        return int(spans[episode][0] + within - prior)

    @classmethod
    def from_manifest(cls, path, seed, mode):
        manifest = json.loads(Path(path).read_text())
        episodes = manifest["episodes"]
        if any(e["source"] in ("development", "final") for e in episodes):
            raise ValueError("evaluation trajectories cannot enter a training sampler")
        return cls(
            [e["length"] for e in episodes],
            [e["task_index"] for e in episodes],
            manifest["long_task_indices"],
            seed,
            mode,
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
