from itertools import repeat
import wandb

import numpy as np


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def make_train_image(samples: np.array, n: int = 8):
    _, c, h, w = samples.shape
    mega_image = np.zeros((h, w * n, c))
    for i in range(n):
        mega_image[0:h, i * w : (i + 1) * w, :] = samples[i].transpose(1, 2, 0)
    return mega_image


def make_mega_image(samples: np.array, n: int = 8):
    _, c, h, w = samples.shape
    mega_image = np.zeros((h * n, w * n, c))
    for i in range(n):
        for j in range(n):
            mega_image[j * h : (j + 1) * h, i * w : (i + 1) * w, :] = samples[i * n + j].transpose(1, 2, 0)
    return mega_image


def make_test_image(samples: np.array, targets: np.array, n: int = 3, target_cnt: int = 10):
    used = [0 for _ in range(targets.shape[0])]
    _, c, h, w = samples.shape
    mega_image = np.zeros((h * n, w * target_cnt, c))
    for t in range(target_cnt):
        for i in range(n):
            for idx in range(len(used)):
                if targets[idx] == t and used[idx] == 0:
                    used[idx] = 1
                    mega_image[h * i : h * (i + 1), w * t : w * (t + 1), :] = samples[idx].transpose(1, 2, 0)
                    break
    return mega_image


class LocalWriter:
    def __init__(self):
        pass

    def log(self, msg, commit=True):
        print(f"local_wandb log: {msg}")

    def finish(self):
        pass


class WandbWriter:
    def __init__(self, project: str = None, name: str = None):
        print(f"wandb project: {project}, name: {name}")
        wandb.init(project=project, name=name)

    def log(self, msg, commit=True):
        wandb.log(msg, commit=commit)

    def finish(self):
        wandb.finish()
