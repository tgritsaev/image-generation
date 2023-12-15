from itertools import repeat
import wandb

import numpy as np


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def make_train_image(samples: np.array, n: int = 10):
    _, h, w, c = samples.shape
    mega_image = np.zeros((h * n, w, c))
    for i in range(n):
        print(samples[i].shape)
        mega_image[i * h : (i + 1) * h, 0:w, :] = samples[i]
    return mega_image


def make_test_image(samples: np.array, targets: np.array, n: int = 3, target_cnt: int = 10):
    used = [0 for _ in range(targets.shape[0])]
    _, h, w, c = samples.shape
    mega_image = np.zeros((h * target_cnt, w * n, c))
    for t in range(target_cnt):
        for i in range(n):
            for idx in range(len(used)):
                if targets[idx] == t and used[idx] == 0:
                    used[idx] = 1
                    mega_image[h * t : h * (t + 1), w * i : w * (i + 1), :] = samples[idx]
                    break
    return mega_image.transpose(1, 2, 0)


class LocalWriter:
    def __init__(self):
        pass

    def log(self, msg):
        print(f"local_wandb log: {msg}")

    def log_table(self, table):
        print(f"local_wandb table: {table}")

    def log_image(self, part, img):
        print(f"local_wandb {part}_img: {img}")

    def finish(self):
        pass


class WandbWriter:
    def __init__(self, project: str = None, name: str = None):
        print(f"wandb project: {project}, name: {name}")
        wandb.init(project=project, name=name)

    def log(self, msg):
        wandb.log(msg)

    def log_image(self, part, img):
        wandb.log({f"{part}_samples": wandb.Image(img)})

    def finish(self):
        wandb.finish()
