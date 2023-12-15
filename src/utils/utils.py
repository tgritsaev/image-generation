from itertools import repeat
import wandb

import numpy as np


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def make_one_img(samples, targets, n=3, target_cnt=10):
    used = [0 for i in range(targets.shape[0])]
    _, h, w, c = samples.shape
    mega_image = np.zeros((h * n, w * target_cnt, c))
    for t in range(target_cnt):
        for i in range(n):
            for idx in range(len(used)):
                if targets[idx] == t and used[idx] == 0:
                    used[idx] = 1
                    mega_image[h * i : h * (i + 1), w * t : w * (t + 1), :] = samples[idx]
                    break
    return mega_image


class LocalWriter:
    def __init__(self):
        pass

    def log(self, msg):
        print(f"local_wandb log: {msg}")

    def log_table(self, table):
        print(f"local_wandb table: {table}")

    def log_img(self, img):
        print(f"local_wandb img: {img}")

    def finish(self):
        pass


class WandbWriter:
    def __init__(self, project: str = None, name: str = None):
        print(f"wandb project: {project}, name: {name}")
        wandb.init(project=project, name=name)

    def log(self, msg):
        wandb.log(msg)

    def log_table(self, table):
        wandb.log({"train": wandb.Table(data=table, columns=["pred", "target"])})

    def log_img(self, img):
        wandb.log({"samples": wandb.Image(img)})

    def finish(self):
        wandb.finish()
