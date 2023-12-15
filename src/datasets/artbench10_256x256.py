import random
import os

import torch
import torchvision


DIRS = [
    "art_nouveau",
    "baroque",
    "expressionism",
    "impressionism",
    "post_impressionism",
    "realism",
    "renaissance",
    "romanticism",
    "surrealism",
    "ukiyo_e",
]


def read_dir(path):
    img_path = []
    target = []
    for t, subdir in enumerate(DIRS):
        if subdir[0] == ".":
            break
        for img_name in os.listdir(f"{path}/{subdir}"):
            img_path.append(f"{path}/{subdir}/{img_name}")
            target.append(t)
    return img_path, target


class ArtBench10_256x256:
    def __init__(self, root, train: bool = True, limit: int = None):
        subdir_name = "train" if train else "test"
        self.img_path, self.target = read_dir(f"{root}/{subdir_name}")
        random.seed(0)
        random.shuffle(self.img_path)
        random.seed(0)
        random.shuffle(self.target)
        self.img_path = self.img_path[:limit]
        self.target = self.target[:limit]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        return torchvision.io.read_image(self.img_path[index]).to(torch.float32), self.target[index]
