import random
import os

import torch
import torchvision
import torchvision.transforms.functional as TF


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
        # complex logic as long as we want all classes to be presented.
        if limit:
            img_path = []
            target = []
            for i in range(10):
                plus = 1 if i < limit % 10 else 0
                cnt = 0
                for j in range(len(self.target)):
                    if cnt == (limit // 10) + plus:
                        break
                    if self.target[j] == i:
                        img_path.append(self.img_path[j])
                        target.append(i)
                        cnt += 1
            assert len(img_path) == limit
            self.img_path = img_path
            self.target = target

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        return (
            TF.normalize(
                torchvision.io.read_image(self.img_path[index]).to(torch.float32) / 255.0,
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
            self.target[index],
        )
