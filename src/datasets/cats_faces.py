import os

import torch
import torchvision
import torchvision.transforms.functional as TF


class CatsFaces:
    def __init__(self, root, train: bool = True, limit: int = None):
        if train:
            self.img_name = sorted(os.listdir(root))[: limit - 1]
        else:
            self.img_name = sorted(os.listdir(root))[-limit - 1 : -1]
        self.root = root

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        return TF.Normalize(
            torchvision.io.read_image(f"{self.root}/{self.img_name[index]}").to(torch.float32),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        )
