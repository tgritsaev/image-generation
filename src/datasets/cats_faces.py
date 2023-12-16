import os

import torch
import torchvision


class CatsFaces:
    def __init__(self, root, train: bool = True, limit: int = None):
        if train:
            self.img_name = sorted(os.listdir(root))[:limit]
        else:
            self.img_name = sorted(os.listdir(root))[-limit:]
        self.root = root

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        print(self.img_name[index])
        return torchvision.io.read_image(f"{self.root}/{self.img_name[index]}").to(torch.float32)
