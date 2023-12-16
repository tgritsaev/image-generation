import os

import torch
import torchvision


class CatsFaces:
    def __init__(self, root, train: bool = True, limit: int = None):
        if train:
            self.img_path = sorted(os.listdir(root))[:limit]
        else:
            self.img_path = sorted(os.listdir(root))[-limit:]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        return torchvision.io.read_image(self.img_path[index]).to(torch.float32)
