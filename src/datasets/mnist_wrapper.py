import torch
from torchvision.datasets import MNIST


class MNISTWrapper(MNIST):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        return img.unsqueeze(0).to(torch.float32), target
