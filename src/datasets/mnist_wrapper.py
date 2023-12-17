import torch
import torchvision
from torchvision.datasets import MNIST


class MNISTWrapper(MNIST):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resize = torchvision.transforms.Resize(size=(32, 32))

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        return self.resize(img.unsqueeze(0)).to(torch.float32), target
