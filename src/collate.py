from typing import List, Tuple

import torch


def collate_fn(batch: List[Tuple]):
    img = []
    target = []
    for i, t in batch:
        img.append(i)
        target.append(t)
    return {
        "img": torch.stack(img),
        "target": torch.tensor(target),
    }
