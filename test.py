import json
import argparse

import torch

from bhw2.src.unet import Transformer


def main(args):
    with open(args.config) as fin:
        config = json.load(fin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = Transformer(**config["model"])
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        while prefix := input("Write prefix, the model will continue: "):
            print(f"--------\nprefix:\n{prefix}")
            print(f"generated:\n{model.inference(prefix, 0.6)}\n-----------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/test.json", type=str, help="config file path (default: configs/test.json)")
    parser.add_argument("-p", "--checkpoint-path", default="checkpoint.pth", type=str, help="checkpoint path (default: checkpoint.pth)")
    parser.add_argument("-t", "--temperature", default=0.6, type=float, help="sampling temperature (default: 0.6)")
    args = parser.parse_args()
    main(args)
