import os
import json
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import src.models as model_module
import src.datasets as datasets_module

from src.collate import collate_fn
from src.trainers.trainer import Trainer
from src.utils.utils import WandbWriter, LocalWriter, inf_loop


def main(args):
    current_time = datetime.now().strftime("%d-%m-%Y--%H-%M")
    save_dir = f"saved/{current_time}"
    os.makedirs(save_dir, exist_ok=True)

    with open(args.config) as fin:
        config = json.load(fin)

    if args.wandb:
        if args.wandb_run_name:
            config["wandb"]["project"] = args.wandb_run_name
        writer = WandbWriter(**config["wandb"])
    else:
        writer = LocalWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dtr_config = config["data"]["train"]
    train_dataset = getattr(datasets_module, dtr_config["type"])(**dtr_config["dataset_args"])
    train_dataloader = inf_loop(DataLoader(train_dataset, collate_fn=collate_fn, **dtr_config["dataloader_args"]))

    dte_config = config["data"]["test"]
    test_dataset = getattr(datasets_module, dte_config["type"])(**dte_config["dataset_args"])
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, **dte_config["dataloader_args"])

    model = getattr(model_module, config["model"]["type"])(**config["model"]["args"])
    model = model.to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])

    max_lr, warmup_iters = config["lr_scheduler"]["linear_warmup"].values()
    linear_lambda = lambda iter: max_lr * (iter + 1) / warmup_iters
    gamma = config["lr_scheduler"]["exponential"]["gamma"]
    exponential_lambda = lambda iter: max_lr * (gamma ** (iter + 1 - warmup_iters))
    lr_lambda = lambda iter: linear_lambda(iter) if iter < warmup_iters else exponential_lambda(iter)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = Trainer(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        writer,
        save_dir,
        device,
        **config["trainer"],
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/train.json", type=str, help="config file path (default: configs/train.json)")
    parser.add_argument("-w", "--wandb", default=False, type=bool, help="send logs to wandb (default: False)")
    parser.add_argument("-wn", "--wandb-run-name", default=None, type=str, help="wandb run name (default: None)")
    args = parser.parse_args()
    main(args)
