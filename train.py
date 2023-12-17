import math
import os
import json
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import src.models as model_module
import src.datasets as datasets_module

from src.collate import collate_fn, collate_w_target_fn
from src.trainers.trainer import Trainer
from src.trainers.gan_trainer import GANTrainer
from src.utils.utils import WandbWriter, LocalWriter, inf_loop


def training_pipeline(args, config):
    current_time = datetime.now().strftime("%d-%m-%Y--%H-%M")
    save_dir = f"saved/{current_time}"
    os.makedirs(save_dir, exist_ok=True)

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
    train_dataloader = inf_loop(DataLoader(train_dataset, collate_fn=collate_w_target_fn, **dtr_config["dataloader_args"]))

    dte_config = config["data"]["test"]
    test_dataset = getattr(datasets_module, dte_config["type"])(**dte_config["dataset_args"])
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_w_target_fn, **dte_config["dataloader_args"])

    model = getattr(model_module, config["model"]["type"])(**config["model"]["args"])
    model = model.to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])

    warmup_iters = config["lr_scheduler"]["linear_warmup"]["warmup_iters"]
    sqrt_linear_lambda = lambda iter: math.sqrt((iter + 1) / warmup_iters)
    gamma = config["lr_scheduler"]["exponential"]["gamma"]
    exponential_lambda = lambda iter: (gamma ** (iter + 1 - warmup_iters))
    lr_lambda = lambda iter: sqrt_linear_lambda(iter) if iter < warmup_iters else exponential_lambda(iter)
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
        config,
        **config["trainer"],
    )

    trainer.train()


def gan_training_pipeline(args, config):
    current_time = datetime.now().strftime("%d-%m-%Y--%H-%M")
    save_dir = f"saved/{current_time}"
    os.makedirs(save_dir, exist_ok=True)

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

    g_model = getattr(model_module, config["generator"]["type"])(**config["generator"]["args"])
    g_model = g_model.to(device)
    print(g_model)
    d_model = getattr(model_module, config["discriminator"]["type"])(**config["discriminator"]["args"])
    d_model = d_model.to(device)
    print(d_model)

    g_optimizer = torch.optim.AdamW(g_model.parameters(), **config["generator_optimizer"])
    d_optimizer = torch.optim.AdamW(d_model.parameters(), **config["discriminator_optimizer"])

    g_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, **config["generator_lr_scheduler"])
    d_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, **config["discriminator_lr_scheduler"])

    trainer = GANTrainer(
        g_model,
        d_model,
        train_dataloader,
        test_dataloader,
        g_optimizer,
        d_optimizer,
        g_lr_scheduler,
        d_lr_scheduler,
        writer,
        save_dir,
        device,
        config,
        **config["trainer"],
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/train.json", type=str, help="config file path (default: configs/train.json)")
    parser.add_argument("-w", "--wandb", default=False, type=bool, help="send logs to wandb (default: False)")
    parser.add_argument("-wn", "--wandb-run-name", default=None, type=str, help="wandb run name (default: None)")
    args = parser.parse_args()
    with open(args.config) as fin:
        config = json.load(fin)

    if "generator" in config.keys():
        gan_training_pipeline(args, config)
    else:
        training_pipeline(args, config)
