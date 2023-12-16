from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from piq import FID, SSIMLoss

from src.utils.utils import make_train_image, make_mega_image


class GANTrainer:
    def __init__(
        self,
        g_model,
        d_model,
        train_inf_dataloader,
        test_dataloader,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        g_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        d_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        writer,
        save_dir: str,
        device: torch.device,
        config,
        epochs: int,
        iterations_per_epoch: int,
        log_every_step: int = 1,
    ):
        self.g_model = g_model
        self.d_model = d_model

        self.train_inf_dataloader = train_inf_dataloader
        self.test_dataloader = test_dataloader

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_lr_scheduler = g_lr_scheduler
        self.d_lr_scheduler = d_lr_scheduler

        self.writer = writer
        self.save_dir = save_dir
        self.device = device
        self.epochs = epochs
        self.iterations_per_epoch = iterations_per_epoch
        self.log_every_step = log_every_step

        self.criterion = nn.BCELoss()
        self.fid_metric = FID().to(self.device)
        self.ssim_metric = SSIMLoss(data_range=255.0).to(self.device)

        self.img_size = config["generator"]["args"]["image_sz"]
        self.fixed_noize = torch.randn(len(test_dataloader), config["generator"]["args"]["hidden_dim"], device=self.device)

    def move_batch_to_device(self, batch):
        for key in ["img"]:
            batch[key] = batch[key].to(self.device)

    def train_epoch(self):
        self.g_model.train()
        self.d_model.train()
        fake_label, real_label = 0, 1
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        for batch_idx, batch in tqdm(enumerate(self.train_inf_dataloader)):
            self.d_model.zero_grad()
            # Format batch
            self.move_batch_to_device(batch)
            b_size = batch["img"].size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
            # Forward pass real batch through D
            output = self.d_model(batch["img"]).view(-1)
            # Calculate loss on all-real batch
            d_real_loss = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            d_real_loss.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, self.img_size, 1, 1, device=self.device)
            # Generate fake image batch with G
            fake = self.g_model(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = self.d_model(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            d_fake_loss = self.criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            d_fake_loss.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            d_loss = d_real_loss + d_fake_loss
            log_wandb = {"train_d_loss": d_loss.item()}
            # Update D
            self.d_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.g_model.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.d_model(fake).view(-1)
            # Calculate G's loss based on this output
            g_loss = self.criterion(output, label)
            # Calculate gradients for G
            g_loss.backward()
            log_wandb.update({"train_g_loss": g_loss.item()})
            D_G_z2 = output.mean().item()
            # Update G
            self.g_optimizer.step()

            log_wandb.update({"g_learning_rate": self.g_lr_scheduler.get_last_lr()[0]})
            log_wandb.update({"d_learning_rate": self.d_lr_scheduler.get_last_lr()[0]})
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()

            self.writer.log(log_wandb)
            if (batch_idx + 1) % self.log_every_step == 0:
                self.writer.log_image("train", make_train_image(fake.detach().cpu().numpy(), 8))

            if batch_idx == self.iterations_per_epoch:
                break

    def test(self):
        self.g_model.eval()
        self.d_model.eval()
        last_idx = 0
        real_imgs = []
        constructed_imgs = []
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                self.move_batch_to_device(batch)
                bs = batch["img"].shape[0]
                print("??", bs, last_idx, last_idx + bs)
                print("??", self.fixed_noize[:, last_idx : last_idx + bs].unsqueeze(-1).unsqueeze(-1).shape)
                samples = self.g_model(self.fixed_noize[:, last_idx : last_idx + bs].unsqueeze(-1).unsqueeze(-1))

                real_imgs.append(batch["img"].detach())
                constructed_imgs.append(samples.detach())
                print("!", real_imgs[-1].shape, constructed_imgs[-1].shape)

        real_imgs = torch.cat(real_imgs)
        constructed_imgs = torch.cat(constructed_imgs)
        print("!!!!!", real_imgs.shape, constructed_imgs.shape)
        self.writer.log(
            {
                "test_FID": self.fid_metric.compute_metric(real_imgs.flatten(1), constructed_imgs.flatten(1)),
                "test_SSIM": self.ssim_metric(real_imgs, constructed_imgs).item(),
            }
        )
        self.writer.log_image("test", make_mega_image(constructed_imgs.numpy(), 8))

    def log_after_training_epoch(self, epoch):
        print(16 * "-")
        print(f"epoch:\t{epoch}")
        print(f"learning_rate:\t{self.g_lr_scheduler.get_last_lr()[0]:.8f}")
        print(16 * "-")

    def save_state(self, epoch):
        state = {
            "epoch": epoch,
            "g_state_dict": self.g_model.state_dict(),
            "d_state_dict": self.d_model.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_lr_scheduler": self.g_lr_scheduler.state_dict(),
            "d_lr_scheduler": self.d_lr_scheduler.state_dict(),
        }
        torch.save(state, f"{self.save_dir}/checkpoint-{epoch}.pth")

    def train(self):
        """
        Training loop.
        """

        for epoch in tqdm(range(self.epochs)):
            self.train_epoch()
            self.log_after_training_epoch(epoch)
            self.test()

            self.save_state(epoch)

        self.writer.finish()
