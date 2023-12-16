from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from piq import FID, SSIMLoss

from src.utils.utils import make_train_image, make_test_image


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
        self.fid_metric = FID()
        self.fid_metric = SSIMLoss(data_range=255.0)

        self.img_size = config["generator"]["args"]["image_sz"]
        self.fixed_noize = torch.randn(len(test_dataloader), config["generator"]["args"]["hidden_dim"])

    def move_batch_to_device(self, batch):
        for key in ["img"]:
            batch[key] = batch[key].to(self.device)

    def train_epoch(self):
        self.g_model.train()
        self.d_model.train()
        fake_label, real_label = 0, 1
        sum_loss = 0
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
            errD_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
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
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
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
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            self.g_optimizer.step()

            if (batch_idx + 1) % self.log_every_step == 0:
                self.writer.log_image("train", make_train_image(batch["img"].detach().cpu().numpy()))

            if batch_idx == self.iterations_per_epoch:
                break

        return sum_loss / self.iterations_per_epoch

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
                samples = self.g_model(self.fixed_noize[:, last_idx : last_idx + bs].unsqueeze(-1).unsqueeze(-1))

                real_imgs.append(batch["img"].detach().cpu().numpy())
                constructed_imgs.append(samples.detach().cpu().numpy())

        real_imgs = np.stack(real_imgs)
        constructed_imgs = np.stack(constructed_imgs)

        self.writer.log({"test_FID": self.fid(real_imgs, constructed_imgs), "test_SSIM": self.ssim(real_imgs, constructed_imgs).item()})
        self.writer.log_image("test", make_train_image(constructed_imgs))

    def log_after_training_epoch(self, epoch, train_avg_loss):
        print(16 * "-")
        print(f"epoch:\t{epoch}")
        print(f"train_avg_loss:\t{train_avg_loss:.8f}")
        print(f"learning_rate:\t{self.lr_scheduler.get_last_lr()[0]:.8f}")
        print(16 * "-")

    def save_state(self, epoch):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(state, f"{self.save_dir}/checkpoint-{epoch}.pth")

    def train(self):
        """
        Training loop.
        """

        for epoch in tqdm(range(self.epochs)):
            train_avg_loss = self.train_epoch()
            self.test()

            self.log_after_training_epoch(epoch, train_avg_loss)
            self.save_state(epoch)

        self.writer.finish()
