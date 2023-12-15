from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from piq import FID, SSIMLoss

from src.utils.utils import make_train_image, make_test_image


def move_batch_to_device(batch, device):
    for key in ["img", "target"]:
        batch[key] = batch[key].to(device)


class Trainer:
    def __init__(
        self,
        model,
        train_inf_dataloader,
        test_dataloader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        writer,
        save_dir: str,
        device: torch.device,
        epochs: int,
        iterations_per_epoch: int,
        log_every_step: int = 1,
    ):
        self.model = model
        self.train_inf_dataloader = train_inf_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.save_dir = save_dir
        self.device = device
        self.epochs = epochs
        self.iterations_per_epoch = iterations_per_epoch
        self.log_every_step = log_every_step

        self.z = torch.randn(len(test_dataloader), self.model.latent_dim)
        self.fid_metric = FID()
        self.fid_metric = SSIMLoss(data_range=255.0)

    def train_epoch(self):
        self.model.train()
        sum_loss = 0
        for batch_idx, batch in tqdm(enumerate(self.train_inf_dataloader)):
            disc_model.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = disc_model(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = gen_model(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = disc_model(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            gen_model.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = disc_model(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            if (batch_idx + 1) % self.log_every_step == 0:
                self.writer.log_image("train", make_train_image(batch["img"].detach().cpu().numpy()))

            if batch_idx == self.iterations_per_epoch:
                break

        return sum_loss / self.iterations_per_epoch

    def test(self):
        self.model.eval()
        last_idx = 0
        real_imgs = []
        constructed_imgs = []
        targets = []
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                move_batch_to_device(batch, self.device)
                bs = batch["target"].shape[0]
                samples = self.model.sample(bs, batch["target"], z=self.z[last_idx : last_idx + bs, ...])

                real_imgs.append(batch["img"].detach().cpu().numpy())
                constructed_imgs.append(samples.detach().cpu().numpy())
                targets.append(batch["target"])

        real_imgs = np.stack(real_imgs)
        constructed_imgs = np.stack(constructed_imgs)

        self.writer.log({"test_FID": self.fid(real_imgs, constructed_imgs), "test_SSIM": self.ssim(real_imgs, constructed_imgs).item()})
        self.writer.log_image("test", make_test_image(constructed_imgs, np.cat(targets)))

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
