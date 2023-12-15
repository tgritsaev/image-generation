from tqdm import tqdm

import torch
import torch.nn as nn

from piq import FID, SSIMLoss

from src.utils.utils import make_one_img


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

        self.z = torch.randn(len(test_dataloader), self.model.latent_dim)
        self.fid_metric = FID()
        self.fid_metric = SSIMLoss(data_range=255.0)

    def train_epoch(self):
        self.model.train()
        sum_loss = 0
        for batch_idx, batch in tqdm(enumerate(self.train_inf_dataloader)):
            self.optimizer.zero_grad()

            move_batch_to_device(batch, self.device)
            output = self.model.train_batch(**batch)
            # model.train, because diffusion training pipeline differs from VAE training pipeline
            batch.update(output)

            loss = self.model.loss_function(**batch)
            self.writer.log({"train_loss": loss.item()})
            sum_loss += loss.item()
            loss.backward()

            self.writer.log({"learning_rate": self.lr_scheduler.get_last_lr()[0]})
            self.optimizer.step()
            self.lr_scheduler.step()

            if batch_idx == self.iterations_per_epoch:
                break

        return sum_loss / self.iterations_per_epoch

    def test(self):
        self.model.eval()
        last_idx = 0
        real_imgs = []
        constructed_imgs = []
        targets = []
        for batch in tqdm(self.test_dataloader):
            move_batch_to_device(batch, self.device)
            bs = batch["target"].shape[0]
            samples = self.model.sample(bs, batch["target"], z=self.z[last_idx : last_idx + bs, ...])

            real_imgs.append(batch["img"].cpu())
            constructed_imgs.append(samples.cpu())
            targets.append(batch["target"])

        real_imgs = torch.stack(real_imgs)
        constructed_imgs = torch.stack(constructed_imgs)

        self.writer.log({"test_FID": self.fid(real_imgs, constructed_imgs), "test_SSIM": self.ssim(real_imgs, constructed_imgs).item()})
        self.writer.log_image(show)

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
