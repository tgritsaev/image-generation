import torch
from torch import nn
from torch.nn import functional as F

from typing import List
from torch import Tensor

from src.models.base_model import BaseModel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ConditionalVAE(BaseModel):
    def __init__(
        self,
        n_channels: int,
        num_classes: int,
        latent_dim: int,
        img_size: int,
        hidden_dims: List = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(n_channels, n_channels, kernel_size=1)

        if hidden_dims is None:
            hidden_dims = [32, 32, 64, 128, 256, 512, 1024]

        in_channels = n_channels + 1  # +1 for target label
        # Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.ModuleList(modules)

        self.fc_mu = nn.Linear(4 * hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(4 * hidden_dims[-1], latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim + num_classes, 4 * hidden_dims[-1])
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.ModuleList(modules)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=n_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.hidden_dims = hidden_dims
        self.apply(weights_init)

    def encode(self, img: Tensor) -> List[Tensor]:
        x = img
        xs = []
        for downsample in self.encoder:
            x = downsample(x)
            xs.append(x)
        latent = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)

        return xs, mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        z = self.decoder_input(z).view(z.shape[0], -1, 2, 2)
        zs = []
        for i in range(len(self.decoder)):
            z = self.decoder[i](z)
        result = self.head(z)
        return (result,)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, img: Tensor, target: Tensor) -> List[Tensor]:
        y = F.one_hot(target, self.num_classes).float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(img)

        x = torch.cat([embedded_input, embedded_class], dim=1)
        xs, mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y], dim=1)
        pred, zs = self.decode(z)
        return {"pred": pred, "xs": xs, "zs": zs, "img": img, "mu": mu, "log_var": log_var}

    def train_batch(self, **kwargs) -> List[Tensor]:
        return self.forward(**kwargs)

    def loss_function(self, pred, xs, zs, img, mu, log_var, **kwargs) -> dict:
        xs.reverse()
        feature_maps_loss = 0
        for i in range(xs):
            print(xs[i].shape, zs[i].shape)
            feature_maps_loss += F.mse_loss(xs[i], zs[i])
        reconstruction_loss = F.mse_loss(pred, img)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = reconstruction_loss + img.shape[0] * kld_loss
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kld": kld_loss,
            "feature_maps_loss": feature_maps_loss,
        }

    def sample(self, num_samples: int, target: Tensor, z=None) -> Tensor:
        y = F.one_hot(target, self.num_classes).float()
        if z is None:
            z = torch.randn(num_samples, self.latent_dim)
        z = z.to(target.device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples
