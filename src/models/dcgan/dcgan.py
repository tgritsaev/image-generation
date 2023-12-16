import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_channels):
        super().__init__()

        self.layers = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),
            # state size. ``(hidden_dim * 8) x 4 x 4``
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # state size. ``(hidden_dim * 4) x 8 x 8``
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # state size. ``(hidden_dim * 2) x 16 x 16``
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # state size. ``(hidden_dim) x 32 x 32``
            nn.ConvTranspose2d(hidden_dim, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(n_channels) x 64 x 64``
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.layers(input)


class Discriminator(nn.Module):
    def __init__(self, hidden_dim, n_channels):
        super().__init__()

        self.layers = nn.Sequential(
            # input is ``(n_channels) x 64 x 64``
            nn.Conv2d(n_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(hidden_dim) x 32 x 32``
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(hidden_dim * 2) x 16 x 16``
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(hidden_dim * 4) x 8 x 8``
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(hidden_dim * 8) x 4 x 4``
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.layers(input)
