import torch.nn as nn
import torch.nn.functional as F


class UpConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
    ):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.presample = nn.ConvTranspose2d(
            in_channels=latent_dim, out_channels=512, kernel_size=4, stride=1
        )  # 4
        self.upconv1 = UpConv(512, 512)  # 8
        self.upconv2 = UpConv(512, 256)  # 16
        self.upconv3 = UpConv(256, 256)  # 32
        self.upconv4 = UpConv(256, 128)  # 64
        self.upconv5 = UpConv(128, 64)  # 128
        self.upconv6 = UpConv(64, 3)  # 256

    def forward(self, x):
        x = self.presample(x)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        x = F.relu(self.upconv5(x))
        x = F.relu(self.upconv6(x))
        return x


if __name__ == "__main__":
    import torch

    x = torch.randn(1, 100, 1, 1)
    model = Decoder(latent_dim=100)
    y = model(x)
