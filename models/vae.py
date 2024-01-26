import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu=mu, logvar=logvar)
        z = z.unsqueeze(-1).unsqueeze(-1)
        image = self.decoder(z)
        return image

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

if __name__ == "__main__":
    model = VAE()
    x = torch.randn(1, 3, 512, 512)
    y = model(x)