import torch
import torch.nn as nn
import torchvision.models as models
from facenet_pytorch import InceptionResnetV1


class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        ).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mean = nn.Linear(576, latent_dim)
        self.fc_logvar = nn.Linear(576, latent_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        h = x.view(x.size(0), -1)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


if __name__ == "__main__":
    image = torch.randn(1, 3, 512, 512)
    model = Encoder()
    m, v = model(image)
