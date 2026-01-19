import torch
import torch.nn as nn
import torchvision.models as models

class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        backbone = models.resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.squeeze()
        z = self.projector(h)
        return z
