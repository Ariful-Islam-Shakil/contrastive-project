import torch.nn as nn
import torchvision.models as models

class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        backbone = models.resnet18(pretrained=False)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

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
