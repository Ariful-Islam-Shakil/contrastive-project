import torch
from torchvision import datasets, transforms

class ContrastiveCIFAR10(torch.utils.data.Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor()
        ])

        self.dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        x1 = self.transform(img)
        x2 = self.transform(img)

        return x1, x2
