import torch
from torch.utils.data import DataLoader
from dataset import ContrastiveDataset
from model import ContrastiveModel
from loss import contrastive_loss

def training():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ContrastiveDataset("dataset/images")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ContrastiveModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50

    for epoch in range(epochs):
        total_loss = 0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)

            z1 = model(x1)
            z2 = model(x2)

            loss = contrastive_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}")
if __name__ == "__main__":
    training()