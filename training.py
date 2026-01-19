import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ContrastiveCIFAR10
from model import ContrastiveModel
from loss import nt_xent_loss


def training():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}\n")

    dataset = ContrastiveCIFAR10()
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = ContrastiveModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3
    print("Starting training...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            loader,
            desc=f"Epoch [{epoch+1}/{epochs}]",
            leave=False
        )

        for step, (x1, x2) in enumerate(progress_bar):
            x1, x2 = x1.to(device), x2.to(device)

            z1 = model(x1)
            z2 = model(x2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)

            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    training()
