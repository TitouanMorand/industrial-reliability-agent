from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.ira.models.rul_model import CNNGRURUL


SEQ_LEN = 50
N_FEATURES = 24
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3


class ParquetDataset(Dataset):
    def __init__(self, path: Path):
        df = pd.read_parquet(path)
        X = df.drop(columns=["RUL"]).values
        y = df["RUL"].values
        self.X = torch.tensor(X, dtype=torch.float32).view(-1, SEQ_LEN, N_FEATURES)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Device:", device)

    root = Path(__file__).resolve().parents[2]
    data = root / "data" / "processed"

    train_ds = ParquetDataset(data / "fd001_train.parquet")
    val_ds = ParquetDataset(data / "fd001_val.parquet")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = CNNGRURUL(N_FEATURES, SEQ_LEN).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += loss_fn(model(X), y).item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"train MSE: {train_loss:.2f} | val MSE: {val_loss:.2f}"
        )

    out = root / "artifacts"
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "rul_model.pt")
    print("Model saved to artifacts/rul_model.pt")


if __name__ == "__main__":
    main()
