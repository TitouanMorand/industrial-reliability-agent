from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.ira.models.rul_model import CNNGRURUL
import yaml
import json
import shutil
from datetime import datetime
import math

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ParquetDataset(Dataset):
    def __init__(self, path: Path, seq_len: int, n_features: int):
        df = pd.read_parquet(path)

        X = df.drop(columns=["RUL"]).values
        y = df["RUL"].values

        self.seq_len = seq_len
        self.n_features = n_features

        self.X = (
            torch.tensor(X, dtype=torch.float32)
            .reshape(-1, seq_len, n_features)
        )
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("artifacts") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, run_dir / "config.yaml")

    SEQ_LEN = config["model"]["seq_len"]
    N_FEATURES = config["model"]["n_features"]

    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LR = config["training"]["learning_rate"]

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Device:", device)

    root = Path(__file__).resolve().parents[2]
    data = root / "data" / "processed"

    train_ds = ParquetDataset(
        data / f"fd001_train_seq{SEQ_LEN}.parquet",
        seq_len=SEQ_LEN,
        n_features=N_FEATURES,
    )
    val_ds = ParquetDataset(
        data / f"fd001_val_seq{SEQ_LEN}.parquet",
        seq_len=SEQ_LEN,
        n_features=N_FEATURES,
    )



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

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = {
        "train_mse": train_loss,
        "val_mse": val_loss,
        "train_rmse": math.sqrt(train_loss),
        "val_rmse": math.sqrt(val_loss),
        "epochs": EPOCHS,
        "n_parameters": count_parameters(model),
    }


    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(model.state_dict(), run_dir / "model.pt")

    print(f"Run saved to {run_dir}")

    summary = {
    "run_id": run_id,
    "config": config_path,
    **metrics,
    }

    with open("runs_summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")



if __name__ == "__main__":
    main()
