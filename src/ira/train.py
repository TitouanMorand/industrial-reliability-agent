from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import math
import shutil

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

from src.ira.models.rul_model import CNNGRURUL


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_meta(processed_dir: Path, seq_len: int) -> dict:
    meta_path = processed_dir / f"fd001_meta_seq{seq_len}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}. Run preprocess first.")
    return json.loads(meta_path.read_text(encoding="utf-8"))


class ParquetWindowDataset(Dataset):
    """
    Expects parquet with:
      - flattened features columns
      - RUL, HI, unit
    """
    def __init__(self, path: Path, seq_len: int, n_features: int):
        df = pd.read_parquet(path)

        # Feature columns are numeric unnamed columns created by DataFrame(X_flat)
        target_cols = {"RUL", "HI", "unit"}
        feat_cols = [c for c in df.columns if c not in target_cols]

        X = df[feat_cols].values
        y_rul = df["RUL"].values.astype("float32")
        y_hi = df["HI"].values.astype("float32")

        self.X = torch.tensor(X, dtype=torch.float32).reshape(-1, seq_len, n_features)
        self.y_rul = torch.tensor(y_rul, dtype=torch.float32)
        self.y_hi = torch.tensor(y_hi, dtype=torch.float32)

    def __len__(self):
        return len(self.y_rul)

    def __getitem__(self, idx):
        return self.X[idx], self.y_rul[idx], self.y_hi[idx]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rmse_from_preds(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - y) ** 2)).item())


def build_sample_weights(y_rul: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Weight errors close to failure higher (small RUL => larger weight).
    Modes:
      - exp: w = 1 + alpha * exp(-RUL/tau)
      - inverse: w = 1 + alpha / (RUL + tau)
    """
    mode = (cfg.get("mode") or "exp").lower()
    alpha = float(cfg.get("alpha", 2.0))
    tau = float(cfg.get("tau", 20.0))
    tau = max(tau, 1e-6)

    if mode == "exp":
        w = 1.0 + alpha * torch.exp(-y_rul / tau)
        return w

    if mode == "inverse":
        w = 1.0 + alpha / (y_rul + tau)
        return w

    raise ValueError("near_failure_weighting.mode must be exp or inverse")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    config_path = args.config
    cfg = load_config(config_path)

    # Artifacts
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("artifacts") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.yaml")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    seq_len = int(data_cfg.get("seq_len", 30))

    device = (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    print("Device:", device)

    root = Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"

    meta = load_meta(processed, seq_len=seq_len)
    n_features = int(meta["n_features"])
    rul_cap = float(meta["rul_cap"])

    # Hard guard for benchmark comparability
    if int(meta["counts"]["test_points"]) != 100:
        raise RuntimeError(f"FD001 benchmark requires 100 test engines. Got {meta['counts']['test_points']}.")

    # Paths
    train_path = processed / f"fd001_train_windows_seq{seq_len}.parquet"
    val_path = processed / f"fd001_val_trunc_last_seq{seq_len}.parquet"
    test_path = processed / f"fd001_test_last_seq{seq_len}.parquet"

    train_ds = ParquetWindowDataset(train_path, seq_len=seq_len, n_features=n_features)
    val_ds = ParquetWindowDataset(val_path, seq_len=seq_len, n_features=n_features)
    test_ds = ParquetWindowDataset(test_path, seq_len=seq_len, n_features=n_features)

    batch_size = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    aux_hi = bool(train_cfg.get("aux_hi", True))
    hi_lambda = float(train_cfg.get("hi_lambda", 0.2))
    consistency_lambda = float(train_cfg.get("consistency_lambda", 0.1))

    temporal_agg = str(model_cfg.get("temporal_aggregation", "attention"))

    model = CNNGRURUL(
        n_features=n_features,
        seq_len=seq_len,
        cnn_channels=int(model_cfg.get("cnn_channels", 32)),
        cnn_kernel_size=int(model_cfg.get("cnn_kernel_size", 3)),
        gru_hidden_size=int(model_cfg.get("gru_hidden_size", 64)),
        gru_num_layers=int(model_cfg.get("gru_num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        head_hidden_size=int(model_cfg.get("head_hidden_size", 64)),
        head_dropout=float(model_cfg.get("head_dropout", 0.3)),
        temporal_aggregation=temporal_agg,
        aux_hi=aux_hi,
        bidirectional=bool(model_cfg.get("bidirectional", False)),
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Main loss type (computed per-sample if weighting enabled)
    loss_name = str(train_cfg.get("loss", "huber")).lower()
    huber_beta = float(train_cfg.get("huber_beta", 20.0))

    near_cfg = train_cfg.get("near_failure_weighting", {"enabled": True, "mode": "exp", "alpha": 2.0, "tau": 20.0})
    near_enabled = bool(near_cfg.get("enabled", True))

    best_val = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ------------------- TRAIN -------------------
        model.train()
        total_loss = 0.0
        total_rmse_acc = 0.0

        for X, y_rul, y_hi in train_loader:
            X = X.to(device)
            y_rul = y_rul.to(device)
            y_hi = y_hi.to(device)

            optim.zero_grad()

            out = model(X)
            if aux_hi:
                pred_rul, pred_hi = out
            else:
                pred_rul = out
                pred_hi = None

            # per-sample main loss
            if loss_name == "mse":
                per = (pred_rul - y_rul) ** 2
            elif loss_name == "huber":
                per = torch.nn.functional.smooth_l1_loss(
                    pred_rul, y_rul, beta=huber_beta, reduction="none"
                )
            else:
                raise ValueError("training.loss must be mse or huber")

            if near_enabled:
                w = build_sample_weights(y_rul.detach(), near_cfg).to(device)
                per = per * w

            main_loss = per.mean()

            loss = main_loss

            # Auxiliary HI head (Health Index)
            if aux_hi:
                aux_loss = torch.mean((pred_hi - y_hi) ** 2)
                loss = loss + hi_lambda * aux_loss

                # Consistency: HI ≈ RUL / rul_cap (only if HI is linear-like)
                # (harmless regularizer even if HI mode is exp; you can turn it off by setting lambda=0)
                rul_scaled = torch.clamp(pred_rul / rul_cap, 0.0, 1.0)
                cons_loss = torch.mean((pred_hi - rul_scaled) ** 2)
                loss = loss + consistency_lambda * cons_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg.get("grad_clip", 5.0)))
            optim.step()

            total_loss += float(loss.item())
            total_rmse_acc += float(torch.mean((pred_rul.detach() - y_rul) ** 2).item())

        train_loss = total_loss / max(1, len(train_loader))
        train_rmse = math.sqrt(total_rmse_acc / max(1, len(train_loader)))

        # ------------------- VAL (benchmark-like: 1 per engine) -------------------
        model.eval()
        val_mse_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for X, y_rul, _y_hi in val_loader:
                X = X.to(device)
                y_rul = y_rul.to(device)
                out = model(X)
                pred_rul = out[0] if aux_hi else out
                val_mse_sum += float(torch.mean((pred_rul - y_rul) ** 2).item())
                n_val_batches += 1

        val_mse = val_mse_sum / max(1, n_val_batches)
        val_rmse = math.sqrt(val_mse)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss({loss_name}): {train_loss:.4f} | train RMSE: {train_rmse:.2f} | "
            f"val RMSE(trunc-last): {val_rmse:.2f}"
        )

        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"BEST val RMSE(trunc-last): {best_val:.2f} @ epoch {best_epoch}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------- TEST (FD001 benchmark: 1 per engine) -------------------
    model.eval()
    test_mse_sum = 0.0
    n_test_batches = 0
    with torch.no_grad():
        for X, y_rul, _y_hi in test_loader:
            X = X.to(device)
            y_rul = y_rul.to(device)
            out = model(X)
            pred_rul = out[0] if aux_hi else out
            test_mse_sum += float(torch.mean((pred_rul - y_rul) ** 2).item())
            n_test_batches += 1

    test_mse = test_mse_sum / max(1, n_test_batches)
    test_rmse = math.sqrt(test_mse)
    print(f"TEST RMSE (FD001 benchmark, last-window): {test_rmse:.2f}")

    metrics = {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_rmse_trunc_last": best_val,
        "test_rmse_fd001_last_window": test_rmse,
        "n_parameters": count_parameters(model),
        "seq_len": seq_len,
        "n_features": n_features,
        "rul_cap": rul_cap,
        "feature_selection": meta.get("feature_selection", None),
        "hi_cfg": meta.get("hi", None),
        "loss": loss_name,
        "near_failure_weighting": near_cfg,
        "aux_hi": aux_hi,
        "hi_lambda": hi_lambda,
        "consistency_lambda": consistency_lambda,
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Append summary log
    summary = {"run_id": run_id, "config": config_path, **metrics}
    with open("runs_summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
