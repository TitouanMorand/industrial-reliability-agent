from pathlib import Path
import argparse
import yaml
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Load raw CMAPSS FD001 data
# ---------------------------------------------------------------------
def load_fd001(raw_root: Path):
    """
    Load raw CMAPSS FD001 train and test files.

    Returns
    -------
    train_df : pd.DataFrame
    test_df  : pd.DataFrame
    rul_df   : pd.DataFrame (true RUL for test units)
    """
    data_dir = raw_root / "6. Turbofan Engine Degradation Simulation Data Set"

    train_path = data_dir / "train_FD001.txt"
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"

    # Column names (standard CMAPSS)
    cols = (
        ["unit", "cycle"]
        + [f"op{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)]
    )

    train_df = pd.read_csv(
        train_path, sep=r"\s+", header=None, names=cols
    )
    test_df = pd.read_csv(
        test_path, sep=r"\s+", header=None, names=cols
    )
    rul_df = pd.read_csv(
        rul_path, sep=r"\s+", header=None, names=["RUL"]
    )

    return train_df, test_df, rul_df


# ---------------------------------------------------------------------
# Windowing + RUL construction
# ---------------------------------------------------------------------
def make_windows(
    df: pd.DataFrame,
    seq_len: int,
    rul_cap: int,
    is_train: bool,
    rul_df: Optional[pd.DataFrame] = None,
):
    """
    Build sliding windows per engine unit.

    Each window:
      X : (seq_len, n_features)
      y : RUL at the last cycle of the window
    """

    feature_cols = [c for c in df.columns if c not in ("unit", "cycle")]
    X_list, y_list = [], []

    for unit_id, u in df.groupby("unit"):
        u = u.sort_values("cycle").reset_index(drop=True)

        if is_train:
            max_cycle = u["cycle"].max()
            rul_series = max_cycle - u["cycle"]
        else:
            # Test set: RUL given externally per unit
            final_rul = rul_df.iloc[unit_id - 1]["RUL"]
            rul_series = final_rul + (u["cycle"].max() - u["cycle"])

        # Apply RUL cap
        rul_series = np.minimum(rul_series.values, rul_cap)

        data = u[feature_cols].values

        for start in range(0, len(u) - seq_len + 1):
            end = start + seq_len
            X_list.append(data[start:end])
            y_list.append(rul_series[end - 1])

    X = np.stack(X_list)   # (N, seq_len, n_features)
    y = np.array(y_list)   # (N,)

    return X, y


# ---------------------------------------------------------------------
# Main preprocessing routine for FD001
# ---------------------------------------------------------------------
def preprocess_fd001(
    raw_root: Path,
    out_root: Path,
    seq_len: int,
    rul_cap: int,
):
    print(f"Preprocessing FD001 with seq_len={seq_len}, rul_cap={rul_cap}")

    train_df, test_df, rul_df = load_fd001(raw_root)

    # Train windows
    X_train, y_train = make_windows(
        train_df,
        seq_len=seq_len,
        rul_cap=rul_cap,
        is_train=True,
    )

    # Validation split (simple temporal split on windows)
    n_train = int(0.85 * len(X_train))
    X_tr, X_val = X_train[:n_train], X_train[n_train:]
    y_tr, y_val = y_train[:n_train], y_train[n_train:]

    # Test windows
    X_test, y_test = make_windows(
        test_df,
        seq_len=seq_len,
        rul_cap=rul_cap,
        is_train=False,
        rul_df=rul_df,
    )

    # Save as parquet (flattened)
    def save_parquet(X, y, path):
        n, t, f = X.shape
        X_flat = X.reshape(n, t * f)
        df = pd.DataFrame(X_flat)
        df["RUL"] = y
        df.to_parquet(path, index=False)

    out_root.mkdir(parents=True, exist_ok=True)

    train_path = out_root / f"fd001_train_seq{seq_len}.parquet"
    val_path   = out_root / f"fd001_val_seq{seq_len}.parquet"
    test_path  = out_root / f"fd001_test_seq{seq_len}.parquet"

    save_parquet(X_tr, y_tr, train_path)
    save_parquet(X_val, y_val, val_path)
    save_parquet(X_test, y_test, test_path)

    print("Saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    seq_len = config["model"]["seq_len"]
    rul_cap = config["data"]["rul_cap"]

    root = Path(__file__).resolve().parents[3]
    raw_root = root / "data" / "raw" / "cmapss"
    out_root = root / "data" / "processed"

    preprocess_fd001(
        raw_root=raw_root,
        out_root=out_root,
        seq_len=seq_len,
        rul_cap=rul_cap,
    )


if __name__ == "__main__":
    main()
