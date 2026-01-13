from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_fd001(raw_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    # Standard CMAPSS columns
    cols = (
        ["unit", "cycle"]
        + [f"op{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)]
    )

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=cols)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=cols)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    return train_df, test_df, rul_df


def compute_piecewise_rul_train(df: pd.DataFrame, rul_cap: int) -> pd.Series:
    """Row-level piecewise RUL for training set: RUL = max_cycle(unit)-cycle, capped."""
    out = np.zeros(len(df), dtype=np.float32)
    # groupby is safe because df is not too big
    for unit_id, u in df.groupby("unit", sort=False):
        max_cycle = int(u["cycle"].max())
        rul = (max_cycle - u["cycle"].values).astype(np.float32)
        rul = np.minimum(rul, float(rul_cap))
        out[u.index.values] = rul
    return pd.Series(out, index=df.index, name="RUL")


def select_feature_columns(
    train_df_raw: pd.DataFrame,
    rul_cap: int,
    selection_cfg: Dict[str, Any],
) -> List[str]:
    """
    Feature selection on RAW train_df (no normalization), to avoid std=1 artefacts.
    Modes:
      - all: keep all op1..op3 + s1..s21
      - variance: keep cols with std > threshold
      - spearman: keep top_k by abs spearman corr with piecewise RUL
    """
    base_cols = [c for c in train_df_raw.columns if c not in ("unit", "cycle")]
    mode = (selection_cfg.get("mode") or "all").lower()

    if mode == "all":
        return base_cols

    if mode == "variance":
        thr = float(selection_cfg.get("std_threshold", 1e-6))
        stds = train_df_raw[base_cols].astype(np.float32).std(axis=0).values
        keep = [c for c, s in zip(base_cols, stds) if float(s) > thr]
        if not keep:
            raise ValueError("Variance feature selection removed all features; lower std_threshold.")
        return keep

    if mode == "spearman":
        top_k = int(selection_cfg.get("top_k", 14))
        min_abs = float(selection_cfg.get("min_abs", 0.0))

        rul_row = compute_piecewise_rul_train(train_df_raw, rul_cap=rul_cap)

        scores = []
        for c in base_cols:
            # Spearman handles monotonic relationships
            corr = train_df_raw[c].corr(rul_row, method="spearman")
            if pd.isna(corr):
                corr = 0.0
            scores.append((c, abs(float(corr))))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Filter by min_abs then take top_k
        filtered = [c for c, s in scores if s >= min_abs]
        keep = filtered[:top_k] if len(filtered) >= top_k else filtered

        if not keep:
            raise ValueError("Spearman feature selection removed all features; lower min_abs or increase top_k.")
        return keep

    raise ValueError(f"Unknown feature selection mode: {mode}. Use all|variance|spearman.")


def fit_normalizer(train_df_raw: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = train_df_raw[cols].astype(np.float32).values
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def apply_normalizer(df_raw: pd.DataFrame, cols: List[str], mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    out = df_raw.copy()
    # Ensure float columns to avoid pandas dtype warnings
    out[cols] = out[cols].astype(np.float32)
    out[cols] = ((out[cols].values - mean) / std).astype(np.float32)
    return out


def rul_to_hi(rul: np.ndarray, rul_cap: int, hi_cfg: Dict[str, Any]) -> np.ndarray:
    """
    Health Index in [0,1], 1 = healthy, 0 = failed.
    Modes:
      - linear: HI = RUL / rul_cap
      - exp:    HI = exp(-(rul_cap - RUL)/tau)   (HI=1 at RUL=rul_cap, decays towards 0)
    """
    mode = (hi_cfg.get("mode") or "linear").lower()
    rul = np.clip(rul.astype(np.float32), 0.0, float(rul_cap))
    if mode == "linear":
        return (rul / float(rul_cap)).astype(np.float32)

    if mode == "exp":
        tau = float(hi_cfg.get("tau", 30.0))
        tau = max(tau, 1e-6)
        return np.exp(-(float(rul_cap) - rul) / tau).astype(np.float32)

    raise ValueError(f"Unknown HI mode: {mode}. Use linear|exp.")


def build_train_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    rul_cap: int,
    hi_cfg: Dict[str, Any],
    max_train_rul: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding windows for training units.
    Returns X, y_rul, y_hi, unit_ids
    """
    X_list, y_list, hi_list, unit_list = [], [], [], []

    for unit_id, u in df.groupby("unit", sort=False):
        u = u.sort_values("cycle").reset_index(drop=True)
        max_cycle = int(u["cycle"].max())
        rul_series = (max_cycle - u["cycle"].values).astype(np.float32)
        rul_series = np.minimum(rul_series, float(rul_cap)).astype(np.float32)

        hi_series = rul_to_hi(rul_series, rul_cap=rul_cap, hi_cfg=hi_cfg)

        data = u[feature_cols].values.astype(np.float32)

        if len(u) < seq_len:
            # no window possible
            continue

        for start in range(0, len(u) - seq_len + 1):
            end = start + seq_len
            y = float(rul_series[end - 1])
            if (max_train_rul is not None) and (y > float(max_train_rul)):
                continue
            X_list.append(data[start:end])
            y_list.append(y)
            hi_list.append(float(hi_series[end - 1]))
            unit_list.append(int(unit_id))

    if not X_list:
        raise ValueError("No training windows generated. Check seq_len and max_train_rul.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    hi = np.array(hi_list, dtype=np.float32)
    units = np.array(unit_list, dtype=np.int32)
    return X, y, hi, units


def build_last_points_train_trunc(
    df_full: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    rul_cap: int,
    hi_cfg: Dict[str, Any],
    trunc_rul: float,
    pad_short: bool,
    padding_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each unit in df_full (train units), truncate its trajectory to simulate test:
    keep data until the point where true RUL ~= trunc_rul, then take the last window.
    Returns X_last, y_rul_last, y_hi_last, unit_ids
    """
    X_list, y_list, hi_list, unit_list = [], [], [], []
    padding_mode = (padding_mode or "zeros").lower()

    for unit_id, u in df_full.groupby("unit", sort=False):
        u = u.sort_values("cycle").reset_index(drop=True)

        max_cycle = int(u["cycle"].max())
        rul_full = (max_cycle - u["cycle"].values).astype(np.float32)
        rul_full = np.minimum(rul_full, float(rul_cap)).astype(np.float32)
        hi_full = rul_to_hi(rul_full, rul_cap=rul_cap, hi_cfg=hi_cfg)

        # cut at cycle where RUL is about trunc_rul
        cut_cycle = max_cycle - int(trunc_rul)
        u_trunc = u[u["cycle"] <= cut_cycle].copy()
        if len(u_trunc) == 0:
            # If trunc is too aggressive, fall back to earliest available
            u_trunc = u.iloc[:1].copy()

        # Index in the FULL sequence for last truncated point
        last_cycle = int(u_trunc["cycle"].iloc[-1])
        idx_full = int(np.where(u["cycle"].values == last_cycle)[0][-1])

        y = float(rul_full[idx_full])
        hi = float(hi_full[idx_full])

        data_trunc = u_trunc[feature_cols].values.astype(np.float32)

        if len(data_trunc) >= seq_len:
            window = data_trunc[-seq_len:]
        else:
            if not pad_short:
                continue
            pad_len = seq_len - len(data_trunc)
            if padding_mode == "zeros":
                pad = np.zeros((pad_len, data_trunc.shape[1]), dtype=np.float32)
            elif padding_mode == "edge":
                pad = np.repeat(data_trunc[:1], repeats=pad_len, axis=0).astype(np.float32)
            else:
                raise ValueError("padding_mode must be zeros or edge")
            window = np.concatenate([pad, data_trunc], axis=0).astype(np.float32)

        X_list.append(window)
        y_list.append(y)
        hi_list.append(hi)
        unit_list.append(int(unit_id))

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    hi = np.array(hi_list, dtype=np.float32)
    units = np.array(unit_list, dtype=np.int32)
    return X, y, hi, units


def build_last_points_test(
    df: pd.DataFrame,
    rul_df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    rul_cap: int,
    hi_cfg: Dict[str, Any],
    pad_short: bool,
    padding_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each test unit, take the last available window (padding if needed).
    Target RUL is the provided RUL_FD001 value (at last observed cycle).
    """
    X_list, y_list, hi_list, unit_list = [], [], [], []
    padding_mode = (padding_mode or "zeros").lower()

    for unit_id, u in df.groupby("unit", sort=False):
        u = u.sort_values("cycle").reset_index(drop=True)
        max_cycle = int(u["cycle"].max())

        final_rul = float(rul_df.iloc[int(unit_id) - 1]["RUL"])
        # RUL series over observed test trajectory
        rul_series = (final_rul + (max_cycle - u["cycle"].values)).astype(np.float32)
        rul_series = np.minimum(rul_series, float(rul_cap)).astype(np.float32)
        hi_series = rul_to_hi(rul_series, rul_cap=rul_cap, hi_cfg=hi_cfg)

        data = u[feature_cols].values.astype(np.float32)

        if len(data) >= seq_len:
            window = data[-seq_len:]
        else:
            if not pad_short:
                continue
            pad_len = seq_len - len(data)
            if padding_mode == "zeros":
                pad = np.zeros((pad_len, data.shape[1]), dtype=np.float32)
            elif padding_mode == "edge":
                pad = np.repeat(data[:1], repeats=pad_len, axis=0).astype(np.float32)
            else:
                raise ValueError("padding_mode must be zeros or edge")
            window = np.concatenate([pad, data], axis=0).astype(np.float32)

        # last index corresponds to last observed cycle => target is final_rul (capped if needed)
        y = float(rul_series[-1])
        hi = float(hi_series[-1])

        X_list.append(window)
        y_list.append(y)
        hi_list.append(hi)
        unit_list.append(int(unit_id))

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    hi = np.array(hi_list, dtype=np.float32)
    units = np.array(unit_list, dtype=np.int32)
    return X, y, hi, units


def save_parquet_windows(X: np.ndarray, y_rul: np.ndarray, y_hi: np.ndarray, units: np.ndarray, path: Path) -> None:
    n, t, f = X.shape
    X_flat = X.reshape(n, t * f)
    df = pd.DataFrame(X_flat)
    df["RUL"] = y_rul.astype(np.float32)
    df["HI"] = y_hi.astype(np.float32)
    df["unit"] = units.astype(np.int32)
    df.to_parquet(path, index=False)


def preprocess_fd001(raw_root: Path, out_root: Path, cfg: dict) -> None:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    seq_len = int(data_cfg.get("seq_len", 30))
    rul_cap = int(data_cfg.get("rul_cap", 125))

    # This is the "piecewise linear" target used in a lot of FD001 papers:
    # RUL is capped at `rul_cap` (constant early), then decreases linearly.
    # (Your existing behavior already did this; here it’s explicit and consistent.)
    hi_cfg = data_cfg.get("hi", {"mode": "linear"})
    pad_short = bool(data_cfg.get("pad_short_sequences", True))
    padding_mode = str(data_cfg.get("padding_mode", "zeros"))

    # Validation is done as 1 point per engine, at a truncated point with RUL ~ trunc_rul
    val_units_frac = float(data_cfg.get("val_units_frac", 0.2))
    val_trunc_rul = float(data_cfg.get("val_trunc_rul", 70))

    # Optional: filter training windows to focus on degradation region
    max_train_rul = data_cfg.get("max_train_rul", None)
    if max_train_rul is not None:
        max_train_rul = float(max_train_rul)

    selection_cfg = (data_cfg.get("feature_selection") or {"mode": "all"})

    print(f"Preprocessing FD001 with seq_len={seq_len}, rul_cap={rul_cap}")

    train_df_raw, test_df_raw, rul_df = load_fd001(raw_root)

    # 1) Feature selection (raw)
    feature_cols = select_feature_columns(train_df_raw, rul_cap=rul_cap, selection_cfg=selection_cfg)

    # 2) Fit normalizer on raw train, apply to train+test
    mean, std = fit_normalizer(train_df_raw, cols=feature_cols)
    train_df = apply_normalizer(train_df_raw, cols=feature_cols, mean=mean, std=std)
    test_df = apply_normalizer(test_df_raw, cols=feature_cols, mean=mean, std=std)

    # Keep only needed columns (prevents mixing dtypes later)
    keep_cols = ["unit", "cycle"] + feature_cols
    train_df = train_df[keep_cols].copy()
    test_df = test_df[keep_cols].copy()

    # 3) Split validation by ENGINE units (no leakage)
    unit_ids = sorted(train_df["unit"].unique().tolist())
    n_units = len(unit_ids)
    n_val = max(1, int(round(val_units_frac * n_units)))
    val_units = set(unit_ids[-n_val:])
    tr_units = set(unit_ids[:-n_val])

    df_tr = train_df[train_df["unit"].isin(tr_units)].copy()
    df_val = train_df[train_df["unit"].isin(val_units)].copy()

    # 4) Build datasets
    X_tr, y_tr, hi_tr, units_tr = build_train_windows(
        df_tr,
        feature_cols=feature_cols,
        seq_len=seq_len,
        rul_cap=rul_cap,
        hi_cfg=hi_cfg,
        max_train_rul=max_train_rul,
    )

    X_val, y_val, hi_val, units_val = build_last_points_train_trunc(
        df_full=df_val,
        feature_cols=feature_cols,
        seq_len=seq_len,
        rul_cap=rul_cap,
        hi_cfg=hi_cfg,
        trunc_rul=val_trunc_rul,
        pad_short=pad_short,
        padding_mode=padding_mode,
    )

    X_test, y_test, hi_test, units_test = build_last_points_test(
        df=test_df,
        rul_df=rul_df,
        feature_cols=feature_cols,
        seq_len=seq_len,
        rul_cap=rul_cap,
        hi_cfg=hi_cfg,
        pad_short=pad_short,
        padding_mode=padding_mode,
    )

    out_root.mkdir(parents=True, exist_ok=True)

    train_path = out_root / f"fd001_train_windows_seq{seq_len}.parquet"
    val_path = out_root / f"fd001_val_trunc_last_seq{seq_len}.parquet"
    test_path = out_root / f"fd001_test_last_seq{seq_len}.parquet"
    meta_path = out_root / f"fd001_meta_seq{seq_len}.json"

    save_parquet_windows(X_tr, y_tr, hi_tr, units_tr, train_path)
    save_parquet_windows(X_val, y_val, hi_val, units_val, val_path)
    save_parquet_windows(X_test, y_test, hi_test, units_test, test_path)

    meta = {
        "dataset": "FD001",
        "seq_len": seq_len,
        "rul_cap": rul_cap,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "normalizer_mean": mean.astype(float).tolist(),
        "normalizer_std": std.astype(float).tolist(),
        "hi": hi_cfg,
        "padding_mode": padding_mode,
        "pad_short_sequences": pad_short,
        "val_units_frac": val_units_frac,
        "val_trunc_rul": val_trunc_rul,
        "max_train_rul": max_train_rul,
        "feature_selection": selection_cfg,
        "counts": {
            "train_windows": int(X_tr.shape[0]),
            "val_points": int(X_val.shape[0]),
            "test_points": int(X_test.shape[0]),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {meta_path}")
    print(f"Counts: train_windows={meta['counts']['train_windows']} | val_points={meta['counts']['val_points']} | test_points={meta['counts']['test_points']}")

    # Hard guard: benchmark must predict 100 engines on FD001 test.
    if meta["counts"]["test_points"] != 100:
        raise RuntimeError(
            f"FD001 test is typically 100 engines; test_points={meta['counts']['test_points']} indicates a problem (padding/reading)."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    root = Path(__file__).resolve().parents[3]
    raw_root = root / "data" / "raw" / "cmapss"
    out_root = root / "data" / "processed"

    preprocess_fd001(raw_root=raw_root, out_root=out_root, cfg=cfg)


if __name__ == "__main__":
    main()
