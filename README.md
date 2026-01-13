# Industrial Reliability Agent — NASA CMAPSS RUL Prediction (FD001)

Predict **Remaining Useful Life (RUL)** for turbofan engines using multivariate sensor time series from the **NASA CMAPSS** benchmark (FD001).

This repo focuses on:
- **Reproducible experiments** driven by YAML configs
- **Paper-comparable evaluation** (FD001: 100 test engines, last-window RMSE)
- A lightweight, explainable model pipeline (CNN/GRU + structured targets)

---

## Best result (current)

**Experiment 11 — Health Index–Driven RUL Prediction (FD001 benchmark)**  
- Temporal window: **75**
- RUL cap (piecewise target): **125**
- Feature selection: **Spearman correlation (top features)**
- Model: **CNN + GRU + temporal attention**
- Objective: **Huber + HI auxiliary supervision + near-failure weighting**
- **Test RMSE (FD001 benchmark, last-window): 17.88 cycles**

Config: `configs/11_HI_optimisation.yaml`  
Artifacts: `artifacts/<run_id>/` (includes `config.yaml`, `metrics.json`, `model.pt`)

---

## Quickstart

### 1) Environment
Create and activate a virtualenv, then install deps:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Dataset
Place the raw CMAPSS data under:
```text
data/raw/cmapss/6. Turbofan Engine Degradation Simulation Data Set/
  train_FD001.txt
  test_FD001.txt
  RUL_FD001.txt
```

### 3) Preprocess + Train
```bash
python3 -m src.ira.data.preprocess --config configs/11_HI_optimisation.yaml
python3 -m src.ira.train           --config configs/11_HI_optimisation.yaml
```

---

## How evaluation works (FD001 benchmark)

To match common FD001 benchmarks:

- Training uses **sliding windows** (many windows per engine)
- Validation and test use **one window per engine**: the **last available window**
- RMSE is computed over **100 test engines** (FD001 standard)

This avoids “window-wise” metrics that are not comparable to the literature.

---

## Project structure

```text
src/
  ira/
    data/
      preprocess.py        # CMAPSS loading + windowing + feature selection + saving parquet
    models/
      rul_model.py         # CNN/GRU backbone + temporal aggregation + heads
    train.py               # training loop + benchmark evaluation + artifact logging
configs/
  *.yaml                   # experiment configs (data, model, training)
data/
  raw/                     # CMAPSS raw files (not committed)
  processed/               # parquet outputs from preprocessing
artifacts/
  <run_id>/                # per-run config + metrics + weights
runs_summary.jsonl         # appended run summaries
EXPERIMENTS.md             # experiment log
```

---

## Key modeling choices

- **Piecewise RUL**: `RUL = min(RUL_true, RUL_cap)` to remove uninformative early-life variance
- **Health Index (HI)**: bounded proxy of degradation (e.g., `HI = RUL / RUL_cap`) used as auxiliary supervision
- **Near-failure weighting**: penalize errors more when true RUL is small
- **Feature selection**: Spearman correlation against piecewise RUL to remove low-signal sensors

---

## Reproducibility

Each training run writes:
- `artifacts/<run_id>/config.yaml`
- `artifacts/<run_id>/metrics.json`
- `artifacts/<run_id>/model.pt`

and appends a line to `runs_summary.jsonl`.

---

## Roadmap (future work)

- Sensor-wise attention / feature gating (SE blocks)
- BiGRU vs GRU
- Temporal Convolutional Networks (TCN)
- HI-only training + inversion to RUL
- PHM/NASA scoring (asymmetric penalties)


