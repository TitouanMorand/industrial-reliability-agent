# Experiments — Industrial Reliability Agent (CMAPSS FD001)

This document summarizes the main experiments conducted on the NASA CMAPSS FD001 dataset.
The objective is to predict the Remaining Useful Life (RUL) of turbofan engines from multivariate sensor time series.

All experiments are fully reproducible using the corresponding YAML configuration files.
Each run is tracked with its configuration, metrics, and trained model weights.

---

## Experiment 1 — Baseline CNN-GRU (seq_len = 50)

**Configuration**
- Config file: `configs/baseline.yaml`
- Temporal window: 50 cycles
- Model: CNN + GRU + regression head
- Device: Apple MPS (M1)

**Results**
- Train MSE: ~1699
- Validation MSE: ~1818
- Train RMSE: ~41.2 cycles
- Validation RMSE: ~42.6 cycles

**Observations**
- The model converges quickly within a few epochs.
- Training and validation errors are close, indicating limited overfitting.
- Performance degrades slightly near end-of-life, which is expected with a relatively short temporal context.

---

## Experiment 2 — Longer Temporal Window (seq_len = 75)

**Configuration**
- Config file: `configs/longer_window.yaml`
- Temporal window: 75 cycles
- Model: CNN + GRU + regression head
- Device: Apple MPS (M1)

**Results**
- Train MSE: ~1582
- Validation MSE: ~1786
- Train RMSE: ~39.8 cycles
- Validation RMSE: ~42.3 cycles

**Observations**
- Increasing the temporal window improves training error and slightly improves validation performance.
- The model benefits from longer historical context, especially during early degradation phases.
- Training remains stable despite increased sequence length, at the cost of slightly higher computational load.

---