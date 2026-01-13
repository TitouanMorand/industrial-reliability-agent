# Experiments — Industrial Reliability Agent (CMAPSS FD001)

This document summarizes the main experiments conducted on the **NASA CMAPSS FD001** dataset.
The objective is to predict the **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor time series.

All experiments are **reproducible** using the corresponding YAML configuration files.
Each run is tracked with its **configuration**, **metrics**, and **trained model weights** under `artifacts/<run_id>/`.

---

## Notes on evaluation protocols

Two evaluation protocols appear in the history of this project:

- **Window-wise evaluation (legacy)**: RMSE computed over *all sliding windows* (many samples per engine).  
  Useful for debugging training stability, but **not comparable** to FD001 literature benchmarks.

- **FD001 benchmark evaluation (paper-comparable)**: **one prediction per engine** on the **last available window** (100 test engines in FD001), then compute RMSE over those 100 points.  
  This is the protocol used in **Experiment 11** and should be considered the reference moving forward.

---

## Experiment 1 — Baseline CNN-GRU (seq_len = 50)

**Configuration**
- Config file: `configs/baseline.yaml`
- Temporal window: 50 cycles
- Model: CNN + GRU + regression head
- Device: Apple MPS (M1)

**Results** *(window-wise evaluation)*
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

**Results** *(window-wise evaluation)*
- Train MSE: ~1582
- Validation MSE: ~1786
- Train RMSE: ~39.8 cycles
- Validation RMSE: ~42.3 cycles

**Observations**
- Increasing the temporal window improves training error and slightly improves validation performance.
- The model benefits from longer historical context, especially during early degradation phases.
- Training remains stable despite increased sequence length, at the cost of slightly higher computational load.

---

## Experiment 3 — Longer Regression Head (seq_len = 50)

**Configuration**
- Config file: `configs/baseline.yaml` (regression head deepened)
- Temporal window: 50 cycles
- Model: CNN + GRU + deeper regression head
- Device: Apple MPS (M1)

**Results** *(window-wise evaluation; later pipeline versions report smaller MSE scales due to refactors)*
- Train MSE: ~26.2
- Validation MSE: ~29.1
- Train RMSE: ~41.0 cycles
- Validation RMSE: ~43.5 cycles

**Observations**
- Compared to the previous baseline with the same temporal window, training error is slightly reduced.
- Validation RMSE is slightly higher, indicating no improvement in cycle-level accuracy.
- Increasing the depth of the regression head improves fitting capacity but does not improve generalization.
- This suggests that model performance is primarily limited by the temporal representation rather than the final prediction head.

---

## Experiment 4 — Longer Temporal Window (seq_len = 100)

**Configuration**
- Config file: `configs/longer_window.yaml`
- Temporal window: 100 cycles
- Model: CNN + GRU + regression head
- Device: Apple MPS (M1)

**Results** *(window-wise evaluation)*
- Train MSE: ~23.7
- Validation MSE: ~28.7
- Train RMSE: ~39.2 cycles
- Validation RMSE: ~44.4 cycles

**Observations**
- Compared to the previous experiment (seq_len = 75), training error is further reduced.
- Validation RMSE increases, indicating a degradation in cycle-level accuracy.
- Increasing the temporal window beyond 75 cycles improves fitting on the training set but does not improve generalization.
- This suggests that longer sequences introduce additional noise or irrelevant context under the current architecture.

---

## Experiment 5 — Temporal Mean Pooling (seq_len = 75)

**Configuration**
- Config file: `configs/5_temporal_mean_pooling.yaml`
- Temporal window: 75 cycles
- Model: CNN + GRU + temporal mean pooling + regression head
- Device: Apple MPS (M1)

**Results** *(window-wise evaluation)*
- Train MSE: ~26.1
- Validation MSE: ~29.2
- Train RMSE: ~40.9 cycles
- Validation RMSE: ~43.6 cycles

**Observations**
- Compared to the previous experiment with seq_len = 75, training error is similar.
- Validation RMSE is slightly higher, indicating no improvement in cycle-level accuracy.
- Temporal mean pooling smooths the representation but does not improve generalization.
- This suggests that uniformly aggregating temporal information discards useful degradation dynamics captured by the GRU output.

---

## Experiment 6 — Temporal Attention Pooling (seq_len = 75)

**Configuration**
- Config file: `configs/6_attention.yaml`
- Temporal window: 75 cycles
- Model: CNN + GRU + temporal attention pooling + regression head
- Device: Apple MPS (M1)

**Results** *(window-wise evaluation)*
- Train MSE: ~26.3
- Validation MSE: ~29.2
- Train RMSE: ~41.1 cycles
- Validation RMSE: ~43.6 cycles

**Observations**
- Compared to temporal mean pooling, training and validation errors are similar.
- Validation RMSE does not improve compared to the standard GRU output at the same sequence length.
- Temporal attention does not yield a measurable gain in cycle-level accuracy under the current setup.
- This suggests that the attention mechanism, as configured, does not better isolate informative time steps than the GRU’s implicit temporal weighting.

---

## Experiment 7 — Adding Engine Age as Input Feature (seq_len = 75)

**Rationale**
The engine age (cycle index normalized over the sequence) is added as an explicit input feature to provide the model with absolute temporal positioning information.
The objective is to help the model distinguish between similar sensor patterns occurring at different stages of the engine lifecycle.

**Configuration**
- Config file: `configs/7_add_age.yaml`
- Temporal window: 75 cycles
- Additional feature: normalized engine age
- Model: CNN + GRU + regression head
- Device: Apple MPS (M1)

**Results** *(window-wise evaluation)*
- Train MSE: ~26.1
- Validation MSE: ~29.2
- Train RMSE: ~40.8 cycles
- Validation RMSE: ~43.6 cycles

**Observations**
- Compared to the same setup without engine age, training and validation errors are nearly unchanged.
- The explicit age feature does not improve cycle-level accuracy.
- This indicates that the GRU already captures temporal ordering and progression implicitly through the sequence dynamics.
- Adding age information is therefore redundant under the current architecture and target formulation.

---

## Experiment 8 — Adding Global Engine Age as Input Feature (seq_len = 75)

**Rationale**
The engine age is encoded as a global, absolute feature representing the engine’s position within its full lifespan rather than its relative position inside the input sequence.
Unlike Experiment 7, the global age provides information about how far the engine is from the beginning of its operational life, independently of the selected temporal window.

**Configuration**
- Config file: `configs/8_global_age.yaml`
- Temporal window: 75 cycles
- Additional feature: global normalized engine age
- Model: CNN + GRU + regression head
- Device: Apple MPS (M1)

**Results** *(window-wise evaluation)*
- Train MSE: ~24.9
- Validation MSE: ~27.8
- Train RMSE: ~39.4 cycles
- Validation RMSE: ~42.0 cycles

**Observations**
- Compared to the local age feature, training error is slightly reduced.
- Validation RMSE remains in the same range as previous experiments and does not show a clear improvement.
- Providing absolute lifecycle position does not fundamentally change the model’s ability to predict remaining useful life.
- This indicates that, under the current setup, age information—whether local or global—is already implicitly encoded through the temporal evolution of sensor signals.

---

## Experiment 9 — Evaluation on Last Sliding Window Only (seq_len = 75)

**Rationale**
The training procedure remains unchanged, but the evaluation protocol is modified.
Instead of evaluating on all sliding windows of the validation set, the model is evaluated only on the last available window for each engine, corresponding to the most recent state before failure.
This setup better reflects a realistic operational scenario, where predictions are made using the latest observed data.

**Configuration**
- Config file: `configs/9_last_eval.yaml`
- Temporal window: 75 cycles
- Evaluation: last sliding window per engine (validation & test)
- Model: CNN + GRU + regression head
- Device: Apple MPS (M1)

**Results**
- Train RMSE: ~41.8 cycles
- Validation RMSE (last-window): ~67.0 cycles

**Observations**
- Training performance remains consistent with previous experiments.
- Validation RMSE increases sharply when evaluation is restricted to the last window.
- This indicates a mismatch between the training objective (dominated by early/mid-life windows) and the end-of-life focus of last-window evaluation.
- End-of-life regimes are under-represented in the training loss, leading to large errors close to failure.

---

## Experiment 10 — Truncated Last-Window Training with Huber Loss (seq_len = 75)

**Modifications applied**
- Training restricted to windows close to failure (low RUL).
- RUL targets truncated to emphasize end-of-life regimes.
- Evaluation performed on the last sliding window only.
- Loss changed from MSE to Huber to reduce sensitivity to large residuals.

**Configuration**
- Config file: `configs/10_max_train_RUL.yaml`
- Temporal window: 75 cycles
- Model: CNN + GRU + regression head
- Loss: Huber
- Evaluation: last sliding window only
- Device: Apple MPS (M1)

**Results**
- Best validation RMSE (trunc-last): ~6.8 cycles
- Test RMSE (last-window): ~39.9 cycles

**Observations**
- Validation RMSE drops sharply compared to Experiment 9.
- Test performance degrades significantly, revealing strong over-specialization to truncated RUL ranges.
- This confirms that aligning the objective with end-of-life prediction improves local accuracy but can hurt global generalization if the training distribution becomes too narrow.

---

## Experiment 11 — Health Index–Driven RUL Prediction (FD001 Benchmark)

**Objective**
Improve FD001 test RMSE using a paper-comparable evaluation protocol and a physically structured learning objective.

**Configuration**
- Dataset: NASA CMAPSS FD001
- Temporal window: 75 cycles
- RUL cap: 125 cycles (*piecewise RUL*)
- Feature selection: Spearman correlation with piecewise RUL (top features)
- Model:
  - CNN + GRU backbone
  - Attention-based temporal aggregation
  - Dual-head architecture:
    - Main head: RUL regression
    - Auxiliary head: Health Index (HI) prediction
- Loss:
  - Huber loss for RUL
  - Auxiliary HI MSE loss
  - Consistency regularization between RUL and HI
  - Sample-weighted loss emphasizing errors close to failure
- Evaluation protocol (**paper-comparable**):
  - One prediction per engine
  - Last available window per engine
  - RMSE computed on the 100 test engines

**Results**
- Best validation RMSE (trunc-last): **10.79 cycles**
- Test RMSE (FD001 benchmark, last-window): **17.88 cycles**

**Observations**
- HI provides a bounded, monotone proxy of degradation that stabilizes training and improves generalization.
- Piecewise RUL removes uninformative early-life variance, aligning the target with sensor observability.
- Weighting errors close to failure improves performance in the critical end-of-life regime.
- Spearman-based feature selection reduces sensor noise and improves the signal-to-noise ratio for learning.
- The achieved test RMSE (17.88) is consistent with strong FD001 results and materially outperforms direct RUL regression baselines used earlier in the project.

**Conclusion**
This experiment shows that **problem formulation** (targets, losses, evaluation protocol) dominates architectural complexity on FD001.
With a strict benchmark evaluation, the model reaches competitive performance without heavy or opaque architectures.

---
