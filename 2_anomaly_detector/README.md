# Module 2 — Anomaly Detector

This module runs interpretable anomaly detectors on multivariate time series and produces two outputs required by the metric calculator: an anomaly score sequence `S` and a Dimension Contribution Matrix `DCM`.

---

## Pipeline Position

```
1_synthetic_data_generator  →  2_anomaly_detector (here)  →  3_metric_calculator
```

Input `X` comes from **Module 1** or any external dataset. Outputs `S` and `DCM` are consumed by **Module 3**.

---

## Inputs

| Variable | Shape | Description |
|---|---|---|
| `X` | `(T, d)` | Multivariate time series with T timestamps and d dimensions |
| `L` | `(T,)` | Univariate (point-wise) anomaly labels |
| `DL` | `(T, d)` | Dimension-wise anomaly labels |

---

## Outputs

| Variable | Shape | Description |
|---|---|---|
| `S` | `(T,)` | Anomaly score sequence — higher values indicate stronger anomaly signal |
| `DCM` | `(T, d)` | Dimension Contribution Matrix — normalized distribution over dimensions at each timestamp |

The DCM encodes, for each timestamp, how much each dimension contributes to the anomaly score. It is derived from per-dimension errors, density scores, or distance decompositions, then normalized via softmax so that each row sums to 1.

---

## Supported Detectors

Ten interpretable detectors are supported, grouped by category:

| Category | Detector | Strategy | DCM Source |
|---|---|---|---|
| Classical ML | HBOS | Distribution-based (histogram density) | Per-dimension histogram density scores |
| Classical ML | RBF | Forecasting-based (random forest ensemble) | Per-dimension prediction errors |
| Outlier Detection | CBLOF | Clustering-based (cluster size and distance) | Per-dimension distance to cluster centroid |
| Outlier Detection | COPOD | Distribution-based (copula tail probabilities) | Per-dimension tail probability scores |
| Deep Learning | AE | Reconstruction-based (autoencoder) | Per-dimension reconstruction errors |
| Deep Learning | DAE | Reconstruction-based (denoising autoencoder) | Per-dimension reconstruction errors |
| Deep Learning | EncDec-AD | Reconstruction-based (LSTM encoder-decoder) | Per-dimension reconstruction errors |
| Deep Learning | TranAD | Forecasting-based (transformer + adversarial) | Per-dimension prediction errors |
| Deep Learning | OmniAnomaly | Reconstruction-based (stochastic RNN + normalizing flows) | Per-dimension reconstruction errors |
| Deep Learning | MTAD-GAT | Forecasting + reconstruction (graph attention network) | Per-dimension forecasting + reconstruction errors |

An **Average Ensemble (AvgEns)** that averages `S` and `DCM` across all detectors is also supported and consistently achieves the best aVUSi score on both datasets.

---

## DCM Derivation

The DCM is computed differently per detector family:

- **Reconstruction-based** (AE, DAE, EncDec-AD, OmniAnomaly, MTAD-GAT): per-dimension squared reconstruction errors, softmax-normalized.
- **Forecasting-based** (RBF, TranAD, MTAD-GAT): per-dimension squared prediction errors, softmax-normalized.
- **Distribution-based** (HBOS, COPOD): per-dimension density or tail-probability scores, softmax-normalized.
- **Clustering-based** (CBLOF): per-dimension distance to the assigned cluster centroid, softmax-normalized.

Formally, for a raw per-dimension score vector $e_t \in \mathbb{R}^d$ at timestamp $t$:

$$\text{DCM}(t) = \text{softmax}(e_t) = \frac{\exp(e_t)}{\sum_{j=1}^{d} \exp(e_{t,j})}$$

---

## Results Summary

### Synthetic Dataset

| Detector | VUS-PR | IndepNDCG | aVUSi |
|---|---|---|---|
| AvgEns | — | — | **Best** |
| TranAD | **Best** | — | 3rd |
| RBF | — | **Best** | 2nd |

### SMD Dataset

| Detector | VUS-PR | IndepNDCG | aVUSi |
|---|---|---|---|
| AvgEns | **Best** | 2nd | **Best** |
| CBLOF | 2nd | — | — |
| COPOD | — | **Best** | 2nd |

Key finding: **AvgEns consistently ranks first under aVUSi**, demonstrating that ensembling effectively balances accuracy and interpretability even when individual detectors trade one off against the other (e.g., DAE: VUS-PR = 0.85 but IndepNDCG = 0.00; HBOS: VUS-PR = 0.42 but IndepNDCG = 0.98).