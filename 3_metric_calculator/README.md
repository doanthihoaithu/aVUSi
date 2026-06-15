# Module 3 — Metric Calculator

This module computes evaluation metrics for Multivariate Time Series Anomaly Detection (MTS-AD), combining detection accuracy and interpretability into a single unified score.

---

## Pipeline Position

```
1_synthetic_data_generator  →  2_anomaly_detector  →  3_metric_calculator (here)
```

Inputs `S` and `DCM` are produced by **Module 2**. Labels `L` and `DL` are produced by **Module 1** or provided as ground truth.

---

## Inputs

| Variable | Shape | Description |
|---|---|---|
| `X` | `(T, d)` | Multivariate time series |
| `L` | `(T,)` | Univariate (point-wise) anomaly labels |
| `DL` | `(T, d)` | Dimension-wise anomaly labels |
| `S` | `(T,)` | Anomaly score sequence (from Module 2) |
| `DCM` | `(T, d)` | Dimension Contribution Matrix (from Module 2) |

---

## Outputs

| Metric | Range | Measures |
|---|---|---|
| **VUS-PR** | [0, 1] | Detection accuracy (threshold-independent, range-based) |
| **IndepNDCG** | [0, 1] | Interpretability (independent of detection accuracy) |
| **aVUSi** | [0, 1] | Combined accuracy + interpretability |

---

## How aVUSi is Computed

aVUSi proceeds in five steps:

**Step 1 — Smooth the DCM.**  
Score-weighted averaging over a window of size `w` stabilizes noisy dimension contributions across range-based anomalies:

$$\tilde{c}_t = \frac{\sum_{j \in \mathcal{W}_{t,w}} s_j \cdot c_j}{\sum_{j \in \mathcal{W}_{t,w}} s_j}$$

**Step 2 — Compute Interpretability Scores.**  
At each anomalous timestamp, NDCG@k evaluates how well the smoothed DCM ranks truly anomalous dimensions at the top:

$$I(t) = \text{NDCG@}k(t) \in [0, 1]$$

**Step 3 — Penalize Anomaly Scores.**  
For each sensitivity level $m \in [0, 1]$, element-wise penalization is applied:

$$PS^m = S \odot S^m_{\text{Interp}}, \quad S^m_{\text{Interp}}(t) = \begin{cases} I(t), & L_t = 1 \\ m, & \text{otherwise} \end{cases}$$

**Step 4 — Compute $\text{VUSi}^m$.**  
VUS-PR is evaluated on the penalized scores at each sensitivity level:

$$\text{VUSi}^m = \text{VUS-PR}(PS^m)$$

**Step 5 — Aggregate to aVUSi.**  
aVUSi is the area under the $\text{VUSi}(m)$ curve:

$$\text{aVUSi}(X) = \int_0^1 \text{VUSi}^m \, dm \approx \frac{1}{M} \sum_{i=0}^{M-1} \text{VUSi}^{m_i}$$

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `k` | 5 | NDCG ranking cutoff. Stricter with smaller values; start with k ≈ d/3 |
| `w` | 10 | Smoothing window size. Values in [5, 10] give stable results |
| `M` | 50 | Number of sensitivity levels. M = 20 is sufficient; M = 50 for a smoother curve |

---

## Theoretical Properties

| Property | Statement |
|---|---|
| **Boundedness** | aVUSi(X) ∈ [0, 1] for any MTS X |
| **Monotonicity** | Pointwise improvement in interpretability does not decrease aVUSi |
| **Consistency** | When all anomalous timestamps are fully interpretable, aVUSi ≥ VUS-PR(S) |

---

## Usage

A full end-to-end example — generating demo data, running all three metrics, and plotting the VUSi curve — is provided in the root [`README.md § Usage`](../README.md#usage) and the accompanying [`main.py`](../main.py).

The core API calls for this module are:

```python
from metrics.ffvus.ffvus_metrics import FFVUS
from metrics.interpretability_metrics import IndependentNDCG, AVUSI

# VUS-PR
vus_pr_result = FFVUS(slope=10).score(L, S)

# IndepNDCG
indep_ndcg_results = IndependentNDCG(...).score_for_different_k(
    y_score=S, y_true_multivariate=DL, dimension_contribution=DCM
)

# aVUSi
avusi_results = AVUSI(...).score_for_different_k(
    y_true_univariate=L, y_score_univariate=S,
    y_true_multivariate=DL, dimension_contribution_multivariate=DCM,
)
```

See [`main.py`](../main.py) for the full parameter setup and [`README.md § Usage`](../README.md#usage) for the expected outputs and figures.
