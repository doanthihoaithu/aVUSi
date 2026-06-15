# aVUSi: Joining Accuracy and Interpretability Evaluation Measures for Time Series Anomaly Detection

> **Interpretable or Accurate? or Both? Joining Accuracy and Interpretability Evaluation Measures for Time Series Anomaly Detection**  
> *Thu T. H. Doan, Emmanouil Sylligardos, Astrid Marie Skålvik, Rogardt Heldal, Themis Palpanas, Patrizio Pelliccione, Paul Boniol*  
> Gran Sasso Science Institute · ENS, PSL, CNRS, Inria · University of Bergen, NORCE · Western Norway University of Applied Sciences · Université Paris Cité

---

## Overview

**aVUSi** is a novel evaluation metric for Multivariate Time Series Anomaly Detection (MTS-AD) that jointly captures **detection accuracy** and **interpretability** in a single quantitative measure. Unlike existing metrics that evaluate the two aspects independently, aVUSi integrates interpretability scores directly into the anomaly score sequence before computing detection performance, penalizing detectors that fail to identify the dimensions responsible for detected anomalies even when their detection accuracy is high.

aVUSi builds upon **VUS-PR** — a threshold-independent, range-based accuracy metric published at VLDB — and extends it by incorporating dimension-wise interpretability evaluation via NDCG@k.

---

## Motivation

![Motivation](figures/avusi-motivation.png)

Existing anomaly detection benchmarks primarily focus on detection accuracy and treat interpretability as a separate post-hoc analysis. This disconnect makes it difficult to systematically compare detectors that trade one aspect against the other. For example:

- A detector may achieve strong detection accuracy while **completely failing** to identify which dimensions caused the anomaly (e.g., DAE: VUS-PR = 0.85, IndepNDCG = 0.00).
- A detector may provide highly accurate explanations but only moderate detection accuracy (e.g., HBOS: VUS-PR = 0.42, IndepNDCG = 0.98).
- Two detectors may achieve identical accuracy and interpretability scores independently, yet behave very differently in practice (e.g., TranAD vs. RBF: both VUS-PR = 0.83, both IndepNDCG ≈ 0.99, yet aVUSi = 0.88 vs. 0.93).

aVUSi addresses this gap by providing a **unified, threshold-independent** measure that rewards detectors that are both accurate and interpretable.

---

## Illustration of aVUSi

![Algorithm](figures/avusi-alg.png)

aVUSi proceeds in five steps (see Algorithm 1 and Figure 1 in the paper). More details on each step are provided in module `3_metric_calculator/README.md.`

### Hyperparameters

| Parameter | Description | Default | Recommendation |
|---|---|---|---|
| `k` | NDCG ranking cutoff | 5 | Select based on domain knowledge; start with k ≈ d/3 |
| `w` | Smoothing window size | 10 | w ∈ [5, 10] provides stable results |
| `M` | Number of sensitivity levels | 50 | M = 20 is sufficient; M = 50 for smoother curve |

**Note:** aVUSi is most sensitive to `k`. Smaller `k` provides stricter interpretability evaluation; larger `k` rewards broader coverage of anomalous dimensions.


### Time Complexity

| Step | Operation | Complexity |
|---|---|---|
| Step 1 | Smoothing DCM | O(T · w · d) |
| Step 2 | Computing I via NDCG@k | O(\|T_A\| · d log d) |
| Steps 3–5 | Penalizing and aggregating | O(η · τ · M) |
| **Total** | | **O(T · w · d + \|T_A\| · d log d + η · τ · M)** |

In practice, aVUSi is approximately 20× slower than VUS-PR and 2× slower than IndepNDCG. Since evaluation is performed **offline** after detector inference, this overhead does not affect detector runtime.

---

[//]: # (### Step 1 — Smooth the Dimension Contribution Matrix &#40;DCM&#41;)

[//]: # (The **DCM** encodes, for each timestamp t, a normalized distribution over dimensions representing how much each dimension contributes to the anomaly score. To handle fluctuating contributions in range-based anomalies, score-weighted averaging is applied over a smoothing window of size `w`:)

[//]: # ()
[//]: # ($$\tilde{c}_t = \frac{\sum_{j \in \mathcal{W}_{t,w}} s_j \cdot c_j}{\sum_{j \in \mathcal{W}_{t,w}} s_j}$$)

[//]: # ()
[//]: # (### Step 2 — Form the Interpretability Score Sequence)

[//]: # (Interpretability is formulated as a **ranking problem**: the smoothed DCM induces a ranking of dimensions by decreasing contribution, and a good detector should place truly anomalous dimensions at the top. Interpretability at each anomalous timestamp t is evaluated using **NDCG@k**:)

[//]: # ()
[//]: # ($$I&#40;t&#41; = \text{NDCG@}k&#40;t&#41; \in [0, 1]$$)

[//]: # ()
[//]: # (- **Fully Interpretable** &#40;I&#40;t&#41; = 1&#41;: all anomalous dimensions occupy the top-k positions exclusively.)

[//]: # (- **Partially Interpretable** &#40;0 < I&#40;t&#41; < 1&#41;: anomalous dimensions appear only partially within top-k, or are outranked by normal dimensions.)

[//]: # (- **Totally Missed** &#40;I&#40;t&#41; = 0&#41;: no anomalous dimension appears within top-k.)

[//]: # ()
[//]: # (### Step 3 — Penalize Anomaly Scores)

[//]: # (For each sensitivity level m ∈ [0, 1], construct the penalized anomaly score sequence:)

[//]: # ()
[//]: # ($$PS^m = S \odot S^m_{\text{Interp}}, \quad \text{where} \quad S^m_{\text{Interp}}&#40;t&#41; = \begin{cases} I&#40;t&#41;, & L_t = 1 \\ m, & \text{otherwise} \end{cases}$$)

[//]: # ()
[//]: # (### Step 4 — Compute $VUSi^m$)

[//]: # (For each sensitivity level m, compute VUS-PR on the penalized scores:)

[//]: # ()
[//]: # ($$\text{VUSi}^m = \text{VUS-PR}&#40;PS^m&#41;$$)

[//]: # ()
[//]: # (### Step 5 — Aggregate to obtain aVUSi)

[//]: # (aVUSi is the area under the VUSi&#40;m&#41; curve as m sweeps from 0 to 1:)

[//]: # ()
[//]: # ($$\text{aVUSi}&#40;X&#41; = \int_0^1 \text{VUSi}^m dm \approx \frac{1}{M} \sum_{i=0}^{M-1} \text{VUSi}^{m_i}$$)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Theoretical Properties)

[//]: # ()
[//]: # (| Property | Statement |)

[//]: # (|---|---|)

[//]: # (| **Boundedness** | aVUSi&#40;X&#41; ∈ [0, 1] for any MTS X |)

[//]: # (| **Monotonicity** | If interpretability improves pointwise, aVUSi does not decrease |)

[//]: # (| **Consistency** | When all anomalous timestamps are fully interpretable, aVUSi ≥ VUS-PR&#40;S&#41; |)

[//]: # ()
[//]: # (---)

[//]: # (---)

[//]: # ()
[//]: # (## Detectors)

[//]: # ()
[//]: # (The following 10 interpretable detectors are evaluated in the paper, grouped by category:)

[//]: # ()
[//]: # (| Category | Detector | Strategy |)

[//]: # (|---|---|---|)

[//]: # (| Classical ML | HBOS | Distribution-based &#40;histogram density&#41; |)

[//]: # (| Classical ML | RBF | Forecasting-based &#40;random forest ensemble&#41; |)

[//]: # (| Outlier Detection | CBLOF | Clustering-based &#40;cluster size and distance&#41; |)

[//]: # (| Outlier Detection | COPOD | Distribution-based &#40;copula tail probabilities&#41; |)

[//]: # (| Deep Learning | AE | Reconstruction-based &#40;autoencoder&#41; |)

[//]: # (| Deep Learning | DAE | Reconstruction-based &#40;denoising autoencoder&#41; |)

[//]: # (| Deep Learning | EncDec-AD | Reconstruction-based &#40;LSTM encoder-decoder&#41; |)

[//]: # (| Deep Learning | TranAD | Forecasting-based &#40;transformer + adversarial&#41; |)

[//]: # (| Deep Learning | OmniAnomaly | Reconstruction-based &#40;stochastic RNN + normalizing flows&#41; |)

[//]: # (| Deep Learning | MTAD-GAT | Forecasting + reconstruction &#40;graph attention network&#41; |)

[//]: # ()
[//]: # (For each detector, the DCM is derived from per-dimension prediction errors, reconstruction errors, density scores, or distance decompositions, normalized via softmax at each timestamp.)

[//]: # ()
[//]: # (---)

## Summary of Experimental Evaluation
### Datasets

| Dataset | Domain | # Dim. | # Subsets | Avg. Length | Avg. Anomaly Ratio |
|---|---|---|---|---|---|
| Synthetic | Sensor | 10 | 974 | 6,536 | 7.23% |
| SMD | Server Machine | 38 | 28 | 25,300 | 4.2% |

Both datasets provide **dimension-wise ground-truth labels**, which are required for quantitative interpretability evaluation. The synthetic dataset is generated by adopting an extended version of [synthsensor (Demo Web)](https://sensordiagnostics.shinyapps.io/app_synth/) - [GitHub](https://github.com/AstridMarie2/synthsensor.git) to support higher-dimensional MTS.

---

### Overal Results

#### Synthetic Dataset (top detectors by each measure)

| Detector | VUS-PR | IndepNDCG | aVUSi |
|---|---|---|---|
| AvgEns | — | — | **Best** |
| TranAD | **Best** | — | 3rd |
| RBF | — | **Best** | 2nd |

#### SMD Dataset (top detectors by each measure)

| Detector | VUS-PR | IndepNDCG | aVUSi |
|---|---|---|---|
| AvgEns | **Best** | 2nd | **Best** |
| CBLOF | 2nd | — | — |
| COPOD | — | **Best** | 2nd |

Key finding: **AvgEns (Average Ensemble) consistently ranks first under aVUSi** on both datasets, demonstrating that ensembling effectively balances accuracy and interpretability.

---

[//]: # (## Citation)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@inproceedings{doan2026avusi,)

[//]: # (  title     = {Interpretable or Accurate? or Both? Joining Accuracy and )

[//]: # (               Interpretability Evaluation Measures for Time Series Anomaly Detection},)

[//]: # (  author    = {Doan, Thu T. H. and Sylligardos, Emmanouil and Sk{\aa}lvik, Astrid Marie )

[//]: # (               and Heldal, Rogardt and Palpanas, Themis and Pelliccione, Patrizio )

[//]: # (               and Boniol, Paul},)

[//]: # (  booktitle = {IEEE International Conference on Data Engineering &#40;ICDE&#41;},)

[//]: # (  year      = {2026})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (---)

## Related Work

This work builds directly on:

- **VUS-PR** — Boniol et al., *VUS: Effective and Efficient Accuracy Measures for Time-Series Anomaly Detection*, VLDB Journal 2025.
- **synthsensor** — Skålvik, *Synthsensor: Synthetic Two-Sensor Time Series with Labeled Anomalies*, v0.1.0, 2025.

---

## Installation

```bash
git clone https://github.com/doanthihoaithu/aVUSi.git
cd aVUSi
pip install -r requirements.txt
```

**Requirements:**
- Python 3.11
- PyTorch 2.1
- TensorFlow 2.15
- CUDA 11.8 (optional, for GPU acceleration)

---

## Repository Structure

```
aVUSi/
├── 1_synthetic_data_generator/   # Module 1: generate synthetic MTS with dimension-wise anomaly labels
│   └── README.md
├── 2_anomaly_detector/           # Module 2: run interpretable detectors, produce S and DCM
│   └── README.md
├── 3_metric_calculator/          # Module 3: compute VUS-PR, IndepNDCG, and aVUSi
│   └── README.md
├── conf/                         # Configuration files
│   ├── config.yaml
│   └── config.yaml.example
├── data/                         # Input datasets (Synthetic, SMD)
├── figures/                      # Paper figures
│   ├── avusi-motivation.png      # Figure 1: motivation for aVUSi
│   └── avusi-alg.png             # Figure 2: aVUSi algorithm illustration
├── notebooks/                    # Jupyter notebooks for analysis and visualization
├── output/                       # Detector outputs and evaluation results
├── requirements.txt
└── README.md
```

The three numbered modules form a sequential pipeline:

```
1_synthetic_data_generator  →  2_anomaly_detector  →  3_metric_calculator
       (X, L, DL)               (S, DCM)              (VUS-PR, IndepNDCG, aVUSi)
```

---

## Usage

Run `main.py` to execute the full pipeline on synthetic demo data and compute all three metrics. The script is organized into three sections corresponding to the three modules.

### 1. Generate demo data

Generates a synthetic multivariate time series with three anomalous windows and two normal windows that contain elevated anomaly scores, then plots the per-dimension values, univariate anomaly score, and ground-truth labels.

**Output**: MTS `X` of shape `(T, d)`, univariate labels `L` of shape `(T,)`, dimension-wise labels `DL` of shape `(T, d)`, anomaly scores `S` of shape `(T,)`, and dimension contribution matrix `DCM` of shape `(T, d)`.

<table><tr><td>

```python
from main import generate_demo_data, plot_demo_data

demo = generate_demo_data(
    T=1000,
    d=5,
    anomalous_areas=((100, 150), (400, 470), (700, 760)),
    normal_areas=((250, 290), (580, 620)),
    seed=42,
)

# demo keys:
#   DL         (T, d) — dimension-wise labels
#   L          (T,)   — univariate labels
#   S          (T,)   — anomaly score sequence
#   DCM        (T, d) — dimension contribution matrix
#   X          (T, d) — multivariate time series

plot_demo_data(demo, out_dir="figures/usage_in_readme")
```

</td><td>

![Demo data](figures/usage_in_readme/demo_data.png)

</td></tr></table>

### 2. Compute metrics

Computes three metrics on the demo data: `VUS-PR` for independent accuracy, `IndepNDCG` for independent interpretability, and `aVUSi` for the combined evaluation. Default hyperparameters are `k = d/2`, `w = 5`, and `M = 50`.

> The `IndepNDCG` and `aVUSi` APIs support sweeping over hyperparameter grids. Results are returned as dictionaries keyed by `(k, w)` or `(k, w, M)`, where each entry contains a `value` field for the metric score and a `computation_time` field. The default hyperparameter combination is selected for display here.

**Output**: VUS-PR, IndepNDCG, and aVUSi scores, each in `[0, 1]`.

```python
import numpy as np
from metrics.ffvus.ffvus_metrics import FFVUS
from metrics.interpretability_metrics import IndependentNDCG, AVUSI

S   = demo["S"].astype(np.float64)
L   = demo["L"].astype(np.float64)
DL  = demo["DL"].astype(np.float64)
DCM = demo["DCM"].astype(np.float64)

k = demo["X"].shape[1] // 2   # NDCG cutoff ≈ d/2
w = 5                                   # smoothing window
M = 50                                  # sensitivity levels

# VUS-PR — detection accuracy
vus_pr_result = FFVUS(slope=10).score(L, S)
print(f"VUS-PR    = {vus_pr_result['value']:.4f}")

# IndepNDCG — interpretability (independent of accuracy)
indep_ndcg_metric = IndependentNDCG(
    max_k=k, max_smoothing_window=w,
    k_slide=1, smoothing_window_slide=1,
    fix_k=True, fix_smoothing_window=True,
    default_selected_k=k, default_selected_smoothing_window=w,
)
indep_ndcg_results = indep_ndcg_metric.score_for_different_k(
    y_score=S, y_true_multivariate=DL, dimension_contribution=DCM,
)
print(f"IndepNDCG = {indep_ndcg_results[(k, w)]['value']:.4f}")

# aVUSi — combined accuracy + interpretability
avusi_metric = AVUSI(
    max_k=k, max_smoothing_window=w, max_n_interpretability_sensitivity_levels=M,
    k_slide=1, smoothing_window_slide=1, n_interpretability_sensitivity_levels_slide=1,
    fix_k=True, fix_smoothing_window=True, fix_n_interpretability_sensitivity_levels=True,
    default_selected_k=k, default_selected_smoothing_window=w,
    default_selected_n_interpretability_sensitivity_levels=M,
    vus_slope=5, vus_n_thresholds=M,
)
avusi_results = avusi_metric.score_for_different_k(
    y_true_univariate=L, y_score_univariate=S,
    y_true_multivariate=DL, dimension_contribution_multivariate=DCM,
)
print(f"aVUSi     = {avusi_results[(k, w, M)]['value']:.4f}")
```

### 3. Plot VUSi curve

After computing `aVUSi`, plots the `VUSi(m)` curve to visualize how interpretability penalization affects detection performance as the sensitivity level `m` varies from 0 to 1. The area under this curve equals the `aVUSi` score.

<table><tr><td>

```python
from main import plot_vusi_curve

plot_vusi_curve(
    avusi_results,
    k=k,
    w=w,
    M=M,
    vus_pr=vus_pr_result["value"],
    indep_ndcg=indep_ndcg_results[(k, w)]["value"],
    out_dir="figures/usage_in_readme",
)
```

The plot shows:
- **VUSi(m) curve** — the penalized VUS-PR score at each sensitivity level m.
- **Shaded area** — the area under the VUSi curve, equal to aVUSi, annotated with its value.
- **VUS-PR line** — a dashed red horizontal reference line with an inline label above it.
- **IndepNDCG line** — a dashed green horizontal reference line with an inline label above it.

</td><td>

![VUSi curve](figures/usage_in_readme/vusi_curve.png)

</td></tr></table>

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact

- Thu T. H. Doan — thihoaithu.doan@gssi.it