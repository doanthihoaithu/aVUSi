# Module 1 — Synthetic Data Generator

This module generates synthetic multivariate time series (MTS) with configurable anomaly patterns and dimension-wise ground-truth labels. 
These labels are required to evaluate both accuracy and interpretability in `aVUSi`.

---

## Pipeline Position

```
1_synthetic_data_generator (here)  →  2_anomaly_detector  →  3_metric_calculator
```

Outputs `X`, `L`, and `DL` from this module will be used directly in **2_anomaly_detector**.

---

## Inputs

The module takes a single YAML configuration file that fully specifies the data generation process.
Example configurations used in the aVUSi paper:
> [`settings_six.yaml`](https://github.com/doanthihoaithu/synthsensor/blob/master/Python/generation/config/settings_six.yaml) — configuration used to generate the 974-subset synthetic benchmark dataset.

Further configuration examples can be found in the [synthsensor generation configurations](https://github.com/doanthihoaithu/synthsensor/tree/master/Python/generation/config).

---

## Outputs

| Variable | Shape | Description |
|---|---|---|
| `X` | `(T, d)` | Multivariate time series with T timestamps and d sensor dimensions |
| `L` | `(T,)` | Univariate anomaly labels — 1 if any dimension is anomalous at timestamp t |
| `DL` | `(T, d)` | Dimension-wise anomaly labels — 1 for each dimension that is truly anomalous |

Dimension-wise labels `DL` are essential for computing `IndepNDCG` and `aVUSi`, as they define the ground truth for interpretability evaluation.

---

## Implementation

This module extends **synthsensor** — a synthetic sensor time series generator originally designed for two-dimensional signals — to support the higher-dimensional MTS needed for the aVUSi benchmark. The key extensions are:

- **Higher dimensionality** — scales from 2 to up to `d` sensor dimensions per time series.
- **Dimension-wise label export** — records which specific dimensions are affected by each injected anomaly, producing the `DL` matrix required by `IndepNDCG` and `aVUSi`.
- **Correlated anomaly injection** — injects correlated sequence anomalies into configurable subsets of dimensions, directly producing the ground-truth labels `DL`.

### Repositories

|                                    | Link |
|------------------------------------|---|
| **Extended version** (this module) | [github.com/doanthihoaithu/synthsensor](https://github.com/doanthihoaithu/synthsensor) |
| **Original repository**            | [github.com/AstridMarie2/synthsensor](https://github.com/AstridMarie2/synthsensor) |
| **Original Interactive demo app**  | [sensordiagnostics.shinyapps.io/app_synth](https://sensordiagnostics.shinyapps.io/app_synth/) |

### Reference to the original `synthsensor` package

```bibtex
@software{skalvik_synthsensor_2025,
  author    = {Skålvik, Astrid Marie},
  title     = {synthsensor: Synthetic Two-Sensor Time Series with Labeled Anomalies},
  month     = sep,
  year      = 2025,
  publisher = {Zenodo},
  version   = {v0.1.0},
  doi       = {10.5281/zenodo.17157666},
  url       = {https://doi.org/10.5281/zenodo.17157666},
}
```

---

## Dataset

The synthetic dataset used in the aVUSi paper was generated with this module using configuration [settings_six.yaml](https://github.com/doanthihoaithu/synthsensor/blob/master/Python/generation/config/settings_six.yaml):

| Domain | # Dim. | # Subsets | Avg. Length | Avg. Anomaly Ratio |
|---|---|---|---|---|
| Sensor | 10 | 974 | 6,536 | 7.23% |

 We generate one anomaly-free MTS for training semi-supervised detectors, and use 973 additional MTS to evaluate both semi-supervised and unsupervised detectors within the aVUSi benchmark. Each time series has 10 dimensions and contains diverse anomaly types, durations, and intensities. For evaluation we provide both univariate labels (`L`) and dimension-wise labels (`DL`).
