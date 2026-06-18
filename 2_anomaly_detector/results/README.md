# results/

Output directory for Module 2. Populated by `runner.py` (individual detectors) and `runner_for_avg_ens.py` (Average Ensemble).

## Expected Structure

```
results/
└── <dataset_name>/                  # matches mts_running_dataset in conf/config.yaml
    └── merged_results/
        └── <detector>/              # e.g. hbos, tran_ad, cblof, avg_ens, ...
            ├── results.csv          # summary row per batch (algorithm, collection, dataset)
            └── <batch>.csv/         # one folder per test batch, named after the batch file
                ├── anomaly-scores.csv                          # S   — final anomaly score sequence (T,)
                ├── docker-algorithm-dimension-contribution.csv # DCM — softmax-normalized dimension contributions (T, d)
                ├── docker-algorithm-scores.csv                 # raw detector score before post-processing (T,)
                ├── docker-algorithm-scores-per-var.csv         # raw per-dimension scores before softmax (T, d)
                ├── docker-algorithm-multivariate-labels.csv    # predicted dimension-wise binary labels (T, d)
                └── docker-algorithm-execute-time.csv           # wall-clock inference time in seconds
```

## File Descriptions

| File | Variable | Shape | Description |
|---|---|---|---|
| `anomaly-scores.csv` | `S` | `(T,)` | Final anomaly score sequence — passed to Module 3 as-is |
| `docker-algorithm-dimension-contribution.csv` | `DCM` | `(T, d)` | Softmax-normalized dimension contributions — passed to Module 3 as-is |
| `docker-algorithm-scores.csv` | — | `(T,)` | Raw detector score before sliding-window post-processing |
| `docker-algorithm-scores-per-var.csv` | — | `(T, d)` | Raw per-dimension scores before softmax normalization |
| `docker-algorithm-multivariate-labels.csv` | — | `(T, d)` | Predicted dimension-wise binary labels |
| `docker-algorithm-execute-time.csv` | — | scalar | Inference wall-clock time in seconds |

The two files consumed by **Module 3** are `anomaly-scores.csv` (`S`) and `docker-algorithm-dimension-contribution.csv` (`DCM`).

## Pre-computed Results (Paper)

Ready-to-use results for the two datasets evaluated in the paper are available on Google Drive:

**[Download from Google Drive](https://drive.google.com/drive/folders/1PNxYpqO3lYkuagxe7IxynApdTjJLvudw?usp=sharing)**

| Dataset | Description |
|---|---|
| `settings_six` | Synthetic MTS benchmark — generated with [`settings_six.yaml`](https://github.com/doanthihoaithu/synthsensor/blob/master/Python/generation/config/settings_six.yaml) (10 sensors, 974 batches) |
| `SMD` | Server Machine Dataset — real-world MTS benchmark |

Each dataset folder contains `anomaly-scores.csv` (`S`) and `docker-algorithm-dimension-contribution.csv` (`DCM`) for all 10 individual detectors (HBOS, RBF, CBLOF, COPOD, AE, DAE, EncDec-AD, TranAD, OmniAnomaly, MTAD-GAT) and the Average Ensemble (`avg_ens`).

Download the corresponding folder and place its contents under `results/<dataset_name>/`.

## How to Generate

```bash
# 1. Prepare input data (if not already done)
#    See 2_anomaly_detector/data/README.md

# 2. Copy and edit the detector config
cp 2_anomaly_detector/conf/config.yaml.example 2_anomaly_detector/conf/config.yaml
#    Set mts_running_dataset and mts_running_detector in config.yaml

# 3. Run an individual detector (repeat for each detector)
python 2_anomaly_detector/runner.py

# 4. (Optional) Compute the Average Ensemble after all individual detectors are done
#    Edit running_detectors in runner_for_avg_ens.py to list available detectors, then:
python 2_anomaly_detector/runner_for_avg_ens.py
```

Individual detector results are written incrementally — one batch folder at a time — so partial runs produce valid output for completed batches. The `avg_ens` folder is populated in a single pass over all batches once the ensemble is computed.