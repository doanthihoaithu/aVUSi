# results/

Output directory for Module 2. Populated automatically by `runner.py` after running each detector on each batch.

## Expected Structure

```
results/
└── <dataset_name>/                  # matches mts_running_dataset in conf/config.yaml
    └── merged_results/
        └── <detector>/              # e.g. hbos, tran_ad, cblof, ...
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

## How to Generate

```bash
# 1. Prepare input data (if not already done)
#    See 2_anomaly_detector/data/README.md

# 2. Copy and edit the detector config
cp 2_anomaly_detector/conf/config.yaml.example 2_anomaly_detector/conf/config.yaml
#    Set mts_running_dataset and mts_running_detector in config.yaml

# 3. Run the detector
python 2_anomaly_detector/runner.py
```

Results are written incrementally — one batch folder at a time — so partial runs produce valid output for completed batches.