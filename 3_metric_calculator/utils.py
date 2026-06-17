import json
import os
from pathlib import Path
from typing import Any

import joblib
from numpy import ndarray, dtype
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from omegaconf import DictConfig
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from metrics.ffvus.ffvus_metrics import FFVUS


def estimate_dimension_contribution_by_anomaly_score_with_a_buffer(
    y_score: np.ndarray,
    dimension_contribution: np.ndarray,
    smoothing_window: int
) -> np.ndarray:
    """
    Estimate the contribution of each dimension with a buffer.

    Args:
        y_score: 1D array of shape (T,) with anomaly scores.
        dimension_contribution: 2D array of shape (T, D) with contribution
            scores for each dimension at each timestamp.
        smoothing_window: number of timestamps to consider before and after
            each timestamp.

    Returns:
        2D array of shape (T, D) with smoothed contribution scores,
        weighted by anomaly scores via trapezoidal integration.
    """
    if smoothing_window == 0:
        return dimension_contribution

    T, D = dimension_contribution.shape
    w = smoothing_window
    window_len = 2 * w + 1
    # print('y_score.shape', y_score.shape)

    y_score =y_score.reshape(-1)

    # Pad along the time axis so every timestamp has a full window
    y_padded  = np.pad(y_score, (w,w), mode="edge")          # (T + 2w,)
    # print('y_padded.shape', y_padded.shape)
    dc_padded = np.pad(dimension_contribution,
                       ((w, w), (0, 0)), mode="edge")          # (T + 2w, D)

    # sliding_window_view on 1D score — shape (T, window_len)
    y_windows = sliding_window_view(y_padded, window_len,axis=0)      # (T, window_len)
    # print('dc_padded.shape', dc_padded.shape)

    # sliding_window_view on 2D array — apply along axis=0 only
    # result shape: (T, D, window_len) — note D and window_len are swapped
    dc_windows = sliding_window_view(
        dc_padded, window_len, axis=0
    )                                                          # (T, D, window_len)

    # Align axes: bring window axis next to time → (T, window_len, D)
    dc_windows = dc_windows.transpose(0, 2, 1)                # (T, window_len, D)

    # Weighted integrand and trapezoidal integration along window axis
    integrand = y_windows[:, :, np.newaxis] * dc_windows      # (T, window_len, D)
    estimated = np.trapz(integrand, axis=1)                   # (T, D)

    return estimated

def transform_to_dimension_contribution(scores_per_var):
    # scores_per_var is a 2D array of shape (n_samples, n_features)
    # We want to transform it to dimension contribution such that higher scores indicate higher contribution to anomaly
    # One simple way is to normalize the scores_per_var for each sample and then take the negative log to get contribution
    # Adding a small epsilon to avoid log(0)
    epsilon = 1e-10
    normalized_scores = scores_per_var / (np.sum(scores_per_var, axis=1, keepdims=True) + epsilon)
    dimension_contribution = normalized_scores
    return dimension_contribution

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def load_data_from_json(config: DictConfig, test_filenames_in_model_selection=None):
    project_root = get_project_root()
    data_folder = config.data_folder
    absolute_data_folder = os.path.join(project_root, data_folder, 'semisupervised')
    dataset_json = os.path.join(absolute_data_folder, 'datasets_merged.json')

    # {
    #     "synthetic_batch_0.csv": {
    #         "train_path": "./synthetic_training.csv",
    #         "test_paths": ["./synthetic_batch_0.csv", ... ],
    #         "type": "synthetic test"
    #     },
    # }

    datasets = json.load(open(dataset_json, 'r'))
    dataset = datasets[list(datasets.keys())[0]]
    train_paths = dataset['train_paths']
    test_paths = dataset['test_paths']
    if test_filenames_in_model_selection is not None:
        test_paths = [f for f in test_paths if os.path.basename(f) in test_filenames_in_model_selection]

    train_dfs = []
    for train_path in train_paths:
        train_df = pd.read_csv(os.path.join(absolute_data_folder, train_path[2:]))
        train_dfs.append(train_df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    # train_df = pd.read_csv(os.path.join(absolute_data_folder, train_path[2:]))
    # data = MinMaxScaler().fit_transform(df.iloc[:,1:-1].values)
    scaler = MinMaxScaler()
    train_data = train_df.iloc[:, 1:-1].values
    train_data = scaler.fit_transform(train_data)
    save_scaler_folder = os.path.join(project_root, config.scaler_folder)
    os.makedirs(save_scaler_folder, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_scaler_folder, 'scaler.gz'))
    print(f'Saved scaler to {os.path.join(save_scaler_folder, "scaler.gz")}')
    train_labels = train_df.iloc[:, -1].values

    test_dfs = [pd.read_csv(os.path.join(absolute_data_folder, tp[2:])) for tp in test_paths]
    test_multivariate_labels_dfs = [pd.read_csv(os.path.join(absolute_data_folder, tp[2:].replace('.csv', '.labels.csv')), index_col=0) for tp in test_paths]
    test_data_list = [scaler.transform(test_df.iloc[:, 1:-1].values) for test_df in test_dfs]
    test_labels_list = [test_df.iloc[:, -1].values for test_df in test_dfs]
    test_multivariate_labels_list = [test_multivariate_labels_df.values for test_multivariate_labels_df in test_multivariate_labels_dfs]
    contamination_list = [test_labels.sum() / len(test_labels) for test_labels in test_labels_list]
    contamination_list = [np.nextafter(0, 1) if contamination == 0. else contamination for contamination in contamination_list]

    test_file_names = [os.path.basename(tp) for tp in test_paths]
    # contamination_list = [test_labels.sum() / len(test_labels) for test_labels in test_labels_list]
    # contamination = train_labels.sum() / len(train_labels)
    # # Use smallest positive float as contamination if there are no anomalies in dataset
    # contamination = np.nextafter(0, 1) if contamination == 0. else contamination

    return (train_data, train_labels), (test_file_names, test_data_list, test_labels_list, test_multivariate_labels_list, contamination_list)


def process_metric_results(
                test_filename,
                metric,
                ready_metric_df: DataFrame,
                vus_pr_with_interpretability_df: DataFrame,
                univariate_anomaly_scores: ndarray[Any, dtype[Any]],
                dimension_contribution: ndarray,
                test_filenames,
                test_labels_list,
                test_multivariate_labels_list,
                ):
    from metrics.interpretability_metrics import AVUSI, IndependentNDCG
    if isinstance(metric, FFVUS):
        metric_data_dict = metric.score(test_labels_list[test_filenames.index(test_filename)], univariate_anomaly_scores)
        ready_metric_df.loc[test_filename, metric.name] = metric_data_dict['value']
        ready_metric_df.loc[test_filename, f'{metric.name}_computation_time'] = metric_data_dict['computation_time']
    elif isinstance(metric, IndependentNDCG):
        all_results_dict = metric.score_for_different_k(
            univariate_anomaly_scores,
            test_multivariate_labels_list[test_filenames.index(test_filename)],
            dimension_contribution
        )
        for key, metric_data in all_results_dict.items():
            metric_id = metric.get_name_template().format(K=key[0], W=key[1])
            ready_metric_df.loc[test_filename, metric_id] = metric_data['value']
            ready_metric_df.loc[test_filename, f'{metric_id}_computation_time'] = metric_data['computation_time']
    elif isinstance(metric, AVUSI):
        all_results_dict = metric.score_for_different_k(
            test_labels_list[test_filenames.index(test_filename)],
            univariate_anomaly_scores,
            test_multivariate_labels_list[test_filenames.index(test_filename)],
            dimension_contribution
        )

        for key, metric_data in all_results_dict.items():
            if 'n_interpretability_sensitivity_levels' in key:
                flatten_columns_l = [f'{key}_M_{m_index}' for m_index, m in enumerate(all_results_dict[key])]
                vus_pr_with_interpretability_df.loc[test_filename, flatten_columns_l] = all_results_dict[key]
                continue
            metric_id = metric.get_name_template().format(K=key[0], W=key[1], M=key[2])
            ready_metric_df.loc[test_filename, metric_id] = metric_data['value']
            ready_metric_df.loc[test_filename, f'{metric_id}_computation_time'] = metric_data['computation_time']

            assert f'n_interpretability_sensitivity_levels_{key[2]}_M_list' in all_results_dict, f'M_list should be in the results dict for interpretability sensitivity levels {key[2]}'
            flatten_columns = [f'n_interpretability_sensitivity_levels_{key[2]}_vus_pr_list_M_{l_index}'
                               for l_index, l in
                               enumerate(
                                   all_results_dict[f'n_interpretability_sensitivity_levels_{key[2]}_M_list'])]
            vus_pr_with_interpretability_df.loc[test_filename, flatten_columns] = metric_data['vus_pr_list']
    else:
        print(f'Skip metric {metric.name} since it is not supported')
