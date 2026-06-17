import logging
import os
import shutil

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utils import load_data_from_json, get_project_root

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config.yaml")
def main(config: DictConfig):
    print(config)

    (train_data, train_labels), (test_filenames, test_data_list, test_labels_list, test_multivariate_labels_list,
                                 test_contamination_list) = load_data_from_json(config)


    test_dict = {
        'test_filenames': test_filenames,
        'test_data_list': test_data_list,
        'test_labels_list': test_labels_list,
        'test_multivariate_labels_list': test_multivariate_labels_list,
        'test_contamination_list': test_contamination_list
    }

    project_root_dir = get_project_root()
    results_folder_of_all_detectors = config.results_folder_of_all_detectors

    results_dir = os.path.join(project_root_dir, results_folder_of_all_detectors)
    aggregated_score_dir = os.path.join(results_dir, 'avg_ens')
    os.makedirs(aggregated_score_dir, exist_ok=True)

    all_supported_detectors = config.mts_supported_detectors
    running_detectors = ['hbos','cblof','tran_ad']
    logger.warning(f"Using hardcoded supported detectors: {running_detectors} instead of {config.mts_supported_detectors} for testing")


    weight_matrix = np.ones(len(running_detectors)) / len(running_detectors)
    for test_filename in (pbar := tqdm(test_filenames, total=len(test_filenames), desc="Processing each test files for avg_ens")):
        pbar.set_postfix({"current_test_file": test_filename})

        aggregated_univariate_score_path = os.path.join(aggregated_score_dir, test_filename.replace('.out', '.csv'), 'anomaly-scores.csv')
        aggregated_multivariate_dimension_contribution_path = os.path.join(aggregated_score_dir, test_filename.replace('.out', '.csv'), 'docker-algorithm-dimension-contribution.csv')
        os.makedirs(os.path.dirname(aggregated_univariate_score_path), exist_ok=True)

        univariate_anomaly_scores, multivariate_dimension_contribution = load_scores_and_dimension_contribution_not_processed(results_dir, running_detectors, test_filename)

        new_univariate_anomaly_scores, new_multivariate_dimension_contribution = calculate_new_scores_with_weights_of_detectors(
            univariate_anomaly_scores, multivariate_dimension_contribution, weight_matrix)

        pd.DataFrame(new_multivariate_dimension_contribution).to_csv(
            aggregated_multivariate_dimension_contribution_path,
            header=False, index=False)
        pd.DataFrame(new_univariate_anomaly_scores).to_csv(aggregated_univariate_score_path,
                                                           header=False, index=False)

def calculate_new_scores_with_weights_of_detectors(univariate_anomaly_scores, multivariate_dimension_contribution,
                                                   weight_matrix):
    weighted_univariate_anomaly_scores = univariate_anomaly_scores * weight_matrix[np.newaxis,
                                                                     :].astype(np.float32)  # shape (n_samples, n_algorithms)
    weighted_univariate_anomaly_scores += 1e-5  # Add a small constant to avoid division by zero

    new_univariate_anomaly_scores = np.average(univariate_anomaly_scores, weights=weight_matrix, axis=1).astype(np.float32)  # shape (n_samples,)

    contribution_univariate_anomaly_scores = weighted_univariate_anomaly_scores / weighted_univariate_anomaly_scores.sum(
        axis=1, keepdims=True)
    new_multivariate_dimension_contribution = np.sum(
        multivariate_dimension_contribution * contribution_univariate_anomaly_scores[:, :, np.newaxis],
        axis=1)  # shape (n_samples, n_algorithms, n_dimensions)

    new_univariate_anomaly_scores = MinMaxScaler().fit_transform(new_univariate_anomaly_scores.reshape(-1, 1)).reshape(-1)  # shape (n_samples,)
    return new_univariate_anomaly_scores, new_multivariate_dimension_contribution

def load_scores_and_dimension_contribution_not_processed(merged_results_dir, detector_order, test_filename):

    univariate_anomaly_scores_matrix = []
    multivariate_dimension_contribution_matrix = []

    for alg in detector_order:
        anomaly_score_path = os.path.join(merged_results_dir, alg,
                                          test_filename.replace('.out', '.csv'),
                                          'anomaly-scores.csv')
        univariate_anomaly_scores = pd.read_csv(anomaly_score_path, header=None).values.reshape(-1)
        dimension_contribution_path = os.path.join(merged_results_dir, alg,
                                                   test_filename.replace('.out', '.csv'),
                                                   'docker-algorithm-dimension-contribution.csv')
        dimension_contribution = pd.read_csv(dimension_contribution_path, header=None).values

        assert dimension_contribution.shape[0] == univariate_anomaly_scores.shape[0], f"Dimension contribution and anomaly scores should have the same number of samples for {test_filename} in {alg}"
        univariate_anomaly_scores_matrix.append(univariate_anomaly_scores)
        multivariate_dimension_contribution_matrix.append(dimension_contribution)

    univariate_anomaly_scores_matrix = np.stack(univariate_anomaly_scores_matrix, axis=1)  # shape (n_samples, n_algorithms)
    multivariate_dimension_contribution_matrix = np.stack(multivariate_dimension_contribution_matrix, axis=1)  # shape (n_samples, n_algorithms, n_dimensions)

    return univariate_anomaly_scores_matrix, multivariate_dimension_contribution_matrix

if __name__ == '__main__':
    main()