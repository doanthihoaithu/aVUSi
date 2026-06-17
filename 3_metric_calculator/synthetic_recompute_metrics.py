import os
from multiprocessing import Pool

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from metrics.ffvus.ffvus_metrics import FFVUS
from metrics.interpretability_metrics import AVUSI, IndependentNDCG
from utils import load_data_from_json, get_project_root, process_metric_results


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(config: DictConfig):
    print(config)

    config.update({'mts_running_dataset':'settings_five'})
    dataset_name = config.mts_running_dataset
    parameter_sensitivity_config = config.parameter_sensitivity_analysis.datasets[dataset_name]
    print(f"Loaded parameter analysis config for {dataset_name}: {parameter_sensitivity_config}")

    project_root_dir = get_project_root()
    merged_results_dir = os.path.join(project_root_dir, config.merge_results.results_path)
    supported_detectors = config.mts_supported_detectors

    #TODO uncomment this line when results of all supported detectors being available,
    # assume currently we only have results of 3 detectors and we want to compute new metrics for them first
    supported_detectors = ['hbos','tran_ad', 'cblof']

    interpretability_conditional_metrics = [AVUSI(**parameter_sensitivity_config)]
    interpretability_unconditional_metrics = [IndependentNDCG(**parameter_sensitivity_config)]
    accuracy_metrics = [FFVUS(slope=64)]
    interpretability_metrics = interpretability_unconditional_metrics + interpretability_conditional_metrics
    total_metrics = accuracy_metrics + interpretability_metrics

    (train_data, train_labels), (test_filenames, test_data_list, test_labels_list, test_multivariate_labels_list,
                                 test_contamination_list) = load_data_from_json(config)

    test_dict = {
        'test_filenames': test_filenames,
        'test_data_list': test_data_list,
        'test_univariate_labels_list': test_labels_list,
        'test_multivariate_labels_list': test_multivariate_labels_list,
    }

    scores_dir = merged_results_dir
    new_metrics_dir = os.path.join(os.path.dirname(merged_results_dir), 'new_metrics')

    print(f'Score directory: {scores_dir}')
    print(f'new_metrics_dir directory: {new_metrics_dir}')

    with Pool(10) as p:
        pool_inputs = [(alg, scores_dir, new_metrics_dir, test_dict, total_metrics) for alg in supported_detectors]
        list(tqdm(p.starmap(process_scores_of_a_detector_synthetic, pool_inputs)))

def process_scores_of_a_detector_synthetic(alg, scores_dir, new_metrics_dir, test_dict, total_metrics):
    test_filenames = [f.replace('.csv','.out') for f in test_dict['test_filenames'] if f.endswith('.csv')]
    test_labels_list = test_dict['test_univariate_labels_list']
    test_multivariate_labels_list = test_dict['test_multivariate_labels_list']
    ready_metric_file_path = os.path.join(new_metrics_dir, alg, 'results.csv')
    ready_metric_file_path_old = os.path.join(scores_dir, alg, 'results.csv')
    if os.path.exists(ready_metric_file_path):
        print(f"Found existing metric results for {alg} at ", ready_metric_file_path)
        ready_metric_df = pd.read_csv(ready_metric_file_path, index_col=0)
    else:
        os.makedirs(os.path.dirname(ready_metric_file_path), exist_ok=True)
        ready_metric_df = pd.DataFrame()
        ready_metric_df.index.name = 'test_batch_id'

    saved_result_path_for_interpretability = os.path.join(os.path.dirname(ready_metric_file_path),
                                                          'vus_pr_list.csv')
    if os.path.exists(saved_result_path_for_interpretability):
        vus_pr_with_interpretability_df = pd.read_csv(saved_result_path_for_interpretability, index_col=0)
        if (vus_pr_with_interpretability_df.shape[0] != len(test_filenames)) or (vus_pr_with_interpretability_df.shape[1] == 0):
            print(
                f"Warning: The number of rows in existing interpretability results ({vus_pr_with_interpretability_df.shape[0]}) does not match the number of test files ({len(test_filenames)}). Re-initializing the interpretability results dataframe.")
            vus_pr_with_interpretability_df = pd.DataFrame(index=test_filenames)
        else:
            print(f"Found existing interpretability metric results for {alg} with shape {vus_pr_with_interpretability_df.shape} at ", saved_result_path_for_interpretability)
    else:
        vus_pr_with_interpretability_df = pd.DataFrame(index=test_filenames)

    for test_filename in (
    pbar2 := tqdm(test_filenames, desc=f"Processing test files for {alg}",
                  leave=True, position=0, total=len(test_filenames))):
        pbar2.set_postfix({"current_test_file": test_filename})
        if test_filename == 'machine-3-10.out':
            print('Debugging for machine-3-10.out')

        anomaly_score_path = os.path.join(scores_dir, alg, test_filename.replace('.out','.csv'), 'anomaly-scores.csv')
        univariate_anomaly_scores = pd.read_csv(anomaly_score_path, header=None).values.reshape(-1)
        dimension_contribution_path = os.path.join(scores_dir, alg, test_filename.replace('.out','.csv'),
                                                   'docker-algorithm-dimension-contribution.csv')
        dimension_contribution = pd.read_csv(dimension_contribution_path, header=None).values

        assert dimension_contribution.shape[0] == univariate_anomaly_scores.shape[0], f"Dimension contribution and anomaly scores should have the same number of samples for {test_filename} in {alg}"
        for metric in total_metrics:
            process_metric_results(
                test_filename,
                metric,
                ready_metric_df,
                vus_pr_with_interpretability_df,
                univariate_anomaly_scores,
                dimension_contribution,
                test_filenames,
                test_labels_list,
                test_multivariate_labels_list
            )

    ready_metric_df.to_csv(ready_metric_file_path, index=True)
    print(f'Saved new metric results for {alg} at ', ready_metric_file_path)

    vus_pr_with_interpretability_df.to_csv(saved_result_path_for_interpretability)
    print(f'Save vus_pr with interpretability at', saved_result_path_for_interpretability)

    if alg == 'avg_ens':
        ready_metric_df.index = ready_metric_df.index.str.replace('.out', '.csv')
        ready_metric_df['dataset'] = 'settings_six'
        ready_metric_df.to_csv(ready_metric_file_path_old, index=True)
        print(f'Saved new metric results for {alg} at ', ready_metric_file_path_old)


if __name__ == '__main__':
    main()