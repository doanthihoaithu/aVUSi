import os
import shutil
from pathlib import Path
import pandas as pd

import hydra
from keras.src.utils import config
from omegaconf import DictConfig

from utils import get_project_root


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    merge_result_cfg = cfg.merge_results
    print(merge_result_cfg)
    results_path = merge_result_cfg.notebook_visualization.results_path
    project_root_dir = get_project_root()
    merged_results_folder = os.path.join(project_root_dir, results_path)
    merged_results_folder = os.path.join(merged_results_folder, 'new_metrics')
    results_dfs = []
    for selected_results_folder in os.listdir(merged_results_folder):
        if selected_results_folder == 'results.csv' or selected_results_folder.startswith('.DS_Store'):
            continue
        selected_results_folder_path = merged_results_folder + '/' + selected_results_folder
        results_file_path = selected_results_folder_path + '/results.csv'
        df = pd.read_csv(results_file_path)
        df['algorithm'] = selected_results_folder
        if 'test_batch_id' in df.columns:
            df['test_batch_id'] = df['test_batch_id'].apply(lambda x: x.replace('.csv', '.out'))
        results_dfs.append(df)

    merged_df = pd.concat(results_dfs, ignore_index=True, axis=0)

    os.makedirs(merged_results_folder, exist_ok=True)
    merged_df.to_csv(os.path.join(os.path.dirname(merged_results_folder), 'results.csv'), index=False)

    print(f'save merged results at {os.path.join(os.path.dirname(merged_results_folder), "results.csv")}')
    print(f'Number of detectors', merged_df['algorithm'].nunique())
    print(merged_df['algorithm'].unique())
    print(merged_df['algorithm'].value_counts())


if __name__ == '__main__':
    main()