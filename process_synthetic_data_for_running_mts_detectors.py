import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig


import pandas as pd
import shutil
import os
import json
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def get_project_root() -> Path:
    return Path(__file__).parent

@hydra.main(config_path="conf", config_name="config.yaml")
def main(config: DictConfig):
    """
    :param cfg:
    :return:
    """
    print(config)
    project_root_dir = get_project_root()
    synthetic_data_dir = os.path.join(project_root_dir, config.synthetic_generation.generated_zip_data_dir)
    training_data_file = os.path.join(synthetic_data_dir, "synthetic_training.csv.zip")
    training_df = pd.read_csv(training_data_file, compression='zip')
    training_df.index.name ='timestamp'


    dfs_dict = dict({})
    for f in os.listdir(synthetic_data_dir):
        if f.endswith('.csv.zip') and f != 'synthetic_training.csv.zip':
            batch_file_path = os.path.join(synthetic_data_dir, f)
            df = pd.read_csv(batch_file_path, compression='zip')
            flag_columns = [f for f in df.columns if f.startswith('AnomalyFlag')]
            df['is_anomaly'] = df[df[flag_columns] != 'Normal'].any(axis=1).astype(float)
            df[flag_columns] = (df[flag_columns] != 'Normal').astype(float)
            dfs_dict[f.replace('.csv.zip','')] = df
    assert len(dfs_dict.keys()) > 0

    timeeval_output_dir = os.path.join(project_root_dir, config.timeeval_mts_data_dir)

    timeeval_output_dir_semisupervised = os.path.join(timeeval_output_dir, 'semisupervised')
    os.makedirs(timeeval_output_dir_semisupervised, exist_ok=True)

    os.makedirs(timeeval_output_dir, exist_ok=True)
    count_discard = 0
    datasets_json_for_timeeval_unsupervised = dict({})
    feature_columns = [f for f in dfs_dict[list(dfs_dict.keys())[0]].columns if f.startswith('Sensor')]
    save_columns = feature_columns + ['is_anomaly']
    flag_columns = [f for f in dfs_dict[list(dfs_dict.keys())[0]].columns if f.startswith('AnomalyFlag')]


    training_df['is_anomaly'] = training_df[training_df[flag_columns] != 'Normal'].any(axis=1).astype(float)

    training_df = training_df[feature_columns + ['is_anomaly']]
    assert training_df['is_anomaly'].sum() == 0
    training_df.to_csv(os.path.join(timeeval_output_dir_semisupervised, f'synthetic_train.csv'), index=True)


    datasets_json_for_timeeval_semisupervised = dict({})
    datasets_json_for_timeeval_semisupervised_merged = dict({
        f'{config.mts_running_dataset}': {
            'train_paths': [f'./synthetic_train.csv'],
            'test_paths': [],
            'type': "synthetic test"
        }
    })

    for index, (idx, partition) in tqdm(enumerate(dfs_dict.items()), total=len(dfs_dict), desc='Processing partitions'):
        if (partition['is_anomaly'].sum() / partition.shape[0] < 0.5) and(partition['is_anomaly'].sum() > 0):
            partition.index.name = 'timestamp'
            idx = int(idx.split('_')[-1])

            datasets_json_for_timeeval_unsupervised[f'synthetic_batch_{idx}.csv'] = dict({
                'test_path': f'./synthetic_batch_{idx}.csv',
                'type': "synthetic test"
            })

            partition[feature_columns + ['is_anomaly']].to_csv(
                os.path.join(timeeval_output_dir_semisupervised, f'synthetic_batch_{idx}.csv'), index=True)

            partition[flag_columns].to_csv(
                os.path.join(timeeval_output_dir_semisupervised, f'synthetic_batch_{idx}.labels.csv'), index=True)

            datasets_json_for_timeeval_semisupervised[f'synthetic_batch_{idx}.csv'] = dict({
                'test_path': f'./synthetic_batch_{idx}.csv',
                'train_path': f'./synthetic_train.csv',
                'type': "synthetic test"
            })

            datasets_json_for_timeeval_semisupervised_merged[
                f'{config.mts_running_dataset}']['test_paths'].append(f'./synthetic_batch_{idx}.csv')
        else:
            count_discard += 1
            print(f'Skipping partition {idx} due to high anomaly ratio or none-anomaly points')

    json.dump(datasets_json_for_timeeval_semisupervised,
              open(os.path.join(timeeval_output_dir_semisupervised, 'datasets.json'), 'w'), indent=4)
    json.dump(datasets_json_for_timeeval_semisupervised_merged,
              open(os.path.join(timeeval_output_dir_semisupervised, 'datasets_merged.json'), 'w'), indent=4)

    print(f'Discarded {count_discard} partitions due to high anomaly ratio or none-anomaly points')

    print(f'Saved {len(dfs_dict) - count_discard} partitions to {timeeval_output_dir_semisupervised}')

if __name__ == '__main__':
    print(f'Arguments: {sys.argv}')

    main()