import json
import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_project_root() -> Path:
    return Path(__file__).parent

def set_random_state(random_state) -> None:
    seed = random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


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

def transform_to_dimension_contribution(scores_per_var):
    # scores_per_var is a 2D array of shape (n_samples, n_features)
    # We want to transform it to dimension contribution such that higher scores indicate higher contribution to anomaly
    # One simple way is to normalize the scores_per_var for each sample and then take the negative log to get contribution
    # Adding a small epsilon to avoid log(0)
    epsilon = 1e-10
    normalized_scores = scores_per_var / (np.sum(scores_per_var, axis=1, keepdims=True) + epsilon)
    dimension_contribution = normalized_scores
    return dimension_contribution

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, horizon=1):
        padding = data[0].repeat(window, 1)
        self.data = torch.cat([padding, data], dim=0).double()
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):

        # x = self.data[index : index + self.window]
        # y = self.data[index + self.window : index + self.window + self.horizon]

        x = self.data[index: index+self.window, :]
        y = self.data[index+self.window: index + self.window + self.horizon, :]
        return x, y

    def __len__(self):
        return len(self.data) - self.window

def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()