import datetime
import os
from time import time


import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from detectors.Detector import Detector
from detectors.mtad_gat.MTAD_GAT import MTAD_GAT
from training import Trainer
from utils import transform_to_dimension_contribution, color, SlidingWindowDataset, create_data_loaders, plot_losses


class MTAD_GATDetector(Detector):
    def __init__(self, customParameters, config):
        super().__init__(customParameters=customParameters, config=config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def name(self) -> str:
        return self.customParameters.model

    def _init_model(self):
        args = self.customParameters
        self.model = MTAD_GAT(
            args
            # mag_window=args.customParameters.mag_window_size,
            # score_window=args.customParameters.score_window_size,
            # batch_size=args.customParameters.batch_size,
            # threshold=args.customParameters.threshold,
            # around_window_size=args.customParameters.context_window_size,
            # kernel_size=args.customParameters.kernel_size,
            # window_size=args.customParameters.window_size,
            # gamma=args.customParameters.gamma,
            # channel_sizes=args.customParameters.linear_layer_shape,
            # latent_size=args.customParameters.latent_size,
            # num_features=args.num_features,
            # split=args.customParameters.split,
            # early_stopping_patience=args.customParameters.early_stopping_patience,
            # early_stopping_delta=args.customParameters.early_stopping_delta
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.customParameters.get('lr', 0.001),
                                          weight_decay=self.customParameters.get('weight_decay', 1e-5))
        self.forecast_criterion = nn.MSELoss()

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            window_size=self.customParameters.get('n_window', 30),
            n_features=self.customParameters.n_feats,
            target_dims=None,
            n_epochs=self.customParameters.get('epochs', 100),
            batch_size=self.customParameters.get('batch_size', 128),
            init_lr=self.customParameters.get('lr', 0.001),
            forecast_criterion = self.forecast_criterion,
            recon_criterion = None,
            use_cuda=self.device == 'cuda',
            dload=self.absolute_modelOutputDir,
            log_dir=self.absolute_modelOutputDir,
            log_tensorboard=False,
        )


    def train(self, data, labels=None)-> float:
        data = torch.DoubleTensor(data)
        self._init_model()

        fname = f'{self.trainer.dload}/model.pt'
        if os.path.exists(fname) and (not self.customParameters.retrain):
            print(f"{color.GREEN}Loading pre-trained model: {self.customParameters.model}{color.ENDC}")
            self.trainer.load(f"{self.trainer.dload}/model.pt")
            self.model = self.trainer.model
            training_time = pd.read_csv(self.absolute_dataOutputDir + '/docker-algorithm-train-time.csv')['train_time'].iloc[0]
            return training_time

        else:
            train_dataset = SlidingWindowDataset(data=data, window=self.customParameters.n_window)
            train_loader, val_loader, test_loader = create_data_loaders(
                train_dataset=train_dataset, batch_size=self.customParameters.get('batch_size', 128), val_split=0.2
            )

            start_training_time = time()
            self.trainer.fit(train_loader=train_loader, val_loader=val_loader)
            end_training_time = time()
            self.trainer.save(f"model.pt")
            plot_losses(self.trainer.losses, save_path=self.trainer.dload, plot=True)
            total_time = end_training_time - start_training_time
            return total_time


    def predict(self, test_dict):
        self.trainer.load(f"{self.trainer.dload}/model.pt")
        self.model = self.trainer.model

        (test_filenames,
         test_data_list,
         test_labels_list,
         test_multivariate_labels_list,
         test_contamination_list) = test_dict['test_filenames'], test_dict['test_data_list'], test_dict[
            'test_labels_list'], test_dict['test_multivariate_labels_list'], test_dict['test_contamination_list']

        result_dict = {test_filename: {} for test_filename in test_filenames}

        torch.zero_grad = True
        self.model.eval()

        print(f'{color.HEADER}Testing {self.customParameters.model} on {self.customParameters.dataset}{color.ENDC}')
        for test_filename, data, labels, multivariate_labels, contamination in tqdm(zip(test_filenames,
                                                                                        test_data_list,
                                                                                        test_labels_list,
                                                                                        test_multivariate_labels_list,
                                                                                        test_contamination_list),
                                                                                    total=len(test_filenames),
                                                                                    desc="Executing test files"):
            multivariate_label_df = pd.DataFrame(multivariate_labels)
            if multivariate_label_df is not None:
                assert multivariate_label_df.shape[0] == data.shape[0]
                test_filepath = os.path.join(self.absolute_dataOutputDir, test_filename)
                os.makedirs(test_filepath, exist_ok=True)
                multivariate_label_df.to_csv(os.path.join(test_filepath, 'docker-algorithm-multivariate-labels.csv'))

            data = torch.tensor(data=data, dtype=torch.float64)
            window_data = SlidingWindowDataset(data=data, window=self.customParameters.n_window)
            test_dataloader = create_data_loaders(window_data, batch_size=self.customParameters.batch_size, val_split=0.0)[0]

            start_process_time = datetime.datetime.now()
            preds = []
            pred_errors = []
            for x, y in test_dataloader:
                x = x.to(device=self.device).float()
                y = y.to(device=self.device).float()
                batch_preds = self.model(x).detach().cpu().numpy()
                if batch_preds.ndim == 3:
                    batch_preds = batch_preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)
                preds.append(batch_preds)
                pred_errors.append(abs(batch_preds - y.detach().cpu().numpy()))
            preds = np.row_stack(preds)
            pred_errors = np.row_stack(pred_errors)

            scores = pred_errors.sum(axis=1, keepdims=True)
            scores_per_var = pred_errors

            # preds_per_var = np.abs(preds - data[:,-1,:].numpy())
            # preds = np.sum(np.abs(preds - data[:,-1,:].numpy()), axis=1, keepdims=True)
            # loss_per_var = loss.copy()
            # scores = np.sum(loss, axis=1, keepdims=True)
            # preds = np.sum(np.abs(preds - data[:, -1, :].numpy()), axis=1, keepdims=True)

            end_process_time = datetime.datetime.now()
            total_time = (end_process_time - start_process_time).total_seconds()
            result_dict[test_filename]['execute_main_time'] = total_time
            result_dict[test_filename]['scores'] = scores
            # pd.DataFrame([total_time], columns=['execute_time']).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, 'docker-algorithm-execute-time.csv'), index=False)

            # np.savetxt(os.path.join(self.absolute_dataOutputDir, test_filename, f'docker-algorithm-scores.csv'), preds,
            #            delimiter=",")

            # scores_per_var = clf.decision_scores_per_var
            # pd.DataFrame(scores_per_var).to_csv(config.anomalyScorePerVarOutput, index=False, header=None)
            # scores_per_var_ranking = np.argsort(-scores_per_var, axis=1)
            # pd.DataFrame(scores_per_var_ranking).to_csv(config.anomalyRankingOutput, index=False, header=None)


            result_dict[test_filename]['scores_per_var'] = scores_per_var
            # pd.DataFrame(scores_per_var).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, f'docker-algorithm-scores-per-var.csv'),
            #     index=False,
            #     header=None)
            # scores_per_var_ranking = np.argsort(-scores_per_var, axis=1)
            # pd.DataFrame(scores_per_var_ranking).to_csv(
            #     os.path.join(absolute_dataOutput, test_filename, f'docker-algorithm-scores-per-var-ranking.csv'),
            #     index=False, header=None)
            dimension_contribution = transform_to_dimension_contribution(scores_per_var)
            result_dict[test_filename]['dimension_contribution'] = dimension_contribution
            # pd.DataFrame(dimension_contribution).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, 'docker-algorithm-dimension-contribution.csv'),
            #     index=False,
            #     header=None)

        return result_dict

