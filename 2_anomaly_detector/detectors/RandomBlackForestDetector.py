import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from detectors.Detector import Detector
from detectors.random_black_forest.model import RandomBlackForestAnomalyDetector
from utils import transform_to_dimension_contribution


class RandomBlackForestDetector(Detector):

    def __init__(self, customParameters, config):
        super().__init__(customParameters=customParameters, config=config)

    def name(self) -> str:
        return "auto_encoder"

    def _init_model(self):
        self.model = RandomBlackForestAnomalyDetector(**self.customParameters)

    def post_random_black_forest(self, preds_per_var:np.ndarray, args: dict) -> np.ndarray:
        window_size = args.get("train_window_size", 50)
        # scores[:window_size, :] = scores[window_size+1, :]
        preds_per_var[:window_size, :] = preds_per_var[window_size + 1, :]
        return preds_per_var

    def train(self, data, labels=None)-> float:
        self._init_model()
        start_process_time = datetime.datetime.now()
        self.model.fit(data)
        self.model.save(Path(self.absolute_modelOutputFile))
        end_process_time = datetime.datetime.now()
        total_time = (end_process_time - start_process_time).total_seconds()
        pd.DataFrame([total_time], columns=['train_time']).to_csv(
            os.path.join(self.absolute_modelOutputDir, 'docker-algorithm-train-time.csv'), index=False)
        return total_time
    def predict(self, test_dict):
        self.model = RandomBlackForestAnomalyDetector.load(Path(self.absolute_modelInputFile))

        (test_filenames,
         test_data_list,
         test_labels_list,
         test_multivariate_labels_list,
         test_contamination_list) = test_dict['test_filenames'], test_dict['test_data_list'], test_dict[
            'test_labels_list'], test_dict['test_multivariate_labels_list'], test_dict['test_contamination_list']

        result_dict = {test_filename: {} for test_filename in test_filenames}
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

            start_process_time = datetime.datetime.now()
            preds = self.model.predict(data)
            preds = self.post_random_black_forest(preds, args=self.customParameters)
            preds_per_var = np.abs(preds - data)
            preds = np.sum(np.abs(preds - data), axis=1, keepdims=True)

            end_process_time = datetime.datetime.now()
            total_time = (end_process_time - start_process_time).total_seconds()
            result_dict[test_filename]['execute_main_time'] = total_time
            # pd.DataFrame([total_time], columns=['execute_time']).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, 'docker-algorithm-execute-time.csv'), index=False)

            # np.savetxt(os.path.join(self.absolute_dataOutputDir, test_filename, f'docker-algorithm-scores.csv'), preds,
            #            delimiter=",")

            # scores_per_var = clf.decision_scores_per_var
            # pd.DataFrame(scores_per_var).to_csv(config.anomalyScorePerVarOutput, index=False, header=None)
            # scores_per_var_ranking = np.argsort(-scores_per_var, axis=1)
            # pd.DataFrame(scores_per_var_ranking).to_csv(config.anomalyRankingOutput, index=False, header=None)

            result_dict[test_filename]['scores'] = preds
            result_dict[test_filename]['scores_per_var'] = preds_per_var
            # pd.DataFrame(scores_per_var).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, f'docker-algorithm-scores-per-var.csv'),
            #     index=False,
            #     header=None)
            # scores_per_var_ranking = np.argsort(-scores_per_var, axis=1)
            # pd.DataFrame(scores_per_var_ranking).to_csv(
            #     os.path.join(absolute_dataOutput, test_filename, f'docker-algorithm-scores-per-var-ranking.csv'),
            #     index=False, header=None)
            dimension_contribution = transform_to_dimension_contribution(preds_per_var)
            result_dict[test_filename]['dimension_contribution'] = dimension_contribution
            # pd.DataFrame(dimension_contribution).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, 'docker-algorithm-dimension-contribution.csv'),
            #     index=False,
            #     header=None)
        return result_dict