import datetime
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from detectors.Detector import Detector
from detectors.encdec_ad.model import EncDecAD
from post_processing_utils.window import ReverseWindowing
from utils import transform_to_dimension_contribution


class EncDecADDetector(Detector):

    def __init__(self, customParameters, config):
        super().__init__(customParameters=customParameters, config=config)

    def name(self) -> str:
        return "encdec_ad"

    def _init_model(self):
        self.model = EncDecAD(**dict(self.customParameters))

    def post_encdec_ad(self, scores: np.ndarray, args: dict) -> np.ndarray:
        window_size = args.get("anomaly_window_size", 30)
        return ReverseWindowing(window_size=2 * window_size).fit_transform(scores)

    def post_encdec_ad_multivariate(self, scores: np.ndarray, args: dict) -> np.ndarray:
        assert scores.ndim == 2, "Scores should be a 2D array with shape (n_samples, n_variables)"
        window_size = args.get("hyper_params", {}).get("anomaly_window_size", 30)
        new_scores = []
        for i in range(scores.shape[1]):
            new_scores.append(ReverseWindowing(window_size=2 * window_size).fit_transform(scores[:, i]))

        return np.stack(new_scores, axis=1)


    def train(self, data, labels=None)-> float:
        self._init_model()
        start_process_time = datetime.datetime.now()
        os.makedirs(os.path.dirname(self.absolute_modelOutputFile), exist_ok=True)
        os.makedirs(os.path.dirname(self.absolute_modelInputFile), exist_ok=True)
        self.model.fit(data, self.absolute_modelOutputFile)
        self.model.save(self.absolute_modelOutputFile)
        end_process_time = datetime.datetime.now()
        total_time = (end_process_time - start_process_time).total_seconds()
        pd.DataFrame([total_time], columns=['train_time']).to_csv(
            os.path.join(self.absolute_modelOutputDir, 'docker-algorithm-train-time.csv'), index=False)
        return total_time
    def predict(self, test_dict):
        # self.model = AutoEn.load(self.absolute_modelInputFile)
        self.model = EncDecAD.load(self.absolute_modelInputFile, **dict(self.customParameters))

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
            anomaly_scores, anomaly_scores_per_var = self.model.anomaly_detection(data)
            end_process_time = datetime.datetime.now()

            anomaly_scores = self.post_encdec_ad(anomaly_scores, args=self.customParameters)
            anomaly_scores = anomaly_scores[:, np.newaxis]

            anomaly_scores_per_var = self.post_encdec_ad_multivariate(anomaly_scores_per_var, args=self.customParameters)

            total_time = (end_process_time - start_process_time).total_seconds()
            result_dict[test_filename]['execute_main_time'] = total_time
            result_dict[test_filename]['scores'] = anomaly_scores
            # pd.DataFrame([total_time], columns=['execute_time']).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, 'docker-algorithm-execute-time.csv'), index=False)

            # np.savetxt(os.path.join(self.absolute_dataOutputDir, test_filename, f'docker-algorithm-scores.csv'), preds,
            #            delimiter=",")

            # scores_per_var = clf.decision_scores_per_var
            # pd.DataFrame(scores_per_var).to_csv(config.anomalyScorePerVarOutput, index=False, header=None)
            # scores_per_var_ranking = np.argsort(-scores_per_var, axis=1)
            # pd.DataFrame(scores_per_var_ranking).to_csv(config.anomalyRankingOutput, index=False, header=None)

            scores_per_var = transform_to_dimension_contribution(anomaly_scores_per_var)
            result_dict[test_filename]['scores_per_var'] = scores_per_var
            # pd.DataFrame(scores_per_var).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, f'docker-algorithm-scores-per-var.csv'),
            #     index=False,
            #     header=None)
            # scores_per_var_ranking = np.argsort(-scores_per_var, axis=1)
            # pd.DataFrame(scores_per_var_ranking).to_csv(
            #     os.path.join(absolute_dataOutput, test_filename, f'docker-algorithm-scores-per-var-ranking.csv'),
            #     index=False, header=None)
            dimension_contribution = transform_to_dimension_contribution(anomaly_scores_per_var)
            result_dict[test_filename]['dimension_contribution'] = dimension_contribution
            # pd.DataFrame(dimension_contribution).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, 'docker-algorithm-dimension-contribution.csv'),
            #     index=False,
            #     header=None)
        return result_dict