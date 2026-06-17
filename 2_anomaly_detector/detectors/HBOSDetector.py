import datetime

from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from detectors.Detector import Detector
from detectors.hbos.model import HBOS


class HBOSDetector(Detector):
    def __init__(self, customParameters, config):
        super().__init__(customParameters=customParameters, config=config)
        self.n_bins = customParameters.n_bins
        self.alpha = customParameters.alpha
        self.bin_tol = customParameters.bin_tol
        self.random_state = customParameters.random_state
        self.contamination = customParameters.contamination


    def name(self) -> str:
        return "HBOS"
    def _init_model(self):
        print('Unsupervised model, no init model required, skipping...')
        # from hbos import HBOS as HBOS_Model
        # self.model = HBOS_Model(n_bins=self.n_bins, alpha=self.alpha, bin_tol=self.bin_tol, random_state=self.random_state, contamination=self.contamination)

    def train(self, data, labels=None):
        print("HBOS does not require training, skipping...")
        return 0.0

    def predict(self, test_dict):
        (test_filenames,
         test_data_list,
         test_labels_list,
         test_multivariate_labels_list,
         test_contamination_list) = test_dict['test_filenames'], test_dict['test_data_list'], test_dict['test_labels_list'], test_dict['test_multivariate_labels_list'], test_dict['test_contamination_list']

        result_dict = {test_filename: {} for test_filename in test_filenames}

        for test_filename, data, labels, multivariate_labels, contamination in tqdm(zip(test_filenames,
                                                                                        test_data_list,
                                                                                        test_labels_list,
                                                                                        test_multivariate_labels_list,
                                                                                        test_contamination_list),
                                        total=len(test_filenames), desc="Executing test files"):
            multivariate_label_df = pd.DataFrame(multivariate_labels)
            if multivariate_label_df is not None:
                assert multivariate_label_df.shape[0] == data.shape[0]
                test_filepath = os.path.join(self.absolute_dataOutputDir, test_filename)
                os.makedirs(test_filepath, exist_ok=True)
                multivariate_label_df.to_csv(os.path.join(test_filepath, 'docker-algorithm-multivariate-labels.csv'))


            start_process_time = datetime.datetime.now()
            clf = HBOS(
                contamination=contamination,
                n_bins=self.customParameters.n_bins,
                alpha=self.customParameters.alpha,
                tol=self.customParameters.bin_tol
            )
            clf.fit(data)
            scores = clf.decision_scores_

            end_process_time = datetime.datetime.now()
            total_time = (end_process_time - start_process_time).total_seconds()
            result_dict[test_filename]['execute_main_time'] = total_time
            result_dict[test_filename]['scores'] = scores if scores.ndim == 2 else scores[:, np.newaxis]
            # with open(self.config.runtimeOutput, 'w') as f:
            #     f.write('Total:{0}\n'.format(total_time))
            #     f.write('Average_observation:{0}\n'.format(total_time / len(data)))
            # np.savetxt(os.path.join(self.absolute_dataOutputDir, test_filename, f'docker-algorithm-scores.csv'), scores, delimiter=",")

            # scores_per_var = clf.decision_scores_per_var
            # pd.DataFrame(scores_per_var).to_csv(config.anomalyScorePerVarOutput, index=False, header=None)
            # scores_per_var_ranking = np.argsort(-scores_per_var, axis=1)
            # pd.DataFrame(scores_per_var_ranking).to_csv(config.anomalyRankingOutput, index=False, header=None)

            scores_per_var = clf.decision_scores_per_var
            result_dict[test_filename]['scores_per_var'] = scores_per_var
            # pd.DataFrame(scores_per_var).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, f'docker-algorithm-scores-per-var.csv'),
            #     index=False,
            #     header=None)
            # scores_per_var_ranking = np.argsort(-scores_per_var, axis=1)
            # pd.DataFrame(scores_per_var_ranking).to_csv(
            #     os.path.join(absolute_dataOutput, test_filename, f'docker-algorithm-scores-per-var-ranking.csv'),
            #     index=False, header=None)
            dimension_contribution = clf.dimension_contribution
            result_dict[test_filename]['dimension_contribution'] = dimension_contribution
            # pd.DataFrame(dimension_contribution).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, 'docker-algorithm-dimension-contribution.csv'),
            #     index=False,
            #     header=None)

        return result_dict
