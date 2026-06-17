import datetime
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow import keras

from detectors.Detector import Detector
from detectors.dae.model import AutoEn
from utils import transform_to_dimension_contribution


class DenoisingAutoEncoderDetector(Detector):
    def __init__(self, customParameters, config):
        super().__init__(customParameters=customParameters, config=config)

    def name(self) -> str:
        return "denoising_auto_encoder"

    def _init_model(self):
        self.model = AutoEn(**dict(self.customParameters))

    def train(self, data, labels=None) -> float:
        self._init_model()
        start_process_time = datetime.datetime.now()
        self.model.fit(data, self.absolute_modelOutputFile)
        shutil.make_archive(self.absolute_modelOutputFile, "zip", "check")
        end_process_time = datetime.datetime.now()
        total_time = (end_process_time - start_process_time).total_seconds()
        pd.DataFrame([total_time], columns=['train_time']).to_csv(
            os.path.join(self.absolute_modelOutputDir, 'docker-algorithm-train-time.csv'), index=False)
        return total_time

    def predict(self, test_dict):
        shutil.unpack_archive(self.absolute_modelInputFile + ".zip", "m", "zip")
        self.model = keras.models.load_model("m")

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
                                                                                    leave=True,
                                                                                    position=0,
                                                                                    desc="Executing test files"):
            multivariate_label_df = pd.DataFrame(multivariate_labels)
            if multivariate_label_df is not None:
                assert multivariate_label_df.shape[0] == data.shape[0]
                test_filepath = os.path.join(self.absolute_dataOutputDir, test_filename)
                os.makedirs(test_filepath, exist_ok=True)
                multivariate_label_df.to_csv(os.path.join(test_filepath, 'docker-algorithm-multivariate-labels.csv'))

            start_process_time = datetime.datetime.now()
            preds = self.model.predict(data)
            preds_per_var = np.abs(preds - data)
            preds = np.sum(np.abs(preds - data), axis=1, keepdims=True)

            end_process_time = datetime.datetime.now()
            total_time = (end_process_time - start_process_time).total_seconds()
            result_dict[test_filename]['execute_main_time'] = total_time
            result_dict[test_filename]['scores'] = preds

            scores_per_var = preds_per_var
            result_dict[test_filename]['scores_per_var'] = scores_per_var

            dimension_contribution = transform_to_dimension_contribution(preds_per_var)
            result_dict[test_filename]['dimension_contribution'] = dimension_contribution

        return result_dict