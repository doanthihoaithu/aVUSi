import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from detectors.AutoEncoderDetector import AutoEncoderDetector
from detectors.CBLOFDetector import CBLOFDetector
from detectors.COPODDetector import COPODDetector
from detectors.DenoisingAutoEncoderDetector import DenoisingAutoEncoderDetector
from detectors.Detector import Detector
from detectors.EncDecADDetector import EncDecADDetector
from detectors.HBOSDetector import HBOSDetector
from detectors.MTAD_GATDetector import MTAD_GATDetector
from detectors.RandomBlackForestDetector import RandomBlackForestDetector
from detectors.TranADDetector import TranADDetector
from utils import load_data_from_json


def init_detector(config: DictConfig)-> Detector:
    algorithm_name = config.mts_running_detector
    if algorithm_name == 'hbos':
        return HBOSDetector(customParameters=config.mts_current_custom_parameters, config=config)
    elif algorithm_name == 'cblof':
        return CBLOFDetector(customParameters=config.mts_current_custom_parameters, config=config)
    elif algorithm_name == 'copod':
        return COPODDetector(customParameters=config.mts_current_custom_parameters, config=config)
    elif algorithm_name == 'encdec_ad':
        return EncDecADDetector(customParameters=config.mts_current_custom_parameters, config=config)
    elif algorithm_name == 'random_black_forest':
        return RandomBlackForestDetector(customParameters=config.mts_current_custom_parameters, config=config)
    elif algorithm_name == 'mtad_gat':
        return MTAD_GATDetector(customParameters=config.mts_current_custom_parameters, config=config)
    elif algorithm_name == 'tran_ad':
        return TranADDetector(customParameters=config.mts_current_custom_parameters, config=config)
    elif algorithm_name == 'omni_anomaly':
        return TranADDetector(customParameters=config.mts_current_custom_parameters, config=config)
    elif algorithm_name == 'gdn':
        return TranADDetector(customParameters=config.mts_current_custom_parameters, config=config)

    # elif algorithm_name == 'cof':
    #     from detectors.COF import COFDetector
    #     return COFDetector(customParameters=config.customParameters['cof'], config=config)
    # elif algorithm_name == 'lof':
    #     from detectors.LOF import LOFDetector
    #     return LOFDetector(customParameters=config.customParameters['lof'], config=config)
    elif algorithm_name == 'auto_encoder':
        return AutoEncoderDetector(customParameters=config.mts_current_custom_parameters, config=config)
    elif algorithm_name == 'denoising_auto_encoder':
        return DenoisingAutoEncoderDetector(customParameters=config.mts_current_custom_parameters, config=config)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

@hydra.main(config_path="conf", config_name="config.yaml")
def main(config: DictConfig):
    print(config)

    (train_data, train_labels), (test_filenames, test_data_list, test_labels_list, test_multivariate_labels_list,
                                 test_contamination_list) = load_data_from_json(config)

    detector = init_detector(config)

    # common_metrics = [
    #     PrAUC(),
    #     FFVUS(slope=64),
    # ]
    # interpretability_metrics = [InterpretabilityHitKScore(top_k=k) for k in range(1, num_dimensions+1)]
    # interpretability_metrics_2 = [InterpretabilityConditionalHitKScore(top_k=k) for k in range(1, num_dimensions+1)]
    # interpretability_metrics = interpretability_metrics_1 + interpretability_metrics_2

    train_main_time = 0.0
    if config.executionType == "train" or config.executionType == "all":
         train_main_time = detector.train(train_data, train_labels)
    if config.executionType == "execute" or config.executionType == "all":
        test_dict = {
            'test_filenames': test_filenames,
            'test_data_list': test_data_list,
            'test_labels_list': test_labels_list,
            'test_multivariate_labels_list': test_multivariate_labels_list,
            'test_contamination_list': test_contamination_list
        }

        result_df = []
        result_dict = detector.predict(test_dict)
        for test_filename, result in (pbar:= tqdm(result_dict.items(), desc="Processing test results",
                                                  leave=True, position=0,
                                                  total=len(result_dict.keys()))):
            pbar.set_postfix({"current_test_file": test_filename})

            test_filepath = os.path.join(detector.absolute_dataOutputDir, test_filename)
            os.makedirs(test_filepath, exist_ok=True)
            pd.DataFrame([result['execute_main_time']], columns=['execute_main_time']).to_csv(
                os.path.join(test_filepath, 'docker-algorithm-execute-time.csv'), index=False)
            result['anomaly_scores'] = MinMaxScaler().fit_transform(result['scores'])[:,0]
            np.savetxt(os.path.join(test_filepath, f'docker-algorithm-scores.csv'), result['scores'], delimiter=',')
            np.savetxt(os.path.join(test_filepath, f'anomaly-scores.csv'), result['anomaly_scores'], delimiter=',')
            pd.DataFrame(result['dimension_contribution']).to_csv(
                os.path.join(test_filepath, 'docker-algorithm-dimension-contribution.csv'),
                index=False,
                header=None)
            pd.DataFrame(result['scores_per_var']).to_csv(
                os.path.join(test_filepath, f'docker-algorithm-scores-per-var.csv'),
                index=False,
                header=None)

            metric_dict = {}
            metric_dict['test_batch_id'] = test_filename
            metric_dict['train_main_time'] = train_main_time
            metric_dict['execute_main_time'] = result['execute_main_time']
            # for metric in common_metrics:
            #     metric_value = metric.score(test_labels_list[test_filenames.index(test_filename)], result['anomaly_scores'])
            #     metric_dict[metric.name] = metric_value
            # for metric in interpretability_metrics:
            #     metric_value = metric.score(test_multivariate_labels_list[test_filenames.index(test_filename)], result['dimension_contribution'])
            #     metric_dict[metric.name] = metric_value
            # result_df.append(metric_dict)
        result_df = pd.DataFrame(result_df)
        result_df.insert(0, 'dataset', config.mts_running_dataset)
        result_df.insert(0, 'collection', 'custom')
        result_df.insert(0, 'algorithm', config.name_mapping[config.mts_running_detector])
        result_df.to_csv(os.path.join(detector.absolute_dataOutputDir, 'results.csv'), index=False)


        # config.update({'mts_running_detector': 'auto_encoder'})
        # customParameters = config.customParameters['auto_encoder']
        # set_random_state(customParameters.random_state)
        #
        # # multivariate_label_df = load_multivariate_labels(config)
        # absolute_dataOutput = os.path.join(project_root_dir, config.dataOutput)
        # absolute_dataInput = os.path.join(project_root_dir, config.dataInput)
        # os.makedirs(absolute_dataOutput, exist_ok=True)
        # os.makedirs(absolute_dataInput, exist_ok=True)

    # algorithm_name = config.algorithm.name.lower()
    # algorithm_config = AlgorithmArgs(**config.algorithm.parameters)
    #
    # if algorithm_name == 'hbos':
    #     hbos_main(algorithm_config)
    # elif algorithm_name == 'cof':
    #     cof_main(algorithm_config)
    # elif algorithm_name == 'lof':
    #     lof_main(algorithm_config)
    # else:
    #     raise ValueError(f"Unsupported algorithm: {algorithm_name}")
if __name__ == '__main__':
    main()