import itertools
import time
from typing import Optional, Dict

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import ndcg_score
from sklearn.metrics._ranking import _ndcg_sample_scores

from metrics.ffvus.vus_torch import VUSTorch
from utils import estimate_dimension_contribution_by_anomaly_score_with_a_buffer


class IndependentNDCG:
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """

    def __init__(self,
                 max_k,
                 max_smoothing_window,
                 k_slide,
                 smoothing_window_slide,
                 fix_k,
                 fix_smoothing_window,
                 default_selected_k,
                 default_selected_smoothing_window,
                 **kwargs
                 ) -> None:
        self.max_k: Optional[int] = max_k
        self.max_smoothing_window = max_smoothing_window
        self.k_slide = k_slide
        self.smoothing_window_slide = smoothing_window_slide
        self.fix_k = fix_k
        self.fix_smoothing_window = fix_smoothing_window
        self.default_selected_k = default_selected_k
        self.default_selected_smoothing_window = default_selected_smoothing_window

    @property
    def name(self) -> str:
        return f'IndepNDCG_k_{self.max_k}_w_{self.max_smoothing_window}_Score'.upper()

    def get_name_template(self) -> str:
        return 'IndepNDCG_k_{k}_w_{w}_Score'.upper()

    def score_for_different_k(self, y_score, y_true_multivariate: np.ndarray, dimension_contribution: np.ndarray) -> Dict:
        assert y_true_multivariate.ndim == 2
        assert dimension_contribution.ndim == 2
        assert y_score.ndim == 1

        results_dict = dict() # Save results for all k

        y_true = (y_true_multivariate.sum(axis=1) >= 1).astype(float)

        k_list = []
        if self.fix_k:
            k_list = [self.default_selected_k]
        else:
            k_list = list(range(1, self.max_k + 1, self.k_slide))

        smoothing_window_list = []
        if self.fix_smoothing_window:
            smoothing_window_list = [self.default_selected_smoothing_window]
        else:
            smoothing_window_list = list(range(0, self.max_smoothing_window + 1, self.smoothing_window_slide))

        parameter_combinations = list(itertools.product(k_list, smoothing_window_list))
        for (k, smoothing_window) in parameter_combinations:
            start_time = time.time()
            # pbar.set_postfix({"k": k, "smoothing_window": smoothing_window})
            # y_score_per_var[np.isnan(y_score_per_var)] = 0.0
            # y_score_per_var = convert_raw_anomaly_score_per_var_to_contribution_percentage(y_score_per_var)
            estimated_dimension_contribution = estimate_dimension_contribution_by_anomaly_score_with_a_buffer(y_score, dimension_contribution, smoothing_window=smoothing_window)
            estimated_dimension_contribution = softmax(estimated_dimension_contribution, axis=1)


            interpretability_score = ndcg_score(y_true_multivariate[y_true == 1.0], estimated_dimension_contribution[y_true == 1.0], k=k)
            end_time = time.time()
            computation_time = end_time - start_time
            results_dict[(k, smoothing_window)] = dict()
            results_dict[(k, smoothing_window)]['value'] = interpretability_score
            results_dict[(k, smoothing_window)]['computation_time'] = computation_time

        return results_dict


class AVUSI:
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """

    def __init__(self,
                 max_k,
                 max_smoothing_window,
                 max_n_interpretability_sensitivity_levels,
                 k_slide,
                 smoothing_window_slide,
                 n_interpretability_sensitivity_levels_slide,
                 fix_k,
                 fix_smoothing_window,
                 fix_n_interpretability_sensitivity_levels,
                 default_selected_k,
                 default_selected_smoothing_window,
                 default_selected_n_interpretability_sensitivity_levels,
                 vus_slope,
                 vus_n_thresholds,
                    **kwargs
                 ) -> None:
        self.max_k: Optional[int] = max_k
        self.max_smoothing_window = max_smoothing_window
        self.max_n_interpretability_sensitivity_levels = max_n_interpretability_sensitivity_levels
        self.k_slide = k_slide
        self.smoothing_window_slide = smoothing_window_slide
        self.n_interpretability_sensitivity_levels_slide = n_interpretability_sensitivity_levels_slide
        self.fix_k = fix_k
        self.fix_smoothing_window = fix_smoothing_window
        self.fix_n_interpretability_sensitivity_levels = fix_n_interpretability_sensitivity_levels
        self.default_selected_k = default_selected_k
        self.default_selected_smoothing_window = default_selected_smoothing_window
        self.default_selected_n_interpretability_sensitivity_levels = default_selected_n_interpretability_sensitivity_levels
        self.vus_slope = vus_slope
        self.vus_n_thresholds = vus_n_thresholds


    @property
    def name(self) -> str:
        return f'AVUSI_k_{self.max_k}_w_{self.max_smoothing_window}_m_{self.max_n_interpretability_sensitivity_levels}_Score'.upper()

    def get_name_template(self) -> str:
        return 'AVUSI_k_{k}_w_{w}_m_{m}_Score'.upper()

    def score_for_different_k(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, dimension_contribution_multivariate: np.ndarray) -> Dict:
        assert y_true_multivariate.ndim == 2
        assert dimension_contribution_multivariate.ndim == 2
        assert y_true_univariate.ndim == 1
        assert y_score_univariate.ndim == 1

        y_true = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
        assert (y_true == y_true_univariate).all()

        k_list = []
        if self.fix_k:
            k_list = [self.default_selected_k]
        else:
            k_list = list(range(1, self.max_k + 1, self.k_slide))

        smoothing_window_list = []
        if self.fix_smoothing_window:
            smoothing_window_list = [self.default_selected_smoothing_window]
        else:
            smoothing_window_list = list(range(0, self.max_smoothing_window + 1, self.smoothing_window_slide))

        interpretability_sensitivity_levels_list = []
        if self.fix_n_interpretability_sensitivity_levels:
            interpretability_sensitivity_levels_list = [self.default_selected_n_interpretability_sensitivity_levels]
        else:
            interpretability_sensitivity_levels_list = list(range(self.n_interpretability_sensitivity_levels_slide, self.max_n_interpretability_sensitivity_levels + 1, self.n_interpretability_sensitivity_levels_slide))


        parameter_combinations = list(itertools.product(k_list, smoothing_window_list, interpretability_sensitivity_levels_list))
        print(f'Number of parameter combinations to evaluate: {len(parameter_combinations)}')

        result_dict = dict() # Save results for all k
        for k, smoothing_window, n_interpretability_sensitivity_levels in parameter_combinations:
            start_time = time.time()
            estimated_dimension_contribution = estimate_dimension_contribution_by_anomaly_score_with_a_buffer(
                y_score=y_score_univariate,
                dimension_contribution=dimension_contribution_multivariate,
                smoothing_window=smoothing_window)
            estimated_dimension_contribution = softmax(estimated_dimension_contribution, axis=1)

            m_sensitivity_levels = np.linspace(0, 1, n_interpretability_sensitivity_levels)
            result_dict[f'n_interpretability_sensitivity_levels_{n_interpretability_sensitivity_levels}_M_list'] = m_sensitivity_levels

            interpretability_scores = _ndcg_sample_scores(y_true_multivariate, estimated_dimension_contribution, k=k)
            vus_pr_list = []
            print(f'Start calculate vusi list for k={k}, w={smoothing_window}, m={n_interpretability_sensitivity_levels}')
            for m in m_sensitivity_levels:
                interpretability_scores[y_true_univariate == 0] = m  # Set interpretability scores of normal samples to the minimum score
                new_anomaly_scores = y_score_univariate * interpretability_scores

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                vus = VUSTorch(slope_size=self.vus_slope, device=device)
                value_data, timing = vus.compute(torch.from_numpy(y_true).to(device), torch.from_numpy(np.round(new_anomaly_scores, decimals=2)).to(device))
                vus_pr_list.append(value_data[0].item())
            end_time = time.time()
            computation_time = end_time - start_time
            result_dict[(k,smoothing_window, n_interpretability_sensitivity_levels)] = dict()
            result_dict[(k,smoothing_window, n_interpretability_sensitivity_levels)]['value'] = np.mean(vus_pr_list)
            result_dict[(k,smoothing_window, n_interpretability_sensitivity_levels)]['vus_pr_list'] = vus_pr_list
            result_dict[(k, smoothing_window, n_interpretability_sensitivity_levels)]['computation_time'] = computation_time
        return result_dict