import warnings
import joblib

import numpy as np

from typing import Optional, Tuple
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import check_paired_arrays
from sklearn.utils.extmath import row_norms
from sklearn.pipeline import make_pipeline
import sklearn

from numpy.lib.stride_tricks import sliding_window_view


class SlidingWindowProcessor(BaseEstimator, TransformerMixin):

    def __init__(self, window_size: int, standardize: bool = False):
        self.window_size = window_size
        if standardize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> 'SlidingWindowProcessor':
        if self.scaler:
            self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        y is unused (exists for compatibility)
        """
        if self.scaler:
            print("Standardizing input data")
            X = self.scaler.transform(X)
        # the last window would have no target to predict, e.g. for n=10: [[1, 2] -> 3, ..., [8, 9] -> 10, [9, 10] -> ?]
        new_X = sliding_window_view(X, window_shape=self.window_size, axis=0)[:-1]
        # reshape to two dimensions (required by regressors)
        new_X = new_X.reshape(new_X.shape[0], -1)
        new_y = np.roll(X, -self.window_size, axis=0)[:-self.window_size]
        return new_X, new_y

    def transform_y(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            print("Standardizing input data")
            X = self.scaler.transform(X)
        return np.roll(X, -self.window_size, axis=0)[:-self.window_size]

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        result = np.full(shape=(self.window_size+y.shape[0], y.shape[1]), fill_value=np.nan)
        result[-len(y):, :] = y
        if self.scaler:
            print("Reversing standardization for prediction")
            result = self.scaler.inverse_transform(result)
        return result


class RandomBlackForestAnomalyDetector(BaseEstimator, RegressorMixin):
    def __init__(self,
            train_window_size: int = 50,
            n_estimators: int = 2,
            max_features_per_estimator: float = 0.5,
            n_trees: float = 100,
            max_features_method: str = "auto",  # "sqrt", "log2"
            bootstrap: bool = True,
            max_samples: Optional[float] = None,  # fraction of all samples
            standardize: bool = False,
            random_state: int = 42,
            verbose: int = 0,
            n_jobs: int = 1,
            # the following parameters control the tree size
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            retrain= False,
            ):
        self.preprocessor = SlidingWindowProcessor(train_window_size, standardize)
        self.clf = BaggingRegressor(
            estimator=RandomForestRegressor(
                n_estimators=n_trees,
                max_features=max_features_method,
                bootstrap=bootstrap,
                max_samples=max_samples,
                random_state=random_state,
                verbose=verbose,
                n_jobs=n_jobs,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            ),
            n_estimators=n_estimators,
            max_features=max_features_per_estimator,
            bootstrap_features=False, # draw features without replacement
            max_samples=1.0, # all samples for every base estimator
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'RandomBlackForestAnomalyDetector':
        if y is not None:
            warnings.warn(f"y is calculated from X. Please don't pass y to RandomBlackForestAnomalyDetector.fit, it will be ignored!")
        X, y = self.preprocessor.fit_transform(X)
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X, _ = self.preprocessor.transform(X)
        y_hat = self._predict_internal(X)
        return self.preprocessor.inverse_transform_y(y_hat)
    
    def euclidean_distances_per_var(self, X, Y):
        '''
        This function is based on sklearn.metrics.pairwise.paired_distances.
        The purpose is to return distances for all variables of the multivariate time series.
        '''
        X, Y = check_paired_arrays(X, Y)
        diff = X - Y
        diff_per_var = np.multiply(diff, diff)
        return diff_per_var


    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        overall_scores_shape = X.shape[0]
        scores_per_var_shape = X.shape
        X, y = self.preprocessor.transform(X)
        y_hat = self._predict_internal(X)
        overall_scores = sklearn.metrics.pairwise.paired_distances(y, y_hat.reshape(y.shape))
        scores_per_var = self.euclidean_distances_per_var(y, y_hat.reshape(y.shape))
        overall_anomaly_scores = np.full(shape=overall_scores_shape, fill_value=np.nan)
        overall_anomaly_scores[-len(overall_scores):] = overall_scores
        anomaly_scores_per_var = np.full(shape=scores_per_var_shape, fill_value=np.nan)
        anomaly_scores_per_var[-len(scores_per_var):] = scores_per_var
        return overall_anomaly_scores, anomaly_scores_per_var
        
    def interpret(self, obs: np.ndarray) -> list:
        assert len(obs)==1, "Please ensure that only one observation is input to the function"
        
        

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> 'RandomBlackForestAnomalyDetector':
        return joblib.load(path)
