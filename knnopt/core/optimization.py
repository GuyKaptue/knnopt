# knnopt/core/optimization.py
# type: ignore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import logging
from typing import Optional, Dict, Any, Tuple

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from tqdm import tqdm

from .preprocessing import KNNPreprocessor, DataPreparationError
from .utils import get_scaler, log_runtime, ScalerType

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ------------------------------------------------------------------------------
# Optimization Suite
# ------------------------------------------------------------------------------

class OptimizationSuite:
    """
    Provides multiple strategies for selecting the optimal K value in KNN classification.
    Includes sample-based elbow method, cross-validation elbow method, and grid search.
    """
    MAX_K_HARD_LIMIT = 500

    def __init__(self, scaler_method: ScalerType = ScalerType.STANDARD):
        self.scaler_method = scaler_method
        self.preprocessor = KNNPreprocessor(scaler_method=scaler_method)
        self.results: Dict[str, Any] = {}
        logger.info(f"OptimizationSuite initialized with scaler: '{scaler_method.value}'.")

    def _determine_k_max(self, n_samples: int, k_max: Optional[int]) -> int:
        """
        Determines the maximum K value to test, using sqrt heuristic and a hard cap.
        """
        if k_max is None:
            calculated = int(math.sqrt(n_samples))
            capped = min(calculated, self.MAX_K_HARD_LIMIT)
            k_max = max(5, capped)
            logger.info(f"k_max auto-set to {k_max} (sqrt heuristic, capped at {self.MAX_K_HARD_LIMIT})")
        else:
            k_max = max(5, k_max)
        return k_max

    def find_sample_k_elbow_method(
        self,
        df: pd.DataFrame,
        target_name: str,
        test_size: float = 0.3,
        random_state: int = 42,
        k_max: Optional[int] = None
    ) -> Tuple[int, KNeighborsClassifier]:
        """
        Uses a simple train/test split to find the optimal K using the elbow method.
        """
        start = time.time()
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_data(df, target_name, test_size, random_state)
        k_max = self._determine_k_max(len(X_train), k_max)
        error_rates = []

        for k in range(1, k_max + 1):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            error_rates.append(1 - accuracy_score(y_test, y_pred))

        optimal_k = np.argmin(error_rates) + 1
        min_error = error_rates[optimal_k - 1]

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, k_max + 1), error_rates, marker='o', linestyle='--', color='blue')
        plt.axvline(optimal_k, color='red', linestyle=':', linewidth=1)
        plt.plot(optimal_k, min_error, marker='o', color='red', markersize=10, label=f"Optimal K = {optimal_k}")
        plt.title("Elbow Method: Error Rate (Test Set)")
        plt.xlabel("K Value")
        plt.ylabel("Error Rate")
        plt.grid(True)
        plt.legend()
        plt.show()

        final_model = KNeighborsClassifier(n_neighbors=optimal_k)
        final_model.fit(X_train, y_train)
        final_accuracy = final_model.score(X_test, y_test)

        self.results['elbow_sample'] = {
            'k': optimal_k,
            'accuracy': final_accuracy,
            'model': final_model,
            'runtime': log_runtime(start, "Elbow Sample")
        }

        return optimal_k, final_model

    def find_k_elbow_method_cv(
        self,
        df: pd.DataFrame,
        target_name: str,
        cv_folds: int = 5,
        k_max: Optional[int] = None
    ) -> Tuple[int, KNeighborsClassifier]:
        """
        Uses cross-validation to find the optimal K using the elbow method.
        """
        start = time.time()
        X, y = self.preprocessor._validate_features(df, target_name)
        k_max = self._determine_k_max(len(X), k_max)
        error_rates = []

        scaler = get_scaler(self.scaler_method.value)
        pipeline = make_pipeline(scaler, KNeighborsClassifier()) if scaler else KNeighborsClassifier()

        for k in range(1, k_max + 1):
            if isinstance(pipeline, KNeighborsClassifier):
                pipeline.set_params(n_neighbors=k)
            else:
                pipeline.set_params(kneighborsclassifier__n_neighbors=k)

            scores = cross_val_score(pipeline, X.values, y, cv=cv_folds, scoring='accuracy', n_jobs=-1)
            error_rates.append(1 - scores.mean())

        optimal_k = np.argmin(error_rates) + 1
        min_error = error_rates[optimal_k - 1]

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, k_max + 1), error_rates, marker='o', linestyle='--', color='blue')
        plt.axvline(optimal_k, color='red', linestyle=':', linewidth=1)
        plt.plot(optimal_k, min_error, marker='o', color='red', markersize=10, label=f"Optimal K = {optimal_k}")
        plt.title("Elbow Method: Mean Error Rate (CV)")
        plt.xlabel("K Value")
        plt.ylabel("Mean Error Rate")
        plt.grid(True)
        plt.legend()
        plt.show()

        final_model = KNeighborsClassifier(n_neighbors=optimal_k)
        final_model.fit(X.values, y)

        self.results['elbow_cv'] = {
            'k': optimal_k,
            'error_rate': min_error,
            'model': final_model,
            'runtime': log_runtime(start, "Elbow CV")
        }

        return optimal_k, final_model

    def find_k_elbow_grid_search(
        self,
        df: pd.DataFrame,
        target_name: str,
        cv_folds: int = 5,
        test_size: float = 0.3,
        random_state: int = 42,
        k_max: Optional[int] = None
    ) -> Tuple[int, KNeighborsClassifier]:
        """
        Uses GridSearchCV to find the optimal K and weighting strategy.
        """
        start = time.time()
        X_train, _, y_train, _ = self.preprocessor.prepare_data(df, target_name, test_size, random_state)
        k_max = self._determine_k_max(len(X_train), k_max)
        k_range = list(range(1, k_max + 1))

        param_grid = {
            'n_neighbors': k_range,
            'weights': ['uniform', 'distance']
        }

        for _ in tqdm(ParameterGrid(param_grid), desc="GridSearch Progress", unit="combination"):
            pass  # Just for progress bar visualization

        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            return_train_score=False
        )

        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        optimal_k = best_params['n_neighbors']
        final_model = grid_search.best_estimator_

        results_df = pd.DataFrame(grid_search.cv_results_)
        avg_scores = results_df.groupby('param_n_neighbors')['mean_test_score'].mean()
        error_rates = 1 - avg_scores.values

        plt.figure(figsize=(10, 5))
        plt.plot(avg_scores.index, error_rates, marker='o', linestyle='--', color='blue')
        plt.axvline(optimal_k, color='red', linestyle=':', linewidth=1)
        plt.plot(optimal_k, 1 - best_score, marker='o', color='red', markersize=10, label=f"Optimal K = {optimal_k}")
        plt.title("Elbow Method: Mean Error Rate (GridSearchCV)")
        plt.xlabel("K Value")
        plt.ylabel("Mean Error Rate")
        plt.grid(True)
        plt.legend()
        plt.show()

        self.results['grid_search'] = {
            'k': optimal_k,
            'accuracy': best_score,
            'weights': best_params['weights'],
            'model': final_model,
            'runtime': log_runtime(start, "Grid Search")
        }

        return optimal_k, final_model

    def get_results(self) -> Dict[str, Any]:
        """
        Returns all stored optimization results from previous runs.
        """
        return self.results
