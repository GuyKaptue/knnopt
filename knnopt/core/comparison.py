# knnopt/knnopt/core/comparison.py
# type: ignore

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
import time

# Import the core optimization class
from .optimization import OptimizationSuite 
from .utils import log_runtime 
from .preprocessing import DataPreparationError # Import for cleaner error handling

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ComparisonSuite:
    """
    A class to orchestrate and compare the results of different K-NN optimal K 
    finding methods (Simple Elbow, CV Elbow, and Grid Search).
    """

    def __init__(self, scaler_method: str = 'standard'):
        """
        Initializes the comparison suite by creating an instance of the 
        OptimizationSuite, which holds all benchmarking methods.

        Args:
            scaler_method (str): The default scaling method to use for all tests.
        """
        # OptimizationSuite instance will perform the actual work
        self.optimization_suite = OptimizationSuite(scaler_method=scaler_method)
        self.comparison_results: Dict[str, Any] = {}
        logger.info(f"ComparisonSuite initialized, using {scaler_method} scaling.")

    def compare_knn_methods(
        self,
        df: pd.DataFrame,
        target_name: str,
        test_size: float = 0.3,
        random_state: int = 42,
        cv_folds: int = 5,
        k_max: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Runs and compares the three implemented K-NN optimization strategies.

        Args:
            df (pd.DataFrame): The raw DataFrame containing features and the target.
            target_name (str): The name of the target column.
            test_size (float): Proportion of the dataset for the test split (for simple methods).
            random_state (int): Seed for reproducibility.
            cv_folds (int): Number of cross-validation folds.
            k_max (Optional[int]): Maximum number of neighbors to test (defaults to sqrt(N)).

        Returns:
            pd.DataFrame: A formatted table summarizing the results of all methods.
        
        Raises:
            DataPreparationError: If input data fails validation checks.
        """
        full_start_time = time.time()
        
        try:
            # --- 1. Simple Elbow Method (Train/Test Split) ---
            logger.info("\n--- [1] Running Simple Elbow Method (Train/Test Split) ---")
            final_knn_simple = self.optimization_suite.find_sample_k_elbow_method(
                df=df, target_name=target_name, test_size=test_size, random_state=random_state, k_max=k_max
            )
            
            # Note: Results are now automatically stored in self.optimization_suite.results['elbow_sample']

            # --- 2. CV Elbow Method ---
            logger.info("\n--- [2] Running CV Elbow Method ---")
            self.optimization_suite.find_k_elbow_method_cv(
                df=df, target_name=target_name, cv_folds=cv_folds, k_max=k_max
            )
            # Note: Results are now automatically stored in self.optimization_suite.results['elbow_cv']

            # --- 3. GridSearchCV ---
            logger.info("\n--- [3] Running GridSearchCV ---")
            self.optimization_suite.find_k_elbow_grid_search(
                df=df, target_name=target_name, cv_folds=cv_folds, test_size=test_size, random_state=random_state, k_max=k_max
            )
            # Note: Results are now automatically stored in self.optimization_suite.results['grid_search']

            # --- 4. Consolidate and Format Results ---
            self.comparison_results = self._consolidate_results()

            logger.info("\n--- Summary Comparison Complete ---")
            log_runtime(full_start_time, "Full Comparison Suite")
            
            return self.comparison_results

        except DataPreparationError as e:
            logger.error(f"Error during data preparation: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred during comparison: {e}")
            raise e

    def _consolidate_results(self) -> pd.DataFrame:
        """Helper to compile and format the benchmark results into a DataFrame."""
        
        data = []
        
        # Simple Elbow Result
        res_simple = self.optimization_suite.results.get('elbow_sample', {})
        data.append({
            'Method': 'Simple Elbow (Train/Test)',
            'Optimal K': res_simple.get('k'),
            'Primary Score': res_simple.get('accuracy'), # Accuracy on Test Set
            'Weights': 'Uniform',
            'Runtime (s)': f"{res_simple.get('runtime', 0):.4f}"
        })
        
        # CV Elbow Result
        res_cv = self.optimization_suite.results.get('elbow_cv', {})
        data.append({
            'Method': 'CV Elbow (k-Fold)',
            'Optimal K': res_cv.get('k'),
            'Primary Score': 1 - res_cv.get('error_rate', np.nan), # Convert Error Rate to Accuracy
            'Weights': 'Uniform (Implicit)',
            'Runtime (s)': f"{res_cv.get('runtime', 0):.4f}"
        })

        # Grid Search Result
        res_grid = self.optimization_suite.results.get('grid_search', {})
        data.append({
            'Method': 'GridSearchCV (K & Weights)',
            'Optimal K': res_grid.get('k'),
            'Primary Score': res_grid.get('accuracy'), # Best CV Accuracy
            'Weights': res_grid.get('weights'),
            'Runtime (s)': f"{res_grid.get('runtime', 0):.4f}"
        })
        
        # Create and format DataFrame
        df_results = pd.DataFrame(data)
        df_results['Primary Score'] = df_results['Primary Score'].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else "N/A"
        )
        
        return df_results.set_index('Method')

    # Note: No need for a separate 'get_results' method, as the compare_knn_methods returns the DataFrame directly.