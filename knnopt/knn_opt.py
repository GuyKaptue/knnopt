# knnopt/knnopt/knn_opt.py


# type: ignore
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging

# Import core components
from knnopt.core.comparison import ComparisonSuite
from knnopt.core.visualization import VisualizationSuite
from knnopt.core.preprocessing import DataPreparationError
from knnopt.core.utils import ScalerType

# ----------------------------
# Logging Configuration
# ----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ----------------------------
# KNNOpt Interface
# ----------------------------

class KNNOpt:
    """
    Public interface for the KNN Benchmarking and Visualization Toolkit.
    Orchestrates optimization, comparison, and visualization of optimal K values.
    """

    def __init__(self, scaler_method: ScalerType = ScalerType.STANDARD):
        """
        Initializes the KNNOpt suite with the desired scaling method.

        Args:
            scaler_method (ScalerType): The scaling method to apply across all stages.
        """
        self.scaler_method = scaler_method
        self.comparison_suite = ComparisonSuite(scaler_method=scaler_method)
        self.visualization_suite = VisualizationSuite(scaler_method=scaler_method)
        self.comparison_results: Optional[pd.DataFrame] = None

        logger.info(f"KNNOpt initialized with scaler: '{scaler_method.value}'.")

    def run(
        self,
        df: pd.DataFrame,
        target_name: str,
        test_size: float = 0.3,
        random_state: int = 42,
        cv_folds: int = 5,
        k_max: Optional[int] = None,
        plot_results: bool = True
    ) -> pd.DataFrame:
        """
        Executes the full K-NN optimization and benchmarking workflow.

        Args:
            df (pd.DataFrame): Raw DataFrame with features and target.
            target_name (str): Name of the target column.
            test_size (float): Proportion for test split.
            random_state (int): Seed for reproducibility.
            cv_folds (int): Number of CV folds.
            k_max (Optional[int]): Max K to test (default: sqrt(N)).
            plot_results (bool): If True, generates cluster visualizations.

        Returns:
            pd.DataFrame: Summary table of benchmark results.

        Raises:
            DataPreparationError: If input data fails validation.
        """
        logger.info("--- Starting Full K-NN Optimization Benchmark ---")

        try:
            self.comparison_results = self.comparison_suite.compare_knn_methods(
                df=df,
                target_name=target_name,
                test_size=test_size,
                random_state=random_state,
                cv_folds=cv_folds,
                k_max=k_max
            )

            if plot_results and self.comparison_results is not None:
                logger.info("--- Generating K-NN Cluster Visualization ---")
                self.visualization_suite.plot_optimal_k_clusters(
                    df=df,
                    target_name=target_name,
                    comparison_results_df=self.comparison_results
                )

            logger.info("--- Benchmark Complete ---")
            return self.comparison_results

        except DataPreparationError as e:
            logger.critical(f"Data validation failed: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error during benchmarking: {e}")
            raise

    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Retrieves the last run comparison results.

        Returns:
            Optional[pd.DataFrame]: Benchmark summary or None if not run.
        """
        if self.comparison_results is None:
            logger.warning("No benchmark has been run yet. Call run() first.")
        return self.comparison_results
