# knnopt/knnopt/core/visualization.py
# type: ignore

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Optional, Tuple

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

from .utils import get_scaler, ScalerType
from .preprocessing import KNNPreprocessor, DataPreparationError

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
# Visualization Suite
# ----------------------------

class VisualizationSuite:
    """
    Visualizes K-NN decision boundaries using optimal K values from benchmarking results.
    """

    def __init__(self, scaler_method: ScalerType = ScalerType.STANDARD):
        self.scaler_method = scaler_method
        self.preprocessor = KNNPreprocessor(scaler_method=scaler_method)
        logger.info(f"VisualizationSuite initialized with scaler: '{scaler_method.value}'.")

    def plot_optimal_k_clusters(
        self,
        df: pd.DataFrame,
        target_name: str,
        comparison_results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plots decision boundaries for each optimal K value using PCA-reduced data.

        Args:
            df (pd.DataFrame): Dataset with features and target.
            target_name (str): Name of the target column.
            comparison_results_df (pd.DataFrame): Output from ComparisonSuite.compare_knn_methods().
            save_path (Optional[str]): Path to save the figure. If None, only displays.
        """
        import time
        start_time = time.time()

        if 'Optimal K' not in comparison_results_df.columns:
            raise ValueError("Comparison results must contain an 'Optimal K' column.")

        k_values: List[int] = comparison_results_df['Optimal K'].dropna().astype(int).tolist()
        method_names: List[str] = comparison_results_df['Optimal K'].dropna().index.tolist()

        if not k_values:
            logger.warning("No optimal K values found to visualize.")
            return

        logger.info(f"Preparing visualization for methods: {list(zip(method_names, k_values))}")

        try:
            X, y = self.preprocessor._validate_features(df, target_name)
        except DataPreparationError as e:
            logger.error(f"Data validation failed: {e}")
            raise e

        scaler = get_scaler(self.scaler_method.value)
        X_scaled = scaler.fit_transform(X.values) if scaler else X.values

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained_variance = sum(pca.explained_variance_ratio_)

        y_labels = y.astype('category').cat.codes.values
        n_classes = len(np.unique(y_labels))

        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))

        #  Modern Matplotlib API (no deprecation warnings)
        cmap = plt.colormaps['coolwarm']
        cmap_light = ListedColormap(cmap(np.linspace(0.2, 0.8, n_classes)))
        cmap_bold = ListedColormap(cmap(np.linspace(0.8, 0.2, n_classes)))
        
        num_plots = len(k_values)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6))

        if num_plots == 1:
            axes = [axes]

        for i, (k, name) in enumerate(zip(k_values, method_names)):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_pca, y_labels)

            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            axes[i].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
            axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=y_labels, cmap=cmap_bold, edgecolors='k', s=30)

            axes[i].set_title(f"{name} (K={k})")
            axes[i].set_xlabel("PCA Component 1")
            axes[i].set_ylabel("PCA Component 2")
            axes[i].set_xlim(xx.min(), xx.max())
            axes[i].set_ylim(yy.min(), yy.max())
            axes[i].set_aspect('equal', adjustable='box')

        plt.suptitle(f"Visualisation des Décisions K-NN (PCA Variance: {explained_variance:.2f})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.9])

        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()

        logger.info(f"Visualization completed in {time.time() - start_time:.2f} seconds.")
