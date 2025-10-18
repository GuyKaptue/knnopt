# knnopt/knnopt/core/preprocessing.py
# type: ignore

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

from .utils import get_scaler, ScalerType

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
# Custom Exception
# ----------------------------

class DataPreparationError(Exception):
    """Raised when input data fails validation checks for KNN preprocessing."""
    pass

# ----------------------------
# KNN Preprocessor Class
# ----------------------------

class KNNPreprocessor:
    """
    Handles data validation, splitting, and scaling for K-NN benchmarking.
    Ensures clean, numerical input and consistent preprocessing.
    """

    def __init__(self, scaler_method: ScalerType = ScalerType.STANDARD):
        self.scaler_method = scaler_method
        self.scaler = get_scaler(scaler_method.value)
        self._is_fitted = False
        logger.info(f"Initialized KNNPreprocessor with scaler: '{scaler_method.value}'.")

    def _validate_features(self, df: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        if target_name not in df.columns:
            raise DataPreparationError(f"Target column '{target_name}' not found.")

        y = df[target_name].copy()
        X = df.drop(columns=[target_name]).copy()

        if X.isnull().any().any():
            missing = X.columns[X.isnull().any()].tolist()
            raise DataPreparationError(f"Missing values in features: {missing}")

        if y.isnull().any():
            raise DataPreparationError("Missing values in target column.")

        non_numeric = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if non_numeric:
            raise DataPreparationError(f"Non-numeric features detected: {non_numeric}")

        logger.info(f"Validation passed. Features shape: {X.shape}")
        return X, y

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_name: str,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Validates, splits, and scales the dataset.

        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test
        """
        X, y = self._validate_features(df, target_name)

        stratify = y if y.nunique() > 1 and len(y.unique()) < len(y) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        logger.info(f"Split complete: Train={len(X_train)}, Test={len(X_test)}")

        if self.scaler:
            X_train_scaled = self.scaler.fit_transform(X_train.values)
            X_test_scaled = self.scaler.transform(X_test.values)
            self._is_fitted = True
            logger.info(f"Scaling applied using '{self.scaler_method.value}' scaler.")
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
            logger.warning("No scaling applied.")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def transform(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Transforms new data using the fitted scaler.

        Returns:
            Scaled numpy array or None if validation fails.
        """
        if self.scaler is None:
            logger.warning("No scaler initialized. Returning raw data.")
            return X.values

        if not self._is_fitted:
            logger.error("Scaler not fitted. Run 'prepare_data' first.")
            return None

        if X.isnull().any().any():
            logger.error("Missing values detected in new data.")
            return None

        non_numeric = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if non_numeric:
            logger.error(f"Non-numeric columns in new data: {non_numeric}")
            return None

        return self.scaler.transform(X.values)
