# tests/test_comparison.py

# type: ignore
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from knnopt.core.comparison import ComparisonSuite
from knnopt.core.utils import ScalerType

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def comparison_data():
    """Provides a small binary classification dataset for benchmarking."""
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_classes=2,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(4)])
    df['target'] = y
    return df

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def test_compare_knn_methods_success(comparison_data):
    """Test that compare_knn_methods runs all stages and returns a valid summary DataFrame."""
    suite = ComparisonSuite(scaler_method=ScalerType.STANDARD)

    results_df = suite.compare_knn_methods(
        df=comparison_data,
        target_name='target',
        cv_folds=2,
        k_max=5
    )

    assert isinstance(results_df, pd.DataFrame)

    expected_methods = [
        'Simple Elbow (Train/Test)',
        'CV Elbow (k-Fold)',
        'GridSearchCV (K & Weights)'
    ]
    assert all(method in results_df.index for method in expected_methods)

    assert 'Optimal K' in results_df.columns
    assert 'Primary Score' in results_df.columns

    assert not results_df['Optimal K'].isnull().any()
    assert all(k >= 1 for k in results_df['Optimal K'])
