# tests/test_optimization.py
# type: ignore

import pytest
import pandas as pd
import numpy as np
import numbers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from knnopt.core.optimization import OptimizationSuite
from knnopt.core.utils import ScalerType

@pytest.fixture
def optimization_data():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(4)])
    df['target'] = y
    return df

def test_find_sample_k_elbow_method(optimization_data, caplog):
    """Test the single-split Elbow method returns optimal K and a fitted KNN model."""
    suite = OptimizationSuite(scaler_method=ScalerType.MINMAX)

    with caplog.at_level('INFO'):
        optimal_k, final_knn = suite.find_sample_k_elbow_method(
            df=optimization_data,
            target_name='target',
            k_max=5
        )

    assert isinstance(optimal_k, numbers.Integral)
    assert isinstance(final_knn, KNeighborsClassifier)
    assert 1 <= optimal_k <= 5
    assert final_knn.n_neighbors == optimal_k

    # Check runtime is positive float instead of substring
    runtime = suite.results['elbow_sample']['runtime']
    assert isinstance(runtime, (float, np.floating))
    assert runtime > 0

    # Check log messages
    assert "Method 'Elbow Sample' completed" in caplog.text

def test_find_k_elbow_method_cv(optimization_data, caplog):
    """Test the CV Elbow method returns optimal K and a fitted KNN model."""
    suite = OptimizationSuite(scaler_method=ScalerType.STANDARD)

    with caplog.at_level('INFO'):
        optimal_k, model = suite.find_k_elbow_method_cv(
            df=optimization_data,
            target_name='target',
            cv_folds=3,
            k_max=5
        )

    assert isinstance(optimal_k, numbers.Integral)
    assert isinstance(model, KNeighborsClassifier)
    assert 1 <= optimal_k <= 5
    assert model.n_neighbors == optimal_k

    # Check runtime is positive float instead of substring
    runtime = suite.results['elbow_cv']['runtime']
    assert isinstance(runtime, (float, np.floating))
    assert runtime > 0

    # Check log messages
    assert "Method 'Elbow CV' completed" in caplog.text

def test_find_k_elbow_grid_search(optimization_data, caplog):
    """Test Grid Search returns optimal K and a fitted KNN model."""
    suite = OptimizationSuite(scaler_method=ScalerType.ROBUST)

    with caplog.at_level('INFO'):
        optimal_k, model = suite.find_k_elbow_grid_search(
            df=optimization_data,
            target_name='target',
            cv_folds=2,
            k_max=5
        )
    
    assert isinstance(optimal_k, numbers.Integral)
    assert isinstance(model, KNeighborsClassifier)
    assert 1 <= optimal_k <= 5
    assert model.n_neighbors == optimal_k
    
    # Check runtime is positive float instead of substring
    runtime = suite.results['grid_search']['runtime']
    assert isinstance(runtime, (float, np.floating))
    assert runtime > 0

    # Check log messages
    assert "Method 'Grid Search' completed" in caplog.text

