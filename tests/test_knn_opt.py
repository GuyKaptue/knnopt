# tests/test_knn_opt.py
# type: ignore

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from knnopt.knn_opt import KNNOpt
from knnopt.core.preprocessing import DataPreparationError
from knnopt.core.utils import ScalerType

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def end_to_end_data():
    """Provides a small, clean dataset suitable for a full benchmark run."""
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target_class'] = y
    return df

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def test_knnopt_run_success(end_to_end_data, monkeypatch):
    """Test the full run() method executes successfully and returns a summary DataFrame."""
    monkeypatch.setattr(plt, 'show', lambda: None)

    optimizer = KNNOpt(scaler_method=ScalerType.STANDARD)
    summary_df = optimizer.run(
        df=end_to_end_data,
        target_name='target_class',
        test_size=0.3,
        cv_folds=3,
        k_max=5,
        plot_results=True
    )

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 3
    assert 'Optimal K' in summary_df.columns
    assert 'Primary Score' in summary_df.columns
    assert summary_df['Optimal K'].notnull().all()
    assert optimizer.get_results() is summary_df
    assert len(plt.get_fignums()) >= 4

    plt.close('all')

def test_knnopt_run_data_error(end_to_end_data):
    """Test that KNNOpt correctly catches and re-raises DataPreparationError."""
    import numpy as np
    df_with_nan = end_to_end_data.copy()
    df_with_nan.loc[0, 'feature_1'] = np.nan

    optimizer = KNNOpt(scaler_method=ScalerType.MINMAX)

    # Match updated error message from preprocessing
    with pytest.raises(DataPreparationError, match=r"Missing values in features"):
        optimizer.run(df=df_with_nan, target_name='target_class')

def test_get_results_before_run():
    """Test get_results returns None if run() hasn't been called."""
    optimizer = KNNOpt()
    assert optimizer.get_results() is None

def test_knnopt_run_no_plot(end_to_end_data):
    """Test the run method when plot_results=False."""
    optimizer = KNNOpt(scaler_method=ScalerType.STANDARD)

    summary_df = optimizer.run(
        df=end_to_end_data,
        target_name='target_class',
        plot_results=False,
        cv_folds=2,
        k_max=3
    )

    assert summary_df is not None
    assert isinstance(summary_df, pd.DataFrame)
    assert len(plt.get_fignums()) == 0
