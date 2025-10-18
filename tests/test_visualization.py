# tests/test_visualization.py

# type: ignore
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from knnopt.core.visualization import VisualizationSuite
from knnopt.core.utils import ScalerType

@pytest.fixture
def visualization_data():
    # Keep n_samples small for fast processing
    X, y = make_classification(
        n_samples=50,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(3)])
    df['target'] = y
    return df

@pytest.fixture
def comparison_results_df():
    """Mock DataFrame simulating the output of ComparisonSuite."""
    data = {
        'Optimal K': [3, 5, 7],
        'Primary Score': ['0.8500', '0.8650', '0.8700'],
        'Weights': ['Uniform', 'Uniform (Implicit)', 'Distance']
    }
    index = ['Simple Elbow (Train/Test)', 'CV Elbow (k-Fold)', 'GridSearchCV (K & Weights)']
    return pd.DataFrame(data, index=index)

def test_plot_optimal_k_clusters_runs(visualization_data, comparison_results_df, monkeypatch, caplog):
    """Test that the plotting function runs without raising exceptions and calls plt.show."""
    suite = VisualizationSuite(scaler_method=ScalerType.STANDARD)

    # Mock plt.show() to prevent the test from blocking
    def mock_show():
        pass

    monkeypatch.setattr(plt, 'show', mock_show)

    with caplog.at_level('INFO'):
        suite.plot_optimal_k_clusters(
            df=visualization_data,
            target_name='target',
            comparison_results_df=comparison_results_df
        )

    assert "Preparing visualization for methods" in caplog.text
    assert plt.gcf().axes is not None
    assert len(plt.gcf().axes) == 3

    plt.close('all')

def test_plot_error_missing_optimal_k(visualization_data, comparison_results_df):
    """Test error handling when the required column is missing."""
    df_bad = comparison_results_df.drop(columns=['Optimal K'])
    suite = VisualizationSuite()

    with pytest.raises(ValueError, match="'Optimal K' column"):
        suite.plot_optimal_k_clusters(
            df=visualization_data,
            target_name='target',
            comparison_results_df=df_bad
        )
