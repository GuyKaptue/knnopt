# tests/test_preprocessing.py
# type: ignore

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from knnopt.core.preprocessing import KNNPreprocessor, DataPreparationError
from knnopt.core.utils import ScalerType

# Fixture for a clean, numerical dataset
@pytest.fixture
def clean_data():
    X, y = make_classification(n_samples=50, n_features=5, n_classes=3, 
                               n_informative=3, n_redundant=0, random_state=42)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
    df['target'] = y
    return df

def test_prepare_data_success(clean_data):
    """Test successful data preparation with standard scaling and splitting."""
    preprocessor = KNNPreprocessor(scaler_method=ScalerType.STANDARD)
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        df=clean_data, 
        target_name='target', 
        test_size=0.4
    )
    
    assert len(X_train) == 30
    assert len(X_test) == 20
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, pd.Series)
    assert np.isclose(X_train.mean(), 0.0, atol=0.1)
    assert np.isclose(X_train.std(), 1.0, atol=0.1)

def test_data_error_missing_value(clean_data):
    """Test handling of missing values in feature columns."""
    df_missing = clean_data.copy()
    df_missing.loc[5, 'f1'] = np.nan
    preprocessor = KNNPreprocessor()
    
    with pytest.raises(DataPreparationError, match=r"(?i)missing values in features"):
        preprocessor.prepare_data(df=df_missing, target_name='target')

def test_data_error_categorical_feature(clean_data):
    """Test handling of categorical features."""
    df_cat = clean_data.copy()
    df_cat['category'] = ['A'] * 25 + ['B'] * 25
    preprocessor = KNNPreprocessor()
    
    with pytest.raises(DataPreparationError, match=r"(?i)non-numeric features detected"):
        preprocessor.prepare_data(df=df_cat, target_name='target')

def test_data_error_target_not_found(clean_data):
    """Test handling of an incorrect target name."""
    preprocessor = KNNPreprocessor()
    
    with pytest.raises(DataPreparationError, match=r"(?i)target column 'nonexistent_target' not found"):
        preprocessor.prepare_data(df=clean_data, target_name='nonexistent_target')


def test_transform_consistency(clean_data):
    preprocessor = KNNPreprocessor(scaler_method=ScalerType.MINMAX)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df=clean_data, target_name='target')

    # Transform only the training set
    X_transformed = preprocessor.transform(pd.DataFrame(X_train, columns=[f'f{i}' for i in range(X_train.shape[1])]))

    # Check min/max within [0, 1]
    assert X_transformed.min() >= 0.0
    assert X_transformed.max() <= 1.0

