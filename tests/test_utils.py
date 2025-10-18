# tests/test_utils.py
# type: ignore
import pytest
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from knnopt.core.utils import get_scaler, log_runtime, ScalerType

def test_get_scaler_standard():
    """Test that get_scaler returns the correct StandardScaler instance."""
    scaler = get_scaler(ScalerType.STANDARD.value)
    assert isinstance(scaler, StandardScaler)

def test_get_scaler_minmax():
    """Test that get_scaler returns the correct MinMaxScaler instance."""
    scaler = get_scaler(ScalerType.MINMAX.value)
    assert isinstance(scaler, MinMaxScaler)

def test_get_scaler_robust():
    """Test that get_scaler returns the correct RobustScaler instance."""
    scaler = get_scaler(ScalerType.ROBUST.value)
    assert isinstance(scaler, RobustScaler)

def test_get_scaler_none():
    """Test that get_scaler returns None when 'none' is passed."""
    scaler = get_scaler(ScalerType.NONE.value)
    assert scaler is None

def test_get_scaler_invalid():
    """Test that get_scaler returns None for an invalid method and logs an error."""
    scaler = get_scaler('invalid_method')
    assert scaler is None

def test_log_runtime_accuracy():
    """Test that log_runtime calculates the runtime correctly."""
    start = time.time()
    time.sleep(0.01)  # Small delay
    end = time.time()
    runtime = log_runtime(start, "TestMethod")
    assert runtime > 0.009
    assert runtime < 0.1  # Should be fast enough
