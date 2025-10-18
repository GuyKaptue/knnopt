# knnopt/knnopt/core/utils.py
# type: ignore

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from enum import Enum
from typing import Optional, Type, Tuple
import logging
import time

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
# Scaler Selection Utility
# ----------------------------

class ScalerType(str, Enum):
    STANDARD = 'standard'
    MINMAX = 'minmax'
    ROBUST = 'robust'
    NONE = 'none'

SCALER_MAP: dict[ScalerType, Optional[Type]] = {
    ScalerType.STANDARD: StandardScaler,
    ScalerType.MINMAX: MinMaxScaler,
    ScalerType.ROBUST: RobustScaler,
    ScalerType.NONE: None,
}

def get_scaler(scaler_type: str = 'standard') -> Optional[object]:
    """
    Returns an instantiated scaler object based on the specified type.

    Args:
        scaler_type (str): One of 'standard', 'minmax', 'robust', or 'none'.

    Returns:
        Optional[object]: A scaler instance or None.
    """
    try:
        scaler_enum = ScalerType(scaler_type.lower().strip())
    except ValueError:
        logger.error(f"Invalid scaler_type '{scaler_type}'. Valid options: {list(ScalerType)}.")
        return None

    scaler_class = SCALER_MAP[scaler_enum]
    if scaler_class is None:
        logger.info("No scaler selected. Returning None.")
        return None

    logger.info(f"Selected {scaler_enum.value.capitalize()}Scaler for preprocessing.")
    return scaler_class()

# ----------------------------
# Runtime Logging Utility
# ----------------------------

def log_runtime(start_time: float, method_name: str) -> float:
    """
    Logs and returns the runtime of a method.

    Args:
        start_time (float): Start time from time.time().
        method_name (str): Name of the method being timed.

    Returns:
        float: Runtime in seconds.
    """
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Method '{method_name}' completed in {duration:.4f} seconds.")
    return duration
