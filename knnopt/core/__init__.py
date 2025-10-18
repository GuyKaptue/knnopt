# knnopt/knnopt/core/__init__.py
#
#
# This file defines the core sub-package, allowing internal modules to be 
# easily accessed by the main KNNOpt class and other components.
#

# Import key classes and functions to be available when importing from .core
from .utils import get_scaler, log_runtime, ScalerType
from .preprocessing import KNNPreprocessor, DataPreparationError
from .optimization import OptimizationSuite
from .comparison import ComparisonSuite
from .visualization import VisualizationSuite

# Define what is accessible externally (useful for user import and testing)
__all__ = [
    # Utilities
    'get_scaler',
    'log_runtime',
    'ScalerType',
    
    # Core Components
    'KNNPreprocessor',
    'DataPreparationError',
    'OptimizationSuite',
    'ComparisonSuite',
    'VisualizationSuite',
]
