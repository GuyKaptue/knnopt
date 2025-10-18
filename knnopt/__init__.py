# knnopt/knnopt/__init__.py
#
# This file serves as the public interface for the entire 'knnopt' subpackage.
# It makes the main orchestrator class and custom exceptions easily accessible.

# Import the main class and custom exception from their respective modules
# NOTE: We assume 'knnopt' is the package name and these files are sibling modules.
# We also assume that comparison, visualization, and preprocessing modules 
# are available as sibling modules within the 'knnopt/knnopt/' directory.

from knnopt.knn_opt import KNNOpt 

# Define the package version (best practice)
__version__ = '0.1.0' 

# Define what is accessible when a user runs 'from knnopt import *' 
# (Highly recommended for user experience)
__all__ = [
    'KNNOpt',
]