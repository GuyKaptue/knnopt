# Demo script for KNN optimization
# examples/demo_knnopt.py
# type: ignore

import pandas as pd
from sklearn.datasets import load_iris

from knnopt.knn_opt import KNNOpt
from knnopt.core.utils import ScalerType

def main():
    # Load sample dataset
    data = load_iris()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['target'] = data.target
    target_name = 'target'

    scalers = [ScalerType.NONE, ScalerType.STANDARD, ScalerType.MINMAX, ScalerType.ROBUST]
    for scaler in scalers:
        print(f"\nRunning KNNOpt with scaler: {scaler.name}")
        optimizer = KNNOpt(scaler_method=scaler)
        optimizer.run(
            df=df,
            target_name=target_name,
            test_size=0.3,
            random_state=42,
            cv_folds=5,
            k_max=15
        )
        print("Optimization Summary:")
        results_df = optimizer.get_results()
        print(results_df)



if __name__ == "__main__":
    main()
