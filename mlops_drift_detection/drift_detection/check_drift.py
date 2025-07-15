
import pandas as pd
import numpy as np
from alibi_detect.cd import KSDrift

def check_drift(reference_data_path, new_data_path, p_val_threshold=0.05):
    # Load reference (training) data
    ref_df = pd.read_csv(reference_data_path)
    ref_X = ref_df.drop(columns=["target"]).values

    # Load new data (simulate drift)
    new_df = pd.read_csv(new_data_path)
    new_X = new_df.drop(columns=["target"]).values

    # Initialize drift detector
    detector = KSDrift(ref_X, p_val=p_val_threshold)

    # Check drift
    preds = detector.predict(new_X)
    drift = preds["data"]["is_drift"]
    p_vals = preds["data"]["p_val"]

    if drift:
        print("Drift detected! p-values:", p_vals)
    else:
        print("No drift detected. p-values:", p_vals)

    return drift

if __name__ == "__main__":
    # drift = check_drift("data/iris.csv", "data/iris.csv")
    drift = check_drift("data/iris.csv", "data/iris_drifted.csv")