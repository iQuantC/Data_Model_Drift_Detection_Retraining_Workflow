import pandas as pd
import numpy as np

def create_drifted_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df_drifted = df.copy()
    # Add noise to numeric columns
    for col in df_drifted.columns:
        if col != "target":
            df_drifted[col] += np.random.normal(3.0, 1.0, size=df_drifted[col].shape)
    df_drifted.to_csv(output_path, index=False)
    print(f"Drifted data saved to {output_path}")

if __name__ == "__main__":
    create_drifted_data("data/iris.csv", "data/iris_drifted.csv")