import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save_model():
    # Load dataset
    df = pd.read_csv("data/iris.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("retrained_models", exist_ok=True)

    # Save original (baseline) model
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Original model saved to models/random_forest.pkl")

    # Generate timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save retrained model with timestamp
    retrain_filename = f"retrained_models/random_forest_retrain_{timestamp}.pkl"
    with open(retrain_filename, "wb") as f:
        pickle.dump(clf, f)
    print(f"Retrained model saved to {retrain_filename}")

if __name__ == "__main__":
    train_and_save_model()