# ML Model Monitoring, Drift Detection, and Automated Retraining Workflows
Here, we simulate an end-to-end MLOps for model monitoring, drift detection, and automated retraining workflows.


## Components
1. Model Training Pipeline (scikit-learn model)
2. Model Drift Detector (Alibi Detect)
3. Airflow DAG to orchestrate retraining


## Project Folder Structure
```sh
mkdir mlops_drift_detection && cd mlops_drift_detection

mkdir data models drift_detection airflow_dags retrained_models
```

## Install Dependencies

### Set up Python Virtual Environment
```sh
python3 -m venv ddrift-venv
source ddrift-venv/bin/activate
```

### Required Packages
```sh
touch requirements.txt
```

```sh
# requirements.txt

scikit-learn 
pandas 
alibi-detect 
apache-airflow==3.0.2
```

## Part I: Prepare Training Data
For simplicity, we will use the Iris Dataset

```sh
cd data
touch prepare_data.py
```
```sh
# data/prepare_data.py

import pandas as pd
from sklearn.datasets import load_iris

def save_iris_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv("data/iris.csv", index=False)
    print("Iris dataset saved to data/iris.csv")

if __name__ == "__main__":
    save_iris_dataset()
```

Run the data/prepare_data.py script
```sh
cd ..
python data/prepare_data.py
```


## Part II: Train & Save a Model
We will train a RandomForestClassifier model:

```sh
cd models
touch train_model.py
```

```sh
# models/train_model.py

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
```

Run the models/train_model.py script
```sh
cd ..
python models/train_model.py
```


## Part III: Implement Drift Detection
We will use Alibi Detect to monitor drift on new incoming data batches. Check for drift with script below:

```sh
cd drift_detection
touch check_drift.py generate_drifted_data.py
```
```sh
# drift_detection/check_drift.py

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
```


Simulate drift by adding noise to numeric columns except the Target column:
```sh
# drift_detection/generate_drifted_data.py

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
```

### Run the Scripts
Create the drifted dataset
```sh
cd ..
python drift_detection/generate_drifted_data.py
```

To test drift detection:
```sh
python drift_detection/check_drift.py
```



## Part IV: Set up Apache Airflow with Docker & Docker Compose
```sh
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/3.0.2/docker-compose.yaml'
```

Airflow set up with Docker Compose Needs to Know Your Host User Id. If not, the directories dags, logs, plugins, config will created with Root Ownership. Run the commands below:

```sh
mkdir -p ./dags ./logs ./plugins ./config
echo -e "AIRFLOW_UID=$(id -u)" > .env
```


### Initialize Airflow Configurations
```sh
docker compose run airflow-cli airflow config list
```


### Initialize the Database to Create First User Account (with login: airflow, and password: airflow)
```sh
docker compose up airflow-init
```


### Run Airflow
This will run Airflow in "watch mode":
```sh
docker compose up
```

In a second terminal, check the docker processes running to make sure that all containers are in a healthy condition:
```sh
docker ps
```

### Access Airflow Web Interface
The default account has the login: airflow and password: airflow.
```sh
http://localhost:8080
```

### Access Airflow via REST API with Curl command (optional)
```sh
ENDPOINT_URL="http://localhost:8080/"
curl -X GET  \
    --user "airflow:airflow" \
    "${ENDPOINT_URL}/api/v1/pools"
```


## Part V: Create Airflow DAG File to Automate Retrain Workflow

```sh
cd dags
touch drift_retraining_dag.py
```
```sh
# dags/drift_retraining_dag.py

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from datetime import datetime
import subprocess


def decide_retrain_branch():
    """
    Run the drift detection script and decide whether to retrain.
    """
    result = subprocess.run(
        ["python", "drift_detection/check_drift.py"],
        capture_output=True,
        text=True
    )
    output = result.stdout
    print("Drift detection output:\n", output)

    # Determine drift based on script output
    if "Drift detected!" in output:
        return "retrain_model"
    else:
        return "no_retrain"


def retrain_model_task():
    """
    Retrain the model.
    """
    print("Starting retraining process...")
    subprocess.run(["python", "models/train_model.py"])
    print("Retraining completed.")


def skip_retrain_task():
    """
    Skip retraining.
    """
    print("No drift detected. Skipping retraining.")


with DAG(
    dag_id="model_drift_detection_and_retraining",
    schedule=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    description="Detect model drift and retrain if needed",
    tags=["mlops", "drift detection"]
) as dag:

    decide_branch = BranchPythonOperator(
        task_id="decide_retrain_branch",
        python_callable=decide_retrain_branch,
    )

    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model_task,
    )

    no_retrain = PythonOperator(
        task_id="no_retrain",
        python_callable=skip_retrain_task,
    )

    decide_branch >> [retrain_model, no_retrain]
```


### Export the DAGs folder
```sh
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
```

### Start the Airflow Scheduler
```sh
airflow scheduler
```

1. Go to Airflow Web page 
2. Click on Dags on the left panel and look out for dag_id="model_drift_detection_and_retraining". 
3. Toggle the button to enable it
4. Click on the "play button" to run the Dag and manually trigger it.



### How it Works!
1. decide_retrain_branch: 
    1. Runs the check_drift.py script and checks if drift was detected
    2. Returns retrain_model or no_retrain as the next task
2. retrain_model: 
    1. Retrains the model if drift is detected at retrained_models/random_forest_retrain_{timestamp}.pkl.
    2. All retrained model versions are saved separately.
3. no_retrain: 
    1. Just logs that retraining is skipped. Baseline model is kept at models/random_forest.pkl



### Where Retrained Model Artifact is Stored

```sh
ls retrained_models
```


### Clean up Airflow
Shut down the Docker Compose and remove the directory in which you downloaded the docker-compose.yaml:

```sh
docker ps
docker compose down --volumes --rmi all
deactivate
rm -rf ddrift-venv config logs plugins
```

# Like, Comment, and Subscribe to iQuant YouTube Channel

