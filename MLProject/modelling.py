import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# MLflow Tracking URI
# =====================================================

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("titanic_preprocessing.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Hyperparameter Tuning
# =====================================================

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5]
}

rf = RandomForestClassifier(
    random_state=42
)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(
    X_train,
    y_train
)

best_model = grid.best_estimator_

predictions = best_model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

# =====================================================
# MLflow Experiment
# =====================================================

mlflow.set_experiment(
    "Titanic-CI"
)

with mlflow.start_run(
    run_name="RandomForest_CI"
) as run:

    # ===============================================
    # Save Run ID
    # ===============================================

    run_id = run.info.run_id

    with open(
        "run_id.txt",
        "w"
    ) as f:
        f.write(run_id)

    mlflow.log_artifact(
        "run_id.txt"
    )

    # ===============================================
    # Parameters
    # ===============================================

    mlflow.log_params(
        grid.best_params_
    )

    # ===============================================
    # Metrics
    # ===============================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # ===============================================
    # Model Artifact (MLflow)
    # ===============================================

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    # ===============================================
    # Local Saved Model
    # ===============================================

    mlflow.sklearn.save_model(
        sk_model=best_model,
        path="saved_model"
    )

    print(
        f"Run ID: {run_id}"
    )

    print(
        f"Best Params: {grid.best_params_}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

print("Training completed successfully.")