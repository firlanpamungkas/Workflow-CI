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
# MLflow Tracking URI (DagsHub)
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
# MLflow Logging
# =====================================================

active_run = mlflow.active_run()

if active_run:
    run_id = active_run.info.run_id
else:
    run = mlflow.start_run()
    run_id = run.info.run_id

# =====================================================
# Save Run ID
# =====================================================

with open(
    "run_id.txt",
    "w"
) as f:
    f.write(run_id)

mlflow.log_artifact(
    "run_id.txt"
)

# =====================================================
# Log Parameters
# =====================================================

mlflow.log_params(
    grid.best_params_
)

# =====================================================
# Log Metrics
# =====================================================

mlflow.log_metric(
    "accuracy",
    accuracy
)

# =====================================================
# Log Model to MLflow Artifact Store
# =====================================================

mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model"
)

# =====================================================
# Save Local Model
# =====================================================

mlflow.sklearn.save_model(
    sk_model=best_model,
    path="saved_model"
)

# =====================================================
# Console Output
# =====================================================

print(f"Run ID: {run_id}")
print(f"Best Params: {grid.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print("Training completed successfully.")