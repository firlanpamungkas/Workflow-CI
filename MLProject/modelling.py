import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("titanic_preprocessing.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5]
}

rf = RandomForestClassifier(
    random_state=42
)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5
)

grid.fit(
    X_train,
    y_train
)

best_model = grid.best_estimator_

pred = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

with mlflow.start_run():

    mlflow.log_params(
        grid.best_params_
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model"
    )

    mlflow.sklearn.save_model(
        sk_model=best_model,
        path="saved_model"
    )

print(
    f"Accuracy : {accuracy:.4f}"
)