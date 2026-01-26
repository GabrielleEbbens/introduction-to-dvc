import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
from dvclive import Live


def load_model(model_path):
    """Load the trained model from a tar.gz file."""
    # Extract the tarball
    model_path = Path(model_path)
    model = joblib.load(model_path)

    return model


def evaluate_model(model, features_test, targets_test):
    """Evaluate the model and return metrics."""
    # Make predictions
    y_test = targets_test.values.ravel()
    y_pred = model.predict(features_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "r2_score": r2,
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "n_samples": len(y_test),
    }

    return metrics, y_pred


def main(model_path, features_test, targets_test):
    """Evaluate the model and return metrics."""
    model = load_model(model_path)
    metrics, y_pred = evaluate_model(model, features_test, targets_test)
    with Live("data/evaluation") as live:
        for metric, value in metrics.items():
            live.log_metric(metric, value)
    test_features_with_predictions_and_targets = features_test.copy()
    test_features_with_predictions_and_targets["target"] = targets_test.values
    test_features_with_predictions_and_targets["prediction"] = y_pred
    return metrics, test_features_with_predictions_and_targets


if __name__ == "__main__":
    model_path = "models/linear_regression_model.pkl"
    features_test = pd.read_csv(
        "data/featurized/features_test.csv", index_col="prod_id"
    )
    targets_test = pd.read_csv("data/featurized/targets_test.csv", index_col="prod_id")
    metrics, test_features_with_predictions_and_targets = main(
        model_path, features_test, targets_test
    )
    test_features_with_predictions_and_targets.to_csv(
        "data/evaluation/test_features_with_predictions_and_targets.csv"
    )
