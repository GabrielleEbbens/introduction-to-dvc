import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
from dvclive import Live
from matplotlib import pyplot as plt


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


def plot_feature_importance(model, features_test, top_k=10):
    """Plot the top k feature importance of the model."""
    feature_importance = model.feature_importances_
    feature_names = features_test.columns

    # Get indices of top k features in descending order
    top_indices = feature_importance.argsort()[::-1][:top_k]

    # Get top k features and their importance
    top_features = feature_names[top_indices]
    top_importance = feature_importance[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Reverse the order for plotting so highest importance appears at top
    ax.barh(top_features[::-1], top_importance[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_k} Random Forest Feature Importance")
    fig.tight_layout()
    return fig


def main(model_path, features_test, targets_test, plot_dir):
    """Evaluate the model and return metrics."""
    model = load_model(model_path)
    metrics, y_pred = evaluate_model(model, features_test, targets_test)
    with Live("data/evaluation") as live:
        for metric, value in metrics.items():
            live.log_metric(metric, value)
    test_features_with_predictions_and_targets = features_test.copy()
    test_features_with_predictions_and_targets["target"] = targets_test.values
    test_features_with_predictions_and_targets["prediction"] = y_pred
    fig = plot_feature_importance(model, features_test)
    with Live(str(plot_dir), resume=True) as live:
        live.log_image("feature_importance.png", fig)
    plt.close(fig)
    return metrics, test_features_with_predictions_and_targets, fig


if __name__ == "__main__":
    model_path = "models/random_forest_model.pkl"
    features_test = pd.read_csv(
        "data/featurized/features_test.csv", index_col="prod_id"
    )
    targets_test = pd.read_csv("data/featurized/targets_test.csv", index_col="prod_id")
    plot_path = "data/evaluation/plots/images/feature_importance.png"
    metrics, test_features_with_predictions_and_targets, fig = main(
        model_path, features_test, targets_test, plot_path
    )
    test_features_with_predictions_and_targets.to_csv(
        "data/evaluation/test_features_with_predictions_and_targets.csv"
    )
