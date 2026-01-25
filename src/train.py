from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib
import tarfile
from pathlib import Path


def train_model(features_train, targets_train):
    """Train a linear regression model."""
    model = LinearRegression()

    # Flatten targets to 1D array for sklearn
    y_train = targets_train.values.ravel()

    # Train the model
    model.fit(features_train, y_train)

    return model


def save_model_as_tarball(
    model, model_dir="models", model_name="linear_regression_model"
):
    """Save the trained model as a tar.gz file."""
    # Save model using joblib
    model_path = f"{model_dir}/{model_name}.pkl"
    joblib.dump(model, model_path)

    # Create tar.gz file
    tarball_path = f"{model_dir}/{model_name}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(model_path, arcname=f"{model_name}.pkl")

    # Clean up individual file
    Path(model_path).unlink()

    return tarball_path


if __name__ == "__main__":
    features_train = pd.read_csv(
        "data/featurized/features_train.csv", index_col="prod_id"
    )
    targets_train = pd.read_csv(
        "data/featurized/targets_train.csv", index_col="prod_id"
    )
    model = train_model(features_train, targets_train)
    save_model_as_tarball(model)
