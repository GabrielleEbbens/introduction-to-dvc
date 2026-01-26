from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import dvc.api


def train_model(features_train, targets_train):
    """Train an sklearn model."""
    model = RandomForestRegressor()

    y_train = targets_train.values.ravel()

    # Train the model
    model.fit(features_train, y_train)

    return model


def save_model(model, model_dir="models", model_name="random_forest_model"):
    """Save the trained model as a joblib file."""
    # Save model using joblib
    model_path = f"{model_dir}/{model_name}.pkl"
    joblib.dump(model, model_path)
    return model_path


def main(features_train, targets_train):
    params = dvc.api.params_show()
    model = RandomForestRegressor(
        random_state=params.get("train", "")["random_state"],
        n_estimators=params.get("train", "")["n_estimators"],
        max_depth=params.get("train", "")["max_depth"],
    )
    model.fit(features_train, targets_train.values.ravel())
    save_model(model)
    return model


if __name__ == "__main__":
    features_train = pd.read_csv(
        "data/featurized/features_train.csv", index_col="prod_id"
    )
    targets_train = pd.read_csv(
        "data/featurized/targets_train.csv", index_col="prod_id"
    )
    model = train_model(features_train, targets_train)
    save_model(model)
