import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import dvc.api


def featurize(data: pd.DataFrame, target_column: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Apply feature engineering using sklearn pipeline.

    Args:
        data (pd.DataFrame): Preprocessed data with prod_id as index
        target_column (str): The name of the target column

    Returns:
        pd.DataFrame: Features
        pd.DataFrame: Targets
    """
    # Separate features and target
    features = data.drop(target_column, axis=1)
    targets = pd.DataFrame(data[target_column])

    # Define feature groups
    categorical_features = ["ages", "review_difficulty", "country"]
    numerical_features = ["piece_count", "star_rating"]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    # Fit and transform the features
    features_transformed = preprocessor.fit_transform(features)

    # Get feature names for the transformed data
    num_feature_names = numerical_features
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_features
    )
    all_feature_names = list(num_feature_names) + list(cat_feature_names)

    # Create DataFrame with transformed features
    features_transformed_data_frame = pd.DataFrame(
        features_transformed, columns=all_feature_names, index=features.index
    )

    return features_transformed_data_frame, targets


def main(
    processed_data: pd.DataFrame, target_column: str
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Featurize the data and split into train and test sets.

    Args:
        processed_data (pd.DataFrame): The processed data
        target_column (str): The name of the target column

    Returns:
        tuple: (features_train, features_test, targets_train, targets_test)
               All DataFrames with preserved indices from original data
    """
    params = dvc.api.params_show(stages="featurizing")
    features, targets = featurize(processed_data, target_column)

    # Split while preserving indices
    train_indices, test_indices = train_test_split(
        features.index,
        test_size=params.get("featurize", "")["split_ratio"],
        random_state=params.get("featurize", "")["random_state"],
    )

    features_train = features.loc[train_indices]
    features_test = features.loc[test_indices]
    targets_train = targets.loc[train_indices]
    targets_test = targets.loc[test_indices]

    return features_train, features_test, targets_train, targets_test


if __name__ == "__main__":
    processed_data = pd.read_csv(
        "data/processed/preprocessed_lego_sets.csv", index_col="prod_id"
    )
    features_train, features_test, targets_train, targets_test = main(
        processed_data, "list_price"
    )
    features_train.to_csv("data/featurized/features_train.csv")
    features_test.to_csv("data/featurized/features_test.csv")
    targets_train.to_csv("data/featurized/targets_train.csv")
    targets_test.to_csv("data/featurized/targets_test.csv")
