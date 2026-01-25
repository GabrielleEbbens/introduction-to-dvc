import pandas as pd


def select_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Selects the columns we want to keep.

    Args:
        data (pd.DataFrame): The data to select columns from.
        columns (list[str]): The columns to select.

    Returns:
        pd.DataFrame: The data with the selected columns.
    """
    return data[columns]


def impute_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using pandas methods only.

    Args:
        data (pd.DataFrame): The data to impute missing values from.

    Returns:
        pd.DataFrame: The data with missing values imputed.
    """
    data_imputed = data.copy()

    # Mode imputation for categorical columns
    categorical_cols = ["review_difficulty", "star_rating"]
    for col in categorical_cols:
        if col in data_imputed.columns and data_imputed[col].isna().any():
            mode_value = data_imputed[col].mode()
            if len(mode_value) > 0:
                data_imputed[col].fillna(mode_value[0], inplace=True)

    return data_imputed


def main(
    data: pd.DataFrame, columns_to_keep: list[str], columns_to_impute: list[str]
) -> pd.DataFrame:
    """
    Preprocess the data by selecting the columns we want to keep and imputing missing values.

    Args:
        data(pd.DataFrame): The data to preprocess
        columns_to_keep(list[str]): The columns to keep
        columns_to_impute(list[str]): The columns to impute

    Returns:
        pd.DataFrame: The preprocessed data
    """
    data_preprocessed = select_columns(data, columns_to_keep)
    data_imputed = impute_missing_values(data_preprocessed)
    return data_imputed.set_index("prod_id")


if __name__ == "__main__":
    columns_to_keep = [
        "prod_id",
        "ages",
        "piece_count",
        "review_difficulty",
        "country",
        "star_rating",
        "list_price",
    ]
    columns_to_impute = ["review_difficulty", "star_rating"]
    data = pd.read_csv("data/raw/lego_sets.csv")
    data_preprocessed = main(data, columns_to_keep, columns_to_impute)
    data_preprocessed.to_csv("data/processed/preprocessed_lego_sets.csv")
