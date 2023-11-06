import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import timedelta, datetime

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib
import os
import json
from typing import List, Dict, Optional
from catboost import Pool
from catboost import CatBoostClassifier
import traceback

logging.basicConfig(
    level=logging.INFO,  # Set logging level (others: DEBUG, WARNING, ERROR, etc.)
    format="%(levelname)s - %(message)s",  # Define the log message format
)

# We will remeber the classification we did in previous notebook:

predicted = ["outcome"]
information = ["variant_id", "order_id", "user_id", "created_at", "order_date"]
numerical = [
    "user_order_seq",
    "normalised_price",
    "discount_pct",
    "global_popularity",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
    "days_since_purchase_variant_id",
    "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id",
    "days_since_purchase_product_type",
    "avg_days_to_buy_product_type",
    "std_days_to_buy_product_type",
]

categorical = ["product_type", "vendor"]
binary = ["ordered_before", "abandoned_before", "active_snoozed", "set_as_regular"]

cols_with_categ = numerical + binary + categorical + predicted


def assess_NA(data: pd.DataFrame):
    """
    Returns a pd.DataFrame denoting the total number of NA
    values and the percentage of NA values in each column.
    """
    # pd.Datadenoting features and the sum of their null values
    nulls = data.isnull().sum().reset_index().rename(columns={0: "count"})
    nulls["percent"] = nulls["count"] * 100 / len(data)

    return nulls


def read_data(file_path: str) -> pd.DataFrame:
    """
    Read and validate data from a CSV file.
    Parameters: file_path (str)
    Returns: pd.DataFrame
    """
    try:
        data = pd.read_csv(file_path, index_col=False)

        if data is not None and not data.empty:
            logging.info("Data loaded successfully.")

            # Info about its shape and nulls
            rows, cols = data.shape
            logging.info(f"Data shape: {rows} rows x {cols} columns")

            null_assessment = assess_NA(data)
            logging.info(
                "Assessment of NA values:\n" + null_assessment.to_string(index=False)
            )

            # Show data sample
            print(data.head(5))

            return data
        else:
            logging.error("Error: The loaded data is empty or None")
            return

    except FileNotFoundError as e:
        logging.error(f"Error: The CSV file is empty. Details: {e}")
    except pd.errors.EmptyDataError as e:
        logging.error(
            f"Error: An unexpected error occurred while loading the data. Details {e}"
        )


def preprocess_data(
    data: pd.DataFrame, remove_if_all_na: bool = False, num_items: int = 5
) -> pd.DataFrame:
    """
    Preprocess data by removing rows with
    filter orders with at least 5 items

    Parameters:
    - data (pd.DataFrame)
    - remove_all_na_rows (bool): If True, remove rows where at least
    1 value is missing. If False, remove rows where all values are missing.

    Returns:
    - pd.DataFrame. The preprocessed dataset
    """
    try:
        initial_length = len(data)
        if remove_if_all_na:  # remove if everything is NA
            data = data.dropna(how="all")
        else:
            data = data.dropna()
        dropped_length = len(data)

        # Filter orders with >= 5 items
        num_items_ordered = data.groupby("order_id")["outcome"].transform("sum")
        processed_data = data[num_items_ordered >= num_items]

        logging.info(f"Length initial data: {len(data)}")
        logging.info(f"Rows dropped with NA's: {initial_length - dropped_length}")
        logging.info(f"Length filtered data: {len(processed_data)}\n")

        return processed_data

    except FileNotFoundError as e:
        logging.error(f"Error: File not found. Details {e}")
        return


def split_sets(df: pd.DataFrame, label: str) -> (pd.DataFrame, pd.Series):
    """
    Return a df with X set (features) and a series y set (outcome)

    Parameters:
    - df: pd.DataFrame

    Returns:
    - X: pd.DataFrame (features only)
    - y: pd.Series (label to predict)
    """
    X = df.drop(columns=label)
    y = df[label]

    return X, y


def temporal_data_split(
    data: pd.DataFrame,
    validation_days: int = 10,
    test_days: int = 10,
    label: str = "outcome",
    cols: List[str] = None,
):
    """
    Perform the temporal data splitting into train, validation, and test set.

    Parameters:
    - data : pd.DataFrame
    - validation_days (int): Number of days for the validation set (default: 10).
    - test_days (int): Number of days for the test set (default: 10).
    - label (str): Name of the outcome variable (default: 'outcome').

    Returns:
    - pd.DataFrame: Training set features.
    - pd.Series: Training set outcome.
    - pd.DataFrame: Validation set features.
    - pd.Series: Validation set outcome.
    - pd.DataFrame: Test set features.
    - pd.Series: Test set outcome.
    """
    try:
        # Check order_date is a datetime and then we group the orders
        try:
            data["order_date"] = pd.to_datetime(data["order_date"]).dt.date
        except KeyError as ke:
            logging.error(f"Key Error: {str(ke)}")
            return
        daily_orders = data.groupby("order_date").order_id.nunique()

        start_date = daily_orders.index.min()
        end_date = daily_orders.index.max()
        logging.info(f"Date from: {start_date}")
        logging.info(f"Date to: {end_date}")

        # Based on the number of days, we get the train days
        num_days = len(daily_orders)
        train_days = num_days - test_days - validation_days

        # Train
        train_start = daily_orders.index.min()
        train_end = train_start + timedelta(days=train_days)

        # Validation (no need to define test)
        validation_end = train_end + timedelta(days=validation_days)

        if cols is None:
            # Use all columns if cols is not specified
            cols = data.columns.tolist()

        train = data[data.order_date <= train_end][cols]
        val = data[(data.order_date > train_end) & (data.order_date <= validation_end)][
            cols
        ]
        test = data[data.order_date > validation_end][cols]

        logging.info(f"Train set ratio: {len(train) / len(data):.2%}")
        logging.info(f"Validation set ratio: {len(val) / len(data):.2%}")
        logging.info(f"Test set ratio: {len(test) / len(data):.2%}")

        X_train, y_train = split_sets(train, label)
        X_val, y_val = split_sets(val, label)
        X_test, y_test = split_sets(test, label)

        return X_train, y_train, X_val, y_val, X_test, y_test

    # not the best practice to catch all exceptio
    except KeyError as ke:
        logging.error(f"Key Error: {str(ke)}")
        return


def generate_evaluation_curves(
    model_name: str, y_pred, y_test, save_curves_path: str = None
):
    """
    Generate ROC and Precision-Recall curves for a binary classification model
    and save them in a single figure.

    Parameters:
    - model_name (str): Name of the model for labeling the curves.
    - y_pred (array-like): Predicted probabilities or scores.
    - y_test (array-like): True labels.
    - save_curves_path (str, optional): Directory to save the generated figure.
    If None, the figure will not be saved.

    Returns:
    - None
    """

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y_%m_%d")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f}) - {model_name}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f}) - {model_name}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")

    plt.tight_layout()

    if save_curves_path:
        # Define the filename with a timestamp
        figure_filename = f"Evaluation_Curves_{timestamp}.png"
        figure_path = os.path.join(save_curves_path, figure_filename)

        plt.savefig(figure_path)

    plt.show()


def find_precision_threshold(y_val, y_scores, target_precision=0.25):
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)
    # Find the closest precision to the target precision
    idx = np.argmin(np.abs(precisions - target_precision))
    return thresholds[idx]


def train_catboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
) -> CatBoostClassifier:
    """
    Trains a CatBoostClassifier model.

    Parameters:
    - train_data (pd.DataFrame): The training dataset features.
    - train_labels (pd.Series): The training dataset labels.
    - val_data (pd.DataFrame): The validation dataset features.
    - val_labels (pd.Series): The validation dataset labels.
    - test_data (pd.DataFrame, optional): The test dataset features.
    - test_labels (pd.Series, optional): The test dataset labels.
    - params (Dict): Dictionary containing the training parameters,
    which must include the names of categorical features.

    Returns:
    - CatBoostClassifier: The trained CatBoost model.
    """

    # Get the list of categorical feature names and remove it from params
    # (should not be passed to CatBoostClassifier later on)
    categorical_features = params.pop("categorical_features", None)
    categorical_features_indices = [
        X_train.columns.get_loc(col) for col in categorical_features
    ]

    # Create CatBoost Pool for train and validation sets
    train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features_indices)

    # Optionally create a Pool for the test set
    test_pool = None  # noqa
    if X_test is not None and y_test is not None:
        test_pool = Pool( # noqa
            X_test, y_test, cat_features=categorical_features_indices
        )

    catboost_model = CatBoostClassifier(**params)

    # Train the model
    catboost_model.fit(train_pool, eval_set=val_pool)

    # Calculate precision threshold
    threshold = find_precision_threshold(
        y_val, catboost_model.predict_proba(X_val)[:, 1]
    )

    return catboost_model, threshold


def handler_fit(event: dict) -> dict:
    """
    The handler function to load data, train the model, and save it to disk.

    Parameters:
    - event (dict): Dictionary with input parameters such as path, params, etc.

    Returns:
    - dict: Dictionary with model information such as model name, path, etc.
    """
    try:
        # Extract parameters from event or set defaults
        data_path = event.get("data_path")
        preprocessing_params = event.get("preprocessing_params", {})
        split_params = event.get("split_params", {})
        # model_parametrisation = event["model_parametrisation"]
        model_parametrisation = event.get("model_parametrisation", {})
        save_model_path = event.get("save_model_path")
        save_curves_path = event.get("save_curves_path")

        # Load and preprocess data
        data = read_data(data_path)
        data_processed = preprocess_data(data, **preprocessing_params)

        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = temporal_data_split(
            data_processed, **split_params
        )

        # Train model
        model, threshold = train_catboost_model(
            X_train,
            y_train,
            X_val,
            y_val,
            model_parametrisation,
        )

        # Save the model
        training_date = datetime.now().strftime("%Y_%m_%d")
        model_name = f"push_{training_date}"
        model_path = os.path.join(save_model_path, model_name)
        joblib.dump(model, model_path)

        # Generate curves
        generate_evaluation_curves(
            model_name, model.predict_proba(X_val)[:, 1], y_val, save_curves_path
        )

        # Return the output dictionary
        return {
            "statusCode": "200",
            "body": json.dumps(
                {
                    "model_path": model_path,
                    "curves_path": save_curves_path,
                    "threshold probability": threshold,
                }
            ),
        }
    except Exception as e:
        traceback.print_exc()  # This will print the stack trace
        return {"statusCode": "500", "body": json.dumps({"error": str(e)})}


# Define input parameters with all the necessary details
event = {
    "data_path": "/home/dan1dr/data/feature_frame.csv",
    "preprocessing_params": {
        "remove_if_all_na": True,
        "num_items": 5,
    },
    "split_params": {
        "validation_days": 10,
        "test_days": 10,
        "label": "outcome",
        "cols": cols_with_categ,
    },
    "model_parametrisation": {
        "iterations": 200,
        "learning_rate": 0.1,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 50,
        "categorical_features": ["product_type", "vendor"],
    },
    "save_model_path": "src/module_4/models",
    "save_curves_path": "src/module_4/figures",
}


def main():
    model_info = handler_fit(event)
    print(model_info)


if __name__ == "__main__":
    main()
