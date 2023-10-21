# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import timedelta, datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import pickle
import os

# Configure logging system
logging.basicConfig(
    level=logging.INFO,  # Set logging level (others: DEBUG, WARNING, ERROR, etc.)
    format="%(levelname)s - %(message)s",  # Define the log message format
)

# Define local path to load dataset
file_path = "/home/dan1dr/data/feature_frame.csv"

# Define path to save curves and model
save_curves_path = "src/module_3/figures"
save_model_path = "src/module_3/models"


# Initial column classification:
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

# Define the functions


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

        # Defined the cols finally used for model
        cols = numerical + binary + predicted

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
    model_name: str, C: float, y_pred, y_test, save_curves_path: str = None
):
    """
    Generate ROC and Precision-Recall curves for a binary classification model
    and save them in a single figure.

    Parameters:
    - model_name (str): Name of the model for labeling the curves.
    - y_pred (array-like): Predicted probabilities or scores.
    - y_test (array-like): True labels.
    - save_dir (str, optional): Directory to save the generated figure.
    If None, the figure will not be saved.

    Returns:
    - None
    """

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f}) - {model_name} (C = {C:.2e})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(
        recall, precision, label=f"PR (AUC = {pr_auc:.2f}) - {model_name} (C = {C:.2e})"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")

    plt.tight_layout()

    if save_curves_path:
        # Define the filename with a timestamp
        figure_filename = f"Evaluation_Curves_{model_name}_C={C}_{timestamp}.png"
        figure_path = os.path.join(save_curves_path, figure_filename)

        plt.savefig(figure_path)

    plt.show()


def train_evaluate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    save_model_path: str,
):
    """
    Train, evaluate, and save a logistic regression model with hyperparameter
    tuning for regularisation. Additionally, plot the curves for ROC and PR.

    Parameters:
    - X_train (pd.DataFrame): Training set features.
    - y_train (pd.Series): Training set outcome.
    - X_val (pd.DataFrame): Validation set features.
    - y_val (pd.Series): Validation set outcome.
    - save_model_path (str): Path to save the trained model.

    Returns:
    - None
    """
    # Define parameter grid for C values
    # We know in advance that strong params works better
    param_grid = {"lr__C": np.logspace(-8, -2, num=9)}

    # Create the pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(penalty="l2", solver="liblinear")),
        ]
    )
    try:
        # Search for the best C hyperparameter
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Get the best model and best parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Return probabilities for plotting the curves
        y_pred = best_model.predict_proba(X_val)[:, 1]

        # Print and save the evaluation curves (ROC and PR)
        generate_evaluation_curves(
            "LogisticRegression", best_params["lr__C"], y_pred, y_val, save_curves_path
        )

        # Save the best model
        lr_C = best_params["lr__C"]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_filename = f"LogisticRegression_{lr_C}_{timestamp}.joblib"
        model_path = os.path.join(save_model_path, model_filename)
        joblib.dump(best_model, model_path)

        logging.info(f"Model {model_filename} saved  successfully!")

    except ValueError as e:
        logging.error(f"ValueError: {str(e)}")

    except pickle.PickleError as e:
        logging.error(f"Error during model save: {str(e)}")


def main():
    data_loaded = read_data(file_path)
    data_processed = preprocess_data(data_loaded)
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_data_split(data_processed)
    train_evaluate_model(X_train, y_train, X_val, y_val, save_model_path)


if __name__ == "__main__":
    main()
