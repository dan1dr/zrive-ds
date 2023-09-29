""" This is a dummy example to show how to import code from src/ for testing"""
import pandas as pd

from src.module_1.module_1_meteo_api import process_data, VARIABLES


def test_process_data():
    test_variable = VARIABLES.split(",")[1]
    data = pd.DataFrame(
        {
            "city": ["Rio", "Rio", "Rio"],
            "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
            f"{test_variable}_model1": [20, 15, 20],
            f"{test_variable}_model2": [1, 2, 2],
            f"{test_variable}_model3": [4, 5, 7],
        }
    )

    expected = pd.DataFrame(
        {
            "city": {0: "Rio", 1: "Rio", 2: "Rio"},
            "time": {0: "2021-01-01", 1: "2021-01-02", 2: "2021-01-03"},
            f"{test_variable}_mean": {
                0: 8.333333333333334,
                1: 7.333333333333333,
                2: 9.666666666666666,
            },
            f"{test_variable}_std": {
                0: 10.214368964029708,
                1: 6.8068592855540455,
                2: 9.291573243177568,
            },
        }
    )
    # We want to keep only the processed test_variable but not all the other variables
    info_cols = ["city", "time"]
    test_cols = [f"{test_variable}_mean", f"{test_variable}_std"]
    pd.testing.assert_frame_equal(
        process_data(data)[info_cols + test_cols],
        expected,
    )


















def process_data(data):
    """
    Reads a df and for each city calculates mean + std for each variable
    """
    calculated_df = data[["city", "time"]].copy()
    for var in VARIABLES.split(","):
        # For each climate var in the loop, if the name is in variable, it stores the column
        idxs = [col for col in data.columns if col.startswith(var)]
        # Save each variable with its corresponding mean and std. Axis=1 as calculated per each row.
        # We have decided to compute the mean of all models for one day
        # e.g. mean of all precipitation_sum for Madrid on 1950-01-01
        # We could have done the mean of all occurences during one year for one model
        calculated_df[f"{var}_mean"] = data[idxs].mean(axis=1)
        calculated_df[f"{var}_std"] = data[idxs].std(axis=1)
    return calculated_df