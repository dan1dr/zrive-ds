import pandas as pd

from src.module_1.module_1_meteo_api import process_data, VARIABLES


def test_process_data():
    test_variable = VARIABLES[1]
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
            # Reminder that numpy assumes you're dealing with population (divided by N)
            # Pandas assumes sample, so it divides by N-1
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
