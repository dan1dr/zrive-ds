import pandas as pd

from src.module_3.push_mvp import read_data


def test_read_data_valid_file(tmp_path: str):
    """
    Create a temporary CSV file with sample data
    """
    data = pd.DataFrame({"col_1": [1, 4, 7], "col_2": ["pasta", "pizza", "paella"]})
    csv_file = tmp_path / "test_data.csv"
    data.to_csv(csv_file, index=False)

    # Use the function to call data
    result = read_data(csv_file)
    print("Result DataFrame:")
    print(result)

    print("Data DataFrame:")
    print(data)
    # Check if the result is equal
    pd.testing.assert_frame_equal(result, data)


def test_read_data_empty_file(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    # Create empty file
    csv_file.touch()
    result = read_data(csv_file)

    # Check if result is none
    assert result is None


def test_read_data_file_not_exists(tmp_path):
    csv_file = (tmp_path / "test_data.csv").as_posix()
    result = read_data(csv_file)

    assert result is None
