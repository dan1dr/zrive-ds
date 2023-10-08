import pandas as pd
import requests
from unittest.mock import Mock

# import pytest

from src.module_1.module_1_meteo_api import process_data, VARIABLES, call_api


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


class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError(f"HTTPError: {self.status_code}")


def test_call_api_200(monkeypatch):
    mocked_response = Mock(return_value=MockResponse("mocked_response", 200))
    monkeypatch.setattr(requests, "get", mocked_response)
    response = call_api("mock_url")
    assert response.status_code == 200
    assert response.json() == "mocked_response"


def test_call_api_404(monkeypatch):
    mocked_response = Mock(return_value=MockResponse("Not Found", 404))
    monkeypatch.setattr(requests, "get", mocked_response)
    response = call_api("mock_url")
    assert response.status_code == 404
    assert response.json() == "Not Found"
