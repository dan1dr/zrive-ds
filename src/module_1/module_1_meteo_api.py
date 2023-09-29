"""Import libraries"""
import requests
import logging
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import urllib3
# import json
# import numpy as np


# Disable the InsecureRequestWarning due to using verify=False from corporate laptop
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


"""Define global variables"""

API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COOL_OFF_TIME = 5
MAX_RETRY_ATTEMPTS = 3

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

# All models have data for the 3 variables except soil moisture
# which is only provided by MRI_AGCM3 and EC_Earth
MODELS = "CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S" # noqa


"""Define auxiliar functions"""


def call_api(url):
    # Need to add the verify=False as working from my corporate laptop.
    # Tried to authenticate SSL by changing lots of config, disabled SSL, etc
    # It only works from the office - not the best practice I know # noqa
    try:
        # to-do: add the cool off
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            print("\nConnected Succesfully!")
            return response
        else:
            logging.error(f"\nAPI request failed with Code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as exception:
        logging.error(f"\nAPI request failed with Exception: {exception}")
        return None


def get_data_meteo_api(city, from_year, until_year):
    coordinates = COORDINATES.get(city)
    if coordinates is None:
        print("No data for the city")
        return None

    # Define local variables for lattitude and longitude
    lat = coordinates["latitude"]
    long = coordinates["longitude"]

    # Create the final URL
    url = (
        f"{API_URL}latitude={lat}&longitude={long}&start_date={from_year}"
        f"-01-01&end_date={until_year}-12-31&models={MODELS}&daily={VARIABLES}"
    )

    # Define a num of max attempts for calling again
    retry_count = 0

    while retry_count < MAX_RETRY_ATTEMPTS:
        data = call_api(url)
        if data:
            return data.json()
        else:
            print(f"Retrying after {COOL_OFF_TIME} seconds again...\n")
            time.sleep(COOL_OFF_TIME)
            retry_count += 1
    print("You reached the number of maximum attempts. Stopping the execution")
    sys.exit()


def clean_data(raw_data):
    """
    (Not used)
    Function defined for cleaning None values inside a dictionary
    """
    cleaned_data = {}
    for key, value in raw_data.items():
        if value is not None:
            if isinstance(value, dict):
                cleaned_value = clean_data(
                    value
                )  # Recursively clean sub-dictionaries starting again
                if (
                    cleaned_value
                ):  # Check if the sub-dictionary is not empty after cleaning
                    cleaned_data[key] = cleaned_value

            elif isinstance(value, list):  # do the same but for list within the dict
                # Clean each item in a list
                cleaned_list = [item for item in value if item is not None]
                if cleaned_list:
                    cleaned_data[key] = cleaned_list
            else:
                cleaned_data[key] = value
    return cleaned_data


def process_data(data):
    """
    Reads a df and for each city calculates mean + std for each variable
    """
    calculated_df = data[["city", "time"]].copy()
    for var in VARIABLES.split(","):
        # For each climate var in the loop, if the name is in variable it's stored
        idxs = [col for col in data.columns if col.startswith(var)]
        # Save each variable with its corresponding mean and std (axis=1 per each row)
        # We have decided to compute the mean of all models for one day
        # e.g. mean of all precipitation_sum for Madrid on 1950-01-01
        # We could have done the mean of all occurences during one year for one model
        calculated_df[f"{var}_mean"] = data[idxs].mean(axis=1)
        calculated_df[f"{var}_std"] = data[idxs].std(axis=1)
    return calculated_df


def plot_data(data):
    """
    Plot data for each variable and city. 1 plot per variable
    """
    plt.style.use("bmh")
    # plt.rcdefaults()

    # We create the grid which will be 3 graphs
    rows = 3
    cols = 1
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    # Convert the 2D plot into 1D for easier iteration
    axs = axs.flatten()

    # We rescale the dates for taking only years
    data["year"] = pd.to_datetime(data["time"]).dt.year

    # Enter a loop that iterates over unique city
    for i, city in enumerate(data.city.unique()):
        # Filters data to select only that city we're iterating
        city_data = data.loc[lambda x: x.city == city, :]
        print(city_data.head())

        # Now for each variable we
        for k, var in enumerate(VARIABLES.split(",")):
            city_data["mid_line"] = city_data[f"{var}_mean"]
            city_data["upper_line"] = city_data[f"{var}_mean"] + city_data[f"{var}_std"]
            city_data["lower_line"] = city_data[f"{var}_mean"] - city_data[f"{var}_std"]
            # Plot yearly mean values
            # We have rescaled them to keep only year, now we group then by that
            city_data.groupby("year")["mid_line"].apply("mean").plot(
                ax=axs[k], label=f"{city}", color=f"C{i}"
            )
            city_data.groupby("year")["upper_line"].apply("mean").plot(
                ax=axs[k], ls="--", label="_nolegend_", color=f"C{i}"
            )
            city_data.groupby("year")["lower_line"].apply("mean").plot(
                ax=axs[k], ls="--", label="_nolegend_", color=f"C{i}"
            )
            axs[k].set_title(var)

    plt.tight_layout()
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.savefig("src/module_1/climate_evolution.png")  # save fig to path

    pass


def main():
    # data = get_data_meteo_api("Madrid", 1950, 2050)
    # cleaned_data = clean_data(data)
    # processed_data = process_data(cleaned_data)
    data = []
    for city, coord in COORDINATES.items():
        data.append(
            pd.DataFrame(get_data_meteo_api(city, 1950, 2050)["daily"]).assign(
                city=city
            )
        )

    data_df = pd.concat(data)
    print(data_df.head())

    calculated_df = process_data(data_df)
    print(calculated_df)

    plot_data(calculated_df)


if __name__ == "__main__":
    main()
