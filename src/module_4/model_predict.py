import json
import pandas as pd
import joblib
import traceback


# Load your trained model from disk
model = joblib.load("src/module_4/models/push_2023_11_05")


def handler_predict(event: dict, probability_threshold: float = 0.117) -> dict:
    """
    Handle prediction requests with a given event containing user data.

    Parameters:
    - event (dict): Event data containing the user features for prediction.

    Returns:
    - dict: A dictionary with statusCode and body containing predictions.
    """
    try:
        # Convert the 'users' field of the event to a pandas DataFrame
        data_to_predict = pd.DataFrame.from_dict(event["users"], orient="index")

        # Predict probabilities and convert those to binaries using the threshold
        predictions_proba = model.predict_proba(data_to_predict)[:, 1]

        predictions = (predictions_proba >= probability_threshold).astype(int)

        # Convert predictions to a dictionary with user IDs as keys
        predictions_dict = {
            user_id: int(prediction)
            for user_id, prediction in zip(data_to_predict.index, predictions)
        }

        # Return a successful response with predictions
        return {
            "statusCode": "200",
            "body": json.dumps({"prediction": predictions_dict}),
        }
    except Exception as e:
        traceback.print_exc()  # This will print the stack trace
        # If an error occurs, return an error response
        return {"statusCode": "500", "body": json.dumps({"error": str(e)})}


event_example = {
    "users": {
        "user_id1": {
            "user_order_seq": 4,
            "normalised_price": 0.081052,
            "discount_pct": 0.053512,
            "global_popularity": 0.014925,
            "count_adults": 2.0,
            "count_children": 0.0,
            "count_babies": 0.0,
            "count_pets": 0.0,
            "people_ex_baby": 2.0,
            "days_since_purchase_variant_id": 33.0,
            "avg_days_to_buy_variant_id": 42.0,
            "std_days_to_buy_variant_id": 31.134053,
            "days_since_purchase_product_type": 76.0,
            "avg_days_to_buy_product_type": 30.0,
            "std_days_to_buy_product_type": 24.27618,
            "ordered_before": 0.0,
            "abandoned_before": 0.0,
            "active_snoozed": 0.0,
            "set_as_regular": 0.0,
            "product_type": "ricepastapulses",
            "vendor": "clearspring",
        },
        "user_id2": {
            "user_order_seq": 3,
            "normalised_price": 0.081052,
            "discount_pct": 0.053512,
            "global_popularity": 0.014925,
            "count_adults": 2.0,
            "count_children": 0.0,
            "count_babies": 0.0,
            "count_pets": 0.0,
            "people_ex_baby": 2.0,
            "days_since_purchase_variant_id": 33.0,
            "avg_days_to_buy_variant_id": 42.0,
            "std_days_to_buy_variant_id": 31.134053,
            "days_since_purchase_product_type": 30.0,
            "avg_days_to_buy_product_type": 30.0,
            "std_days_to_buy_product_type": 24.27618,
            "ordered_before": 0.0,
            "abandoned_before": 0.0,
            "active_snoozed": 0.0,
            "set_as_regular": 0.0,
            "product_type": "ricepastapulses",
            "vendor": "clearspring",
        },
        "user_id3": {
            "user_order_seq": 2,
            "normalised_price": 0.081052,
            "discount_pct": 0.053512,
            "global_popularity": 0.014925,
            "count_adults": 2.0,
            "count_children": 0.0,
            "count_babies": 0.0,
            "count_pets": 0.0,
            "people_ex_baby": 2.0,
            "days_since_purchase_variant_id": 33.0,
            "avg_days_to_buy_variant_id": 42.0,
            "std_days_to_buy_variant_id": 31.134053,
            "days_since_purchase_product_type": 30.0,
            "avg_days_to_buy_product_type": 30.0,
            "std_days_to_buy_product_type": 24.27618,
            "ordered_before": 0.0,
            "abandoned_before": 0.0,
            "active_snoozed": 0.0,
            "set_as_regular": 0.0,
            "product_type": "ricepastapulses",
            "vendor": "clearspring",
        },
    }
}


def main():
    predictions = handler_predict(event_example)
    print(predictions)


if __name__ == "__main__":
    main()
