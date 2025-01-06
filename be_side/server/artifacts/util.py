import json
import os
import pickle
import numpy as np

# Global variables
__locations = None
__data_columns = None
__model = None


def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def load_saved_artifacts():
    global __locations, __data_columns, __model

    artifacts_path = os.path.dirname(__file__)
    print("artifacts_path",artifacts_path)
    try:
        with open(os.path.join(artifacts_path, "columns.json"), "r") as f:
            data = json.load(f)
            if "data_columns" in data:
                __data_columns = data["data_columns"]
            else:
                __data_columns = data

        __locations = __data_columns[3:]

        with open(os.path.join(artifacts_path, "banglore_home_prices_model.pickle"), "rb") as f:
            __model = pickle.load(f)

        print("Loaded saved artifacts successfully.")
        # Print debug information
        print(f"Number of features: {len(__data_columns)}")
        print(f"First few columns: {__data_columns[:5]}")

    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        raise

def get_location_names():
    load_saved_artifacts()
    print(f"Locations loaded: {__locations}")
    return __locations

def get_data_columns():
    load_saved_artifacts()
    return __data_columns

if __name__ == '__main__':
    try:
        load_saved_artifacts()
        get_location_names()
        # Debug prints
        print("\nTesting predictions:")
        print(f"Total features in model: {len(__data_columns)}")
        print(f"Available locations: {__locations[:5]}...")

        test_cases = [
            ('1st Phase JP Nagar', 1000, 3, 3),
            ('1st Phase JP Nagar', 1000, 2, 2),
            ('Kalhalli', 1000, 2, 2),
            ('Ejizpura', 1000, 2, 2)
        ]

        for location, sqft, bhk, bath in test_cases:
            try:
                price = get_estimated_price(location, sqft, bhk, bath)
                print(f"\nLocation: {location}")
                print(f"Sqft: {sqft}, BHK: {bhk}, Bath: {bath}")
                print(f"Estimated price: {price}")
            except Exception as e:
                print(f"Error predicting for {location}: {str(e)}")

    except Exception as e:
        print(f"Main execution error: {str(e)}")