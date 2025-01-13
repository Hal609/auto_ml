import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def clean_data(auto):
    # Get the listing date from the public reference
    auto["year_listed"] = auto["public_reference"].map(lambda ref: str(ref)[:4])
    auto["month_listed"] = auto["public_reference"].map(lambda ref: str(ref)[4:6])
    auto["day_listed"] = auto["public_reference"].map(lambda ref: str(ref)[6:8])
    auto = auto.drop(columns="public_reference")

    # Drop unneeded columns
    # auto = auto.drop(columns=["standard_colour"])

    # Turn any years older than 1905 to nan values (as they are unlikely to be valid)
    auto["year_of_registration"] = auto["year_of_registration"].mask(auto["year_of_registration"] < 1905)

    # Mileage of 0 is likely erroneous for older used cars
    mask = (auto["mileage"] < 10) & (auto["vehicle_condition"] == "USED") & (auto["year_of_registration"] < 2018)
    auto.loc[mask, "mileage"] = auto.loc[mask, "mileage"].replace(0, np.nan)

    # Set year of reg to 2020 for all new cars
    new_no_year = auto.loc[auto["year_of_registration"].isna() & (auto["vehicle_condition"] == "NEW")].copy()
    new_no_year["year_of_registration"] = 2020
    auto.loc[auto["year_of_registration"].isna() & (auto["vehicle_condition"] == "NEW"), "year_of_registration"] = 2020

    # Update missing years of reg based on the reg code then drop the reg code column
    def reg_to_year(reg_code):
        try:
            if math.isnan(reg_code): return np.nan
            reg_code = int(reg_code)
        except:
            letters = "ABCDEFGHJKLMNPRSTXYZ"
            if reg_code not in letters: return np.nan
            if reg_code == "V":
                return 1999
            if reg_code == "W":
                return 2000
            return 1983 + letters.find(reg_code)
        
        if reg_code > 71: return np.nan

        if reg_code < 50:
            if reg_code > 20: return np.nan
            return reg_code + 2000
        
        return reg_code + 1950

    auto["reg_code"] = auto["reg_code"].map(reg_to_year)
    auto["year_of_registration"] = auto["year_of_registration"].fillna(auto["reg_code"])
    auto = auto.drop(columns="reg_code")

    # Encode condition to two columns w/ one hot encoder
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    for col in ["vehicle_condition", "crossover_car_and_van", "standard_colour", "fuel_type"]:
        encoded_features = one_hot_encoder.fit_transform(auto[[col]])
        encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out([col]))
        auto = pd.concat([auto.drop(col, axis=1), encoded_df], axis=1)

    
    # Encode text values to integers
    label_encoder = LabelEncoder()
    for column in ["standard_make", "standard_model", "body_type"]:
        auto[column] = label_encoder.fit_transform(auto[column])

    # Remove data where the price is not valid
    # auto = auto.loc[auto['price'] != 9999999]
    auto = auto.loc[auto['price'] < 5e5]

    # Drop any columns which still have na values
    auto = auto.dropna()

    return auto
