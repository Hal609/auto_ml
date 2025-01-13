import pickle
import pandas as pd


def reverse_encoding(df, column=None):
    # Load the saved encoder
    with open("label_encoder.pkl", "rb") as f:
        loaded_label_encoder = pickle.load(f)
    with open('categories_map.pkl', 'rb') as f:
        categories_map = pickle.load(f)

    if column is None or column == "make_model":
        # Check the data type and cast to integers if necessary
        if df["make_model"].dtype != int:
            df["make_model"] = df["make_model"].astype(int)

        # Reverse the encoding
        df["make_model"] = loaded_label_encoder.inverse_transform(df["make_model"])

    # Reversing one-hot encoding
    for col, categories in categories_map.items():
        if column is not None:
            if col != column:
                break
        # Find the one-hot encoded columns for this categorical feature
        one_hot_columns = categories
        
        # Map one-hot encoded columns back to original categories
        df[col] = df[one_hot_columns].idxmax(axis=1)
        
        # Drop the one-hot encoded columns
        df.drop(columns=one_hot_columns, inplace=True)

    return df

def reverse_scaling(df):
    with open('robust_scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)

    # Inverse transform (RobustScaler first, then PowerTransformer)
    cols_to_scale = ["mileage", "year_of_registration", "price", "year_listed", "month_listed", "day_listed"]

    # Inverse transform the scaled columns
    unscaled_data = loaded_scaler.inverse_transform(df[cols_to_scale])

    # Replace the scaled columns in the original DataFrame
    df[cols_to_scale] = pd.DataFrame(unscaled_data, columns=cols_to_scale)


def reverse_transform(df):
    # Load the transformer
    with open('power_transformer.pkl', 'rb') as f:
        loaded_pt = pickle.load(f)

    # Inverse transform the data
    original_data = loaded_pt.inverse_transform(df[['price', 'mileage', 'year_of_registration']])

    # Replace the transformed columns in the DataFrame with the original values
    df[['price', 'mileage', 'year_of_registration']] = pd.DataFrame(original_data, columns=['price', 'mileage', 'year_of_registration'])

    return df

def reverse_transform(df):
    df = reverse_encoding(df)
    df = reverse_transform(df)
    df = reverse_scaling(df)

    return df