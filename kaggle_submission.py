import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import holidays

def _encode_dates(X):
    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    # Apply cyclical encoding
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    X["weekday_sin"] = np.sin(2 * np.pi * X["weekday"] / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * X["weekday"] / 7)

    return X

def _add_rush_hour_indicator(X):
    X = X.copy()
    X["is_rush_hour"] = X["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    return X

def _add_weekend_indicator(X):
    X = X.copy()
    X["is_weekend"] = (X["weekday"] >= 5).astype(int)
    return X

def _add_holiday_indicator(X):
    X = X.copy()
    france_holidays = holidays.FR()  # Use the holidays package for France
    X["is_holiday"] = X["date"].dt.date.apply(lambda x: 1 if x in france_holidays else 0)
    return X

def _drop_low_value_features(X):
    X = X.copy()
    X.drop(columns=['counter_id', 'counter_name', 'site_name', 'date', 'counter_installation_date', 'counter_technical_id', 'coordinates'], inplace=True)
    return X

preprocessor = Pipeline(steps=[
    ("encode_dates", FunctionTransformer(_encode_dates, validate=False)),
    ("add_rush_hour", FunctionTransformer(_add_rush_hour_indicator, validate=False)),
    ("add_weekend", FunctionTransformer(_add_weekend_indicator, validate=False)),
    ("add_holiday", FunctionTransformer(_add_holiday_indicator, validate=False)),
    ("drop_low_value_features", FunctionTransformer(_drop_low_value_features, validate=False)),
])

data = pd.read_parquet(Path("data") / "final_test.parquet") 

def preprocess_data(data):
    processed_array = preprocessor.fit_transform(data)
    processed_data = pd.DataFrame(processed_array, index=data.index)
    
    return processed_data

processed_data = preprocess_data(data)

# Load the trained model
model = joblib.load(Path("models") / "random_forest_model.pkl")

# Make predictions
predictions = model.predict(processed_data)

# Prepare the submission
submission = pd.DataFrame({
    "Id": range(len(predictions)),  # Replace "Id" with the actual ID column in your test set
    "y": predictions  # Replace "y" with the actual target variable name
})

# Save submission to CSV
submission.to_csv("submission.csv", index=False)