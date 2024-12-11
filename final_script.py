# Please refer to the Github Repository for the development 
# and underlying thought train of the models and its feature selection
# https://github.com/floriankozikowski/bike_counters
# This final version uses HistGradientBoosting

###################### Import Libraries ######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import holidays

###################### Load Train Data ######################
data = pd.read_parquet(Path("data") / "train.parquet")

####################### Feature Engineering and Preprocessing ######################

def filter_outliers(data):
    data["date_t"] = data["date"].dt.floor("D")

    cleaned_data = (
        data.groupby(["counter_name", "date_t"])
        ["log_bike_count"].sum()
        .to_frame()
        .reset_index()
        .query("log_bike_count == 0")
        [["counter_name", "date_t"]]
        .merge(data, on=["counter_name", "date_t"], how="right", indicator=True)
        .query("_merge == 'right_only'")
        .drop(columns=["_merge", "date_t"])
    )

    return cleaned_data

data = filter_outliers(data)

def _encode_categorical_features(X):
    X = X.copy()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_features = encoder.fit_transform(X[["counter_name", "site_name"]])
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(["counter_name", "site_name"]),
        index=X.index
    )
    # Drop original columns and add encoded features
    X = X.drop(columns=["counter_name", "site_name"], errors="ignore")
    X = pd.concat([X, encoded_df], axis=1)
    
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

tourist_locations = {
    "Eiffel Tower": (48.8584, 2.2945),
    "Louvre Museum": (48.8606, 2.3376),
    "Notre Dame Cathedral": (48.852968, 2.349902),
    "Sacré-Cœur": (48.8867, 2.3431),
    "Arc de Triomphe": (48.8738, 2.2950)
}

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Add minimum tourist distance feature
def _add_tourist_proximity(X):
    X = X.copy()
    distances = []

    # Calculate distances for each tourist location
    for location, (lat_tourist, lon_tourist) in tourist_locations.items():
        distance = _haversine(X["latitude"], X["longitude"], lat_tourist, lon_tourist)
        distances.append(distance)

    # Add minimum distance as a new column
    X["min_tourist_distance"] = np.min(distances, axis=0)
    return X

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

data = _encode_dates(data)

# Define feature dropping for low predictive value and date (for the XGBoost and Histboost)
def _drop_low_value_features_grad(X):
    X = X.copy()
    X.drop(columns=['date', 'month', 'year', 'counter_id', 'counter_installation_date',
                    'counter_technical_id', 'coordinates', 'latitude', 'longitude'], inplace=True)
    return X

preprocessor_grad = Pipeline(steps=[
    ("add_rush_hour", FunctionTransformer(_add_rush_hour_indicator, validate=False)),
    ("add_weekend", FunctionTransformer(_add_weekend_indicator, validate=False)),
    ("add_holiday", FunctionTransformer(_add_holiday_indicator, validate=False)),
    ("encode_categorical", FunctionTransformer(_encode_categorical_features, validate=False)),
    ("encode_dates", FunctionTransformer(_encode_dates, validate=False)),
    ("add_tourist_proximity", FunctionTransformer(_add_tourist_proximity, validate=False)),
    ("drop_low_value_features_grad", FunctionTransformer(_drop_low_value_features_grad, validate=False)),
])

data = preprocessor_grad.fit_transform(data)

###################### Modeling ######################

features = [col for col in data.columns if col not in ["log_bike_count", "bike_count", "counter_id", "coordinates"]]
X = data[features]
y = data["log_bike_count"]

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



# Define the parameter distributions for RandomizedSearchCV
param_distributions = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [20, 30, 50],
    'l2_regularization': [0.0, 0.1, 1.0]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=HistGradientBoostingRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,  # Number of random combinations to test
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Use the best model from RandomizedSearchCV
best_hgb_model = random_search.best_estimator_

# Make predictions
y_train_pred = best_hgb_model.predict(X_train)
y_valid_pred = best_hgb_model.predict(X_valid)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))

# Print results
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best RMSE (CV): {abs(random_search.best_score_):.2f}")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Validation RMSE: {valid_rmse:.2f}")

###################### Apply to Test Data ######################

test_data = pd.read_parquet(Path("data") / "final_test.parquet")

test_data = _encode_dates(test_data)

# Preprocess the test data
test_data = preprocessor_grad.transform(test_data)

best_hgb_model.fit(X_train, y_train)

# Make predictions
test_predictions = best_hgb_model.predict(test_data)

submission_df = pd.DataFrame({
    "Id": range(len(test_predictions)),  # Assuming sequential indices for Id
    "log_bike_count": test_predictions
})

###################### Save to CSV ######################
submission_file_path = "submission.csv"
submission_df.to_csv(submission_file_path, index=False)