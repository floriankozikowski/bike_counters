import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

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

def _add_site_avg_log_bike_count(X):
    X = X.copy()
    site_avg = X.groupby("site_id")["log_bike_count"].transform("mean")
    X["site_avg_log_bike_count"] = site_avg
    return X

def _add_daily_variance(X):
    X = X.copy()
    daily_variance = X.groupby(["site_id", "year", "month", "day"])["log_bike_count"].transform("var")
    X["daily_variance"] = daily_variance.fillna(0)
    return X

preprocessor = Pipeline(steps=[
    ("encode_dates", FunctionTransformer(_encode_dates, validate=False)),
    ("add_rush_hour", FunctionTransformer(_add_rush_hour_indicator, validate=False)),
    ("add_weekend", FunctionTransformer(_add_weekend_indicator, validate=False)),
    ("add_holiday", FunctionTransformer(_add_holiday_indicator, validate=False)),
    ("drop_low_value_features", FunctionTransformer(_drop_low_value_features, validate=False)),
    ("add_site_avg_log_bike_count", FunctionTransformer(_add_site_avg_log_bike_count, validate=False)),
    ("add_daily_variance", FunctionTransformer(_add_daily_variance, validate=False)),
])