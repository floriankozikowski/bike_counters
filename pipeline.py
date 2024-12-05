import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

def _encode_dates(X):
    return X

def _add_rush_hour_indicator(X):
    return X

def _add_weekend_indicator(X):
    return X

def _add_holiday_indicator(X):
    return X

def _drop_low_value_features(X):
    return X

def _add_site_avg_log_bike_count(X):
    return X

def _add_daily_variance(X):
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