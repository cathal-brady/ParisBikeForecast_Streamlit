import os
import joblib
import numpy as np
import pandas as pd
import holidays
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from datetime import datetime as dt
from tqdm import tqdm

# Path to site coordinates file
site_coordinates_path = 'data/site_coordinates.csv'

# Define a class for feature engineering
class DateFormatter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
class DateFormatter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['date'] = pd.to_datetime(X_copy['Date and Time'], utc=True)
        X_copy['month'] = X_copy['date'].dt.month
        X_copy['weekday'] = X_copy['date'].dt.dayofweek + 1
        X_copy['hr'] = X_copy['date'].dt.hour
        X_copy['hr_sin'] = np.sin(X_copy.hr * (2. * np.pi / 24))
        X_copy['hr_cos'] = np.cos(X_copy.hr * (2. * np.pi / 24))
        return X_copy

class AddRestrictionLevel(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        restriction_dates = [
            ('2020-10-17', '2020-11-28', 5),
            ('2020-11-28', '2020-12-15', 4),
            ('2020-12-15', '2021-01-16', 2),
            ('2021-01-16', '2021-03-19', 1),
            ('2021-03-19', '2021-06-09', 5),
        ]
        X_copy['restriction_level'] = 0
        for start, end, level in restriction_dates:
            mask = (X_copy['date'] >= start) & (X_copy['date'] < end)
            X_copy.loc[mask, 'restriction_level'] = level
        return X_copy

class HolidaysFR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fr_holidays = holidays.FR()
        X_copy = X.copy()
        X_copy['is_holiday'] = X_copy['date'].apply(lambda x: 1 if x in fr_holidays else 0)
        X_copy['is_weekend'] = X_copy['weekday'].apply(lambda x: 1 if x >= 6 else 0)
        return X_copy

class RushHour(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['rush_hour'] = ((X_copy['weekday'] <= 5) &
                               ((X_copy['hr'].between(7, 9)) | (X_copy['hr'].between(17, 20))) &
                               (X_copy['is_holiday'] == 0)).astype(int)
        return X_copy

class SiteNameMapper(BaseEstimator, TransformerMixin):
    def __init__(self, site_coordinates_path='data/site_coordinates.csv'):
        self.site_coordinates_path = site_coordinates_path
        
    def fit(self, X, y=None):
        # Load site coordinates to map site_name
        self.site_map = pd.read_csv(self.site_coordinates_path)[['Site Name']]
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Map site_name directly
        X_copy = pd.merge(X_copy, self.site_map, left_on='Site Name', right_on='Site Name', how='left')
        return X_copy
    
    def get_site_names(self):
        return self.site_names

# The final model pipeline
class TrainCatBoostModel:
    def __init__(self, model_save_dir='pretrained_catboost_models/'):
        self.model_save_dir = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    def fit(self, X, y, site_name):
        model = CatBoostRegressor(
            loss_function='RMSE', depth=10, iterations=200, learning_rate=0.1, verbose=False
        )
        model.fit(X, y)
        
        model_filename = f"{site_name}_catboost_model.joblib"
        model_path = os.path.join(self.model_save_dir, model_filename)
        joblib.dump(model, model_path)

    def load_model(self, site_name):
        model_filename = f"{site_name}_catboost_model.joblib"
        model_path = os.path.join(self.model_save_dir, model_filename)
        return joblib.load(model_path)

    def predict(self, X, site_name):
        model = self.load_model(site_name)
        return model.predict(X)

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('date_formatter', DateFormatter()),
    ('add_restriction_level', AddRestrictionLevel()),
    ('holidays_fr', HolidaysFR()),
    ('rush_hour', RushHour()),
    ('site_name_mapper', SiteNameMapper()),
])

def main(train_data_path='data/clean_count_data.csv'):
    # Load training data
    train_df = pd.read_csv(train_data_path)

    # Apply the DateFormatter to convert the 'Date and Time' column
    date_formatter = DateFormatter()
    train_df = date_formatter.fit_transform(train_df)

    # Compute log of bike count (Hourly Count)
    train_df['log_count'] = np.log(1+train_df['Hourly Count'])
    train_df.drop('Hourly Count', axis=1, inplace=True)

    # Preprocess the data
    X = preprocessing_pipeline.fit_transform(train_df)
    y = X.pop('log_count')
    
    # Get unique site names
    unique_sites = X['Site Name'].unique()

    # Train and save the model for each site with progress bar
    model_trainer = TrainCatBoostModel()
    for site in tqdm(unique_sites, desc="Training models", unit="site"):
        site_data = X[X['Site Name'] == site]
        site_target = y[X['Site Name'] == site]
        site_data.drop(['Site Name', 'Date and Time', 'date'], axis=1, inplace=True)
        print(site_data.columns)

if __name__ == "__main__":
    main()
