import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.cluster import  MiniBatchKMeans
import math
random_state=42

def feature_creation(train):
    # Function to calculate the Haversine distance between two points
    def haversine_array(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371  # in km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    train['distance_haversine'] = haversine_array(train['pickup_latitude'].values,
                                                  train['pickup_longitude'].values,
                                                  train['dropoff_latitude'].values,
                                                  train['dropoff_longitude'].values)

    # Function to calculate the bearing between two points
    def bearing_array(lat1, lng1, lat2, lng2):
        AVG_EARTH_RADIUS = 6371  # in km
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))

    train['bearing'] = bearing_array(train['pickup_latitude'].values,
                                     train['pickup_longitude'].values,
                                     train['dropoff_latitude'].values,
                                     train['dropoff_longitude'].values)

    # Function to calculate a dummy Manhattan distance
    def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
        a = haversine_array(lat1, lng1, lat1, lng2)
        b = haversine_array(lat1, lng1, lat2, lng1)
        return a + b

    train['distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values,
                                                                 train['pickup_longitude'].values,
                                                                 train['dropoff_latitude'].values,
                                                                 train['dropoff_longitude'].values)

    train['distance_haversine'] = np.log1p(train.distance_haversine)
    train['distance_dummy_manhattan'] = np.log1p(train.distance_dummy_manhattan)
    train['log_trip_duration'] = np.log1p(train.trip_duration)

    train.drop(columns=['trip_duration', 'pickup_datetime'], inplace=True)
    return train





def feature_extraction(train):
    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['DayofMonth'] = train['pickup_datetime'].dt.day
    train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
    train['month'] = train['pickup_datetime'].dt.month
    train['hour'] = train['pickup_datetime'].dt.hour
    train['dayofyear'] = train['pickup_datetime'].dt.dayofyear

    return train



def Load_and_prepare_data(path):
    df=pd.read_csv(path)
    df = feature_extraction(df)
    df = feature_creation(df)
    return df



def preprocessor(option):
   pre={
       1: MinMaxScaler(),
       2: StandardScaler(),
   }
   return pre[option]


def Get_preprocessor(option, degree):
    numeric_features,categorical_features,features=Define_Features()

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('poly', PolynomialFeatures(degree=degree),numeric_features),
        ('scaling', preprocessor(option), numeric_features)
        ]
        , remainder = 'passthrough'
    )
    return column_transformer



def Define_Features():
    numeric_features = ['distance_haversine', 'distance_dummy_manhattan', 'bearing']
    categorical_features = ['passenger_count', 'vendor_id','DayofMonth', 'dayofweek', 'month', 'hour', 'dayofyear']

    # Combine features
    features = numeric_features + categorical_features

    return numeric_features,categorical_features,features


