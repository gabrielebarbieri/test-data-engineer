import pandas as pd
from os.path import join
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor


def load_weather_data(weather_data_path):
    df = pd.read_csv(weather_data_path, sep=';', parse_dates=['tms_gmt'])
    return df[['tms_gmt', 'temperature']].groupby('tms_gmt').mean()


def load_bike_data(bike_data_path):
    df = pd.read_csv(bike_data_path, sep=';', parse_dates=['tms_gmt'])
    bikes_df = df[['tms_gmt', 'bikes']].groupby('tms_gmt').sum()
    return bikes_df.resample('15Min').median()


def load_parking_data(parking_data_path):
    df = pd.read_csv(parking_data_path, sep=';', parse_dates=['tms_gmt'])
    parking_df = df[['tms_gmt', 'free_slots']].groupby('tms_gmt').sum()
    parking_df.rename(columns={'free_slots': 'parkings'}, inplace=True)
    return parking_df.resample('15Min').median()


def load_traffic_data(traffic_data_path):
    df = pd.read_csv(traffic_data_path, sep=';', parse_dates=['tms_gmt'])
    traffic_df = df[['tms_gmt', 'status']].groupby('tms_gmt').mean()
    traffic_df.rename(columns={'status': 'traffic'}, inplace=True)
    return traffic_df.resample('15Min').mean()


def train_bike_model(data_path):
    weather_df = load_weather_data(join(data_path, 'bordeaux_fr_weather.csv.bz2'))
    bike_df = load_bike_data(join(data_path, 'bordeaux_fr_bikeshare.csv.bz2'))
    df = bike_df.reset_index().merge(weather_df.reset_index())
    clf = Lasso()
    clf.fit(df[['temperature']], df.bikes)
    return clf


def train_parking_model(data_path):
    weather_df = load_weather_data(join(data_path, 'bordeaux_fr_weather.csv.bz2'))
    traffic_df = load_traffic_data(join(data_path, 'bordeaux_fr_traffic.csv.bz2'))
    parking_df = load_parking_data(join(data_path, 'bordeaux_fr_parkings.csv.bz2'))
    df = parking_df.reset_index().merge(weather_df.reset_index()).merge(traffic_df.reset_index())
    clf = GradientBoostingRegressor()
    clf.fit(df[['temperature', 'traffic']], df.parkings)
    return clf


bike_model = train_bike_model('./data')
test_bike_data = pd.DataFrame({'temperature': range(30)})
print bike_model.predict(test_bike_data)


parking_model = train_parking_model('./data')
test_parking_data = pd.DataFrame({'temperature': range(30), 'traffic': 0.3})
print parking_model.predict(test_parking_data)
