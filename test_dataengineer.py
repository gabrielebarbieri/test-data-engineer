import pandas as pd
from os.path import join
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor


def load_weather_data(weather_data_path):
    """
    Load weather data from a .csv file. In practice only keep temperature data
    :param weather_data_path: path of the file containing the weather data
    :return: A pandas dataframe containing temperature data indexed by
    timestamp
    """
    df = pd.read_csv(weather_data_path, sep=';', parse_dates=['tms_gmt'])
    return df[['tms_gmt', 'temperature']].groupby('tms_gmt').mean()


def load_bike_data(bike_data_path):
    """
    Load bike data from a .csv file.
    :param bike_data_path: path of the file containing the bikes data
    :return: A pandas dataframe containing the number of available bikes
    in the whole system at a certain timestamp
    """
    df = pd.read_csv(bike_data_path, sep=';', parse_dates=['tms_gmt'])
    bikes_df = df[['tms_gmt', 'bikes']].groupby('tms_gmt').sum()
    return bikes_df.resample('15Min').median()


def load_parking_data(parking_data_path):
    """
    Load parking data from a .csv file.
    :param parking_data_path: path of the file containing the parking data
    :return: A pandas dataframe containing the number of available off-street
    parking in the whole city at a certain timestamp
    """
    df = pd.read_csv(parking_data_path, sep=';', parse_dates=['tms_gmt'])
    parking_df = df[['tms_gmt', 'free_slots']].groupby('tms_gmt').sum()
    parking_df.rename(columns={'free_slots': 'parkings'}, inplace=True)
    return parking_df.resample('15Min').median()


def load_traffic_data(traffic_data_path):
    """
    Load traffic data from a .csv file.
    :param traffic_data_path: path of the file containing the traffic data
    :return: A pandas dataframe containing the average traffic status
    in the whole city at a certain timestamp
    """
    df = pd.read_csv(traffic_data_path, sep=';', parse_dates=['tms_gmt'])
    traffic_df = df[['tms_gmt', 'status']].groupby('tms_gmt').mean()
    traffic_df.rename(columns={'status': 'traffic'}, inplace=True)
    return traffic_df.resample('15Min').mean()


def train_bike_model(data_folder):
    """
    Train a model that predicts the number of bikes based on temperature
    :param data_folder: the folder containing all the data needed to train the
    model
    :return: A Scikit Learn Lasso model that takes temperature as input and
    predict the number of bikes in the system
    """
    weather_df = load_weather_data(
        join(data_folder, 'bordeaux_fr_weather.csv.bz2'))
    bike_df = load_bike_data(
        join(data_folder, 'bordeaux_fr_bikeshare.csv.bz2'))
    df = bike_df.reset_index().merge(weather_df.reset_index())
    clf = Lasso()
    clf.fit(df[['temperature']], df.bikes)
    return clf


def train_parking_model(data_folder):
    """
    Train a model that predicts the number of available off-street parking in
    the whole city based on temperature and traffic status
    :param data_folder: the folder containing all the data needed to train the
    model
    :return: A Scikit Learn Lasso model that takes temperature and traffic
    status as input and predict the number of available off-street parking
    """
    weather_df = load_weather_data(
        join(data_folder, 'bordeaux_fr_weather.csv.bz2'))
    traffic_df = load_traffic_data(
        join(data_folder, 'bordeaux_fr_traffic.csv.bz2'))
    parking_df = load_parking_data(
        join(data_folder, 'bordeaux_fr_parkings.csv.bz2'))
    df = parking_df.reset_index()\
        .merge(weather_df.reset_index())\
        .merge(traffic_df.reset_index())
    clf = KNeighborsRegressor()
    clf.fit(df[['temperature', 'traffic']], df.parkings)
    return clf


if __name__ == '__main__':
    print 'train the bike model...'
    bike_model = train_bike_model('./data')
    print 'bike model trained'
    print 'test the bike model on some toy data:'
    test_bike_data = pd.DataFrame({'temperature': range(30)})
    bike_predictions = bike_model.predict(test_bike_data)
    print bike_predictions

    print

    print 'train the parking model...'
    parking_model = train_parking_model('./data')
    print 'parking model trained'
    print 'test the parking model on some toy data:'
    test_parking_data = pd.DataFrame({'temperature': range(30),
                                      'traffic': 0.3})
    parking_predictions = parking_model.predict(test_parking_data)
    print parking_predictions
