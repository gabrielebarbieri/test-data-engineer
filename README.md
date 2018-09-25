# Data Engineer Test - Proof of Concept
In this repository you'll find the code needed to develop a simple proof of concept for the data engineer test. This proof of concept should demonstrate a simple implementation of the specified service and the model deployment process.

The file [test_dataengineer.py](https://github.com/gabrielebarbieri/test-data-engineer/blob/master/test_dataengineer.py) contains some code to train two simple models:
- A *bike model*, that predicts the number of bikes available in a city based on the temperature.
- A *parking model*, that predicts the number of available off-street parking in a city based on temperature and traffic status.

The package needed by the script are listed in the file [requirements.txt](https://github.com/gabrielebarbieri/test-data-engineer/blob/master/requirements.txt). 

The data needed by the script to properly train the models are located in the folder [data](https://github.com/gabrielebarbieri/test-data-engineer/tree/master/data)

## Installation

Clone the project: 

```
git clone https://github.com/gabrielebarbieri/test-data-engineer.git
cd test-data-engineer
```

(Optional) You can use [virtualenv](https://virtualenv.pypa.io/en/stable/) to isolate you environnement. Please refer to the virtualenv documentation to [install](https://virtualenv.pypa.io/en/stable/installation/) and [use](https://virtualenv.pypa.io/en/stable/userguide/) it. 

Install all the required package:
```python
pip install -r requirements.txt
```
Test that all is working fine by running:
```python
python test_dataengineer.py
```

## Some code from a Data Scientist

The script `test_dataengineer.py` contains two methods that train and output a machine learning model. 
1. The method `train_bike_model` takes the path of the folder containing the data as input and outputs a machine learning model that predicts the number of bikes available in a city based on the temperature.
2. The method `train_parking_model` takes the path of the folder containing the data as input and outputs a machine learning model that predicts that predicts the number of available off-street parking in a city based on temperature and traffic status.

The `main` section of the script calls these methods and shows how to use the trained models. 

## The data 

The data used to train the model come from the following sources:

*Weather*: https://openweathermap.org/api

*Traffic*: http://data.bordeaux-metropole.fr/data.php?layer=CI_TRAFI_L

*Bike occupations*: http://data.bordeaux-metropole.fr/data.php?layer=CI_VCUB_P

*Parkings occupations*: http://data.bordeaux-metropole.fr/data.php?layer=CI_PARK_P

