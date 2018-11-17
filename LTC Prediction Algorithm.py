import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import quandl
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

# function to retrieve data from quandl database
def get_quandl_data(quandl_id):
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df

# Getting the ltc_usd price
ltc_usd_bitfinex = get_quandl_data('BITFINEX/LTCUSD')
ltc_usd_bitfinex = ltc_usd_bitfinex

# Plotting the closing price
plt.figure(figsize = (12,8))
plt.plot(ltc_usd_bitfinex['Last'], linewidth = 2.5, alpha = 0.8, color = 'g')
plt.title('LTC/USD, Raw Data')
plt.ylabel('Price [$]')
plt.xlabel('Time')
plt.grid()
plt.show()

# Extracting ltc closing price
ltc_close_price = ltc_usd_bitfinex["Last"].values

# We want to predict 30 days into the future
forecast_out = int(10)
ltc_usd_bitfinex['Prediction'] = ltc_usd_bitfinex['Last'].shift(-forecast_out)

# Machine Learning Prediction Sequence
X = np.array(ltc_usd_bitfinex.drop(['Prediction'], 1))
X = preprocessing.scale(X)
X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X
y = np.array(ltc_usd_bitfinex['Prediction'])
y = y[:-forecast_out]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.3)
# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)
forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)

# Plotting the forecast
plt.figure(figsize = (12, 8))
plt.plot(forecast_prediction, linewidth = 2.5, alpha = 0.8, color = 'r')
plt.xlabel('Days')
plt.ylabel('Price [$]')
plt.title('30 Day LTC Forecast 93% Confidence Linear Regression')
plt.grid()
plt.show()







