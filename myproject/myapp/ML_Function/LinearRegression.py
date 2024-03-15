import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf 

# For Machine Learning Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import datetime

def LinearRegressionModel(stockName, colName, endDate, futureDays):
    # Get large stock data
    stockDataFrame = yf.download(stockName, start='2019-01-01', end=endDate)
    # Generate future dates
    future_days = futureDays
    last_date = pd.to_datetime(stockDataFrame.index[-1])
    future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, future_days + 1)]
    
    # Get Stock Data for a single column from stockData
    colData = stockDataFrame.filter([colName])
    # Convert data into numpy array
    npDataset = colData.values
    # Getting the number of rows to train the model on
    # training_data_len = int(np.ceil(len(npDataset) * .99))
    training_data_len = len(npDataset)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(npDataset)
    
    # Creating the scaled training data set
    train_Data = scaled_data[0:int(training_data_len), :]
    # Splitting the data into x_train and y_train data sets
    x_train = []
    y_train = []
    
    for i in range(360, len(train_Data)):
        x_train.append(train_Data[i - 360:i, 0])
        y_train.append(train_Data[i, 0])
    
    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Reshaping the data back to 2D for Linear Regression
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    
    # Creating Linear Regression Model
    model = LinearRegression()
    
    # Training the model
    model.fit(x_train, y_train)
    
    # Prepare the test data to predict future values
    full_dataset = scaled_data
    model_inputs = full_dataset[-360:].reshape(1, -1)
    model_inputs = np.array(model_inputs)
    model_inputs = np.reshape(model_inputs, (model_inputs.shape[1],))

    # Predict future values
    future_predictions = []
    for _ in range(future_days):
        x = model_inputs[-360:]
        x = x.reshape((1, 360))
        pred = model.predict(x)
        future_predictions.append(pred[0])
        model_inputs = np.append(model_inputs, pred[0])

    # Inverse transform to get real values
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten().tolist()