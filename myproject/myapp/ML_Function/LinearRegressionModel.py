import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf 

# For Machine Learning Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def LinearRegressionModel(stockName, colName, endDate):
    #Get large stock data
    stockDataFrame = yf.download(stockName, start='2019-01-01', end=endDate)
    #Get Stock Data for a single column from stockData
    colData = stockDataFrame.filter([colName])
    #convert data into numpy array
    npDataset = colData.values
    #getting the number of row to train the model on
    training_data_len = int(np.ceil(len(npDataset) * .95))

    #scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(npDataset)

    #creating the scaled training data set
    train_Data = scaled_data[0:int(training_data_len), :]
    #spliting the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_Data)):
        x_train.append(train_Data[i - 60:i,0])
        y_train.append(train_Data[i,0])
        if i<=61:
            print()
    
    #convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #reshaping the data
    x_train = np.reshape(x_train, (x_train.shape[0], -1))

    #creating Linear Regression Model
    model = LinearRegression()

    #training the model
    model.fit(x_train, y_train)

    #creating the test datasets
    test_data = scaled_data[training_data_len-60: , :]
    #creating new x_test and y_test
    x_test=[]
    y_test= npDataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i,0])
    
    #convert the data to numpy array
    x_test = np.array(x_test)
    
    #reshaping the data
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    #getting the model to predict price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1,1))

    #getting the root mean squared error (RMSE)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(predictions, y_test))
    print(rmse)

    #plotting the results
    train = colData[:training_data_len]
    valid = colData[training_data_len:]
    valid['Predicitions'] = predictions
    # print(predictions)
    return predictions.flatten().tolist()



