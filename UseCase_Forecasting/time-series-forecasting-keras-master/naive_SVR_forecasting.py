from models import SVR

import util
import eval
from sklearn.preprocessing import MinMaxScaler
import numpy as np
np.random.seed(123)


def SVR_forecasting(dataset, lookBack, C=2.0, epsilon=0.01, plot_flag=False):

    # normalize time series
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # divide the series into training/testing samples
    # NOTE: Not RNN format
    train,test = util.divideTrainTest(dataset)

    trainX, trainY = util.createSamples(train, lookBack, RNN=False)
    testX, testY = util.createSamples(test, lookBack, RNN=False)
    print("trainX shape is", trainX.shape)
    print("trainY shape is", trainY.shape)
    print("testX shape is", testX.shape)
    print("testY shape is", testY.shape)

    # buil model and train
    SVRModel = SVR.SVRModel(C=C, epsilon=epsilon)
    SVRModel.train(trainX, trainY)

    # forecasting
    trainPred = SVRModel.predict(trainX).reshape(-1, 1)
    testPred = SVRModel.predict(testX).reshape(-1, 1)

    # reverse the time series
    trainPred = scaler.inverse_transform(trainPred)
    trainY = scaler.inverse_transform(trainY)
    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    # evaluate
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    if plot_flag:
        util.plot(trainPred,trainY,testPred,testY)

    return trainPred,testPred, MAE, MRSE, SMAPE


if __name__ == "__main__":

    lag = 24
    C = 0.1
    epsilon = 0.01

    # ts, data = util.load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/bike_hour.csv", columnName="cnt")
    # ts, data = util.load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = util.load_data("./data/beijing_pm25.csv", columnName="pm2.5")
    ts, data = util.load_data("./data/pollution.csv", columnName="Ozone")
    trainPred, testPred, mae, mrse, smape = SVR_forecasting(data, lookBack=lag, C=C, epsilon=epsilon)

