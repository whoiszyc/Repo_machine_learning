# Process data
# delete entries that have nan

import numpy as np
import pandas as pd
from scipy.io import loadmat
import csv


# load dataset
trainSet = loadmat('Data/trainSet.mat')
testSet = loadmat('Data/trainSet.mat')

# # split into input (X) and output (Y) variables
trainX = trainSet['trainX']
trainY = trainSet['trainY']
testX = trainSet['trainX']
testY = trainSet['trainY']

# # find row indices that have nan
indices_nan_train = np.isnan(trainX).any(axis=1)
# # delete the corresponding rows in both X and Y
trainX = trainX[~indices_nan_train]
trainY = trainY[~indices_nan_train]

# #  save data to csv
with open('trainX.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(trainX)

