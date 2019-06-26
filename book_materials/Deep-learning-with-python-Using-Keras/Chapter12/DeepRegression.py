# Regression Example With Boston Dataset: Baseline

import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# load dataset
dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]


# # define base model
# model = Sequential()
# model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
# # model.add(Dense(6, init='normal', activation='relu'))
# # model.add(Dense(6, init='normal', activation='relu'))
# # model.add(Dense(6, init='normal', activation='relu'))
# model.add(Dense(1, init='normal'))


# define base model
model = Sequential()
model.add(Dense(13, input_dim=13, init='normal' , activation='relu'))
# model.add(Dense(6, init='normal' , activation='relu'))
model.add(Dense(1, init='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=100, batch_size=5)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Y_hat
Yhat = model.predict(X)

# plot baseline and predictions
plt.plot(Y)
plt.plot(Yhat)
plt.show()