# -*- coding: utf-8 -*-
"""Keras_tensorflow.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B-E75obIUyhvpLFL4wQDOucP7hlW8bLM
"""

import tensorflow as tf
import pandas as pd
import numpy as np

import seaborn as sns

tf

df = pd.read_csv("/content/fake_reg.csv")

df.head()

df.info()

sns.pairplot(df)

from sklearn.model_selection import train_test_split

X = df[['feature1', 'feature2']].values
y = df[['price']].values

X[1:34]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape

X_test.shape

from sklearn.preprocessing import MinMaxScaler
# help(MinMaxScaler)

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

X_train

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# help(Dense)

model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))

model.add(Dense(1))
model.compile(optimizer="rmsprop", loss='mse')

model.fit(x=X_train,y=y_train, epochs=250)

loss_df = pd.DataFrame(model.history.history )

loss_df

loss_df.plot()

# check how our mdoel is performing

# return back the model loss (mse)
model.evaluate(X_test, y_test, verbose=0)

# check the loss on the training set
model.evaluate(X_train, y_train, verbose=0)

test_predictions = model.predict(X_test)

test_predictions = pd.Series(test_predictions.reshape(300,))

pred_df = pd.DataFrame(y_test, columns=['Test True Y'])

pred_df = pd.concat([pred_df,test_predictions], axis=1)

pred_df.columns = ['Test True Y', 'Model Predictions']

pred_df

sns.scatterplot(x="Test True Y", y="Model Predictions", data=pred_df)

# import the libs to get the erros
from sklearn.metrics import mean_absolute_error, mean_squared_error

# get the mean absolute error of the code
mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions'])

mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])**0.5

#  predicting on brand new data
new_gem = [[988,1000]]

new_gem = scaler.transform(new_gem)

new_predict = model.predict(new_gem)

new_predict

from tensorflow.keras.models import load_model

# saving the model
model.save('my_gem_model.h5')

# loading the model in the work space
later_model = load_model('my_gem_model.h5')

later_model.predict(new_gem)
