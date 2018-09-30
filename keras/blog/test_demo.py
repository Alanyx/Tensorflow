"""
Created on 2018/9/6
@author: AlanYx
"""

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils.vis_utils import plot_model

import numpy as np


if __name__ == '__main__':

    data_dim = 16
    timesteps = 8
    num_classes = 10

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # Generate dummy training data
    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = np.random.random((1000, num_classes))

    # Generate dummy validation data
    x_val = np.random.random((100, timesteps, data_dim))
    y_val = np.random.random((100, num_classes))

    model.fit(x_train, y_train,
              batch_size=64, epochs=1,
              validation_data=(x_val, y_val))

    # 用于绘图
    plot_model(model, to_file='model1.png', show_shapes=True)

