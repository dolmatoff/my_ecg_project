import numpy as np
import keras
import pickle
import sys
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, LSTM, Convolution2D, GRU
from keras import optimizers, regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2


def model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape):
    '''
    load training data and testing data, compile and train CNN model, return training history
    Parameters
    Input: train_generator, test_generator
    epochs: number of epochs for training
    Output: training history parameters

    '''
    epochs = 550

    x_train, x_test, x_valid = map(lambda x: get_transformed_input(x), [x_train, x_test, x_valid])
    
    model = Sequential()

    model.add(Convolution2D(4, kernel_size=(5, 5), activation='elu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())

    model.add(Convolution2D(8, kernel_size=(3, 2), activation='elu', kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Convolution2D(16, kernel_size=(1, 2), activation='elu', kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(BatchNormalization())
    
    # CNN to RNN
    model.add(Reshape(target_shape=((4, 16)), name='reshape'))
    model.add(Dense(512, activation='elu', name='dense1'))
    model.add(BatchNormalization())

    # RNN layer
    model.add(GRU(512, return_sequences=True, kernel_initializer='he_normal'))

    model.add(Flatten())
    model.add(Dropout(0.85))

    model.add(Dense(1500, activation='sigmoid', name='dense2'))
    model.add(Dense(numclasses, activation='softmax', name='last_dense'))

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=SGD(lr=0.001),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=epochs, verbose=1)

    return model, history, x_valid

def get_transformed_input(x):
    return np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))