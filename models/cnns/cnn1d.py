import numpy as np
import tensorflow.keras
import pickle
import sys
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization, Activation
from tensorflow.keras import optimizers, regularizers


def model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape, saved_model_path):
    '''
    load data, compile and train CNN model, apply data shape trasformation for ANN inputs
    Parameters
    Input: 
        x_train, y_train - train data: qrs segments and labels
        y_test, y_test - test data: qrs segments and labels
        x_valid - validation data
        numclasses - the number of classes (labels)
        input_shape - the unput shape of the chosen ANN
    Output: 
        model - sequential model
        history - training history parameters
        x_valid - reshaped validation data
    '''

    x_train, x_test, x_valid = map(lambda x: get_transformed_input(x), [x_train, x_test, x_valid])

    epochs = 100

    model = Sequential()

    # Convolutional layers
    model.add(Convolution1D(100, 4, 1, activation='tanh', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Convolution1D(200, 2, 1, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Convolution1D(300, 1, 1, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Convolution1D(400, 1, 1, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))

    model.add(Flatten())
    model.add(Dropout(0.9))
    model.add(Dense(3000, activation='tanh'))
    model.add(Dense(numclasses, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
    callbacks = [ModelCheckpoint(filepath=saved_model_path, monitor='categorical_crossentropy'), reduce_lr]

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=1, callbacks = callbacks)
    
    return model, history, x_valid


def get_transformed_input(x):
    return np.reshape(x, (x.shape[0], x.shape[1], 1))