import numpy as np
import tensorflow.keras
import pickle
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, LSTM, Convolution2D, GRU
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint


def model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape, saved_model_path):
    '''
    load data, compile and train CNN model (CRNN architecture), apply data shape trasformation for ANN inputs
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
    model.add(Reshape(target_shape=((72, 16)), name='reshape'))
    model.add(Dense(512, activation='elu', name='dense1'))
    model.add(BatchNormalization())

    # RNN layer
    model.add(GRU(512, return_sequences=True, kernel_initializer='he_normal'))

    model.add(Flatten())
    model.add(Dropout(0.85))

    model.add(Dense(1500, activation='sigmoid', name='dense2'))
    model.add(Dense(numclasses, activation='softmax', name='last_dense'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001),
                  metrics=['accuracy'])

    callbacks = [ModelCheckpoint(filepath=saved_model_path, monitor='categorical_crossentropy')]

    history = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=epochs, 
              verbose=1,
              callbacks = callbacks)

    return model, history, x_valid

def get_transformed_input(x):
    return np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))