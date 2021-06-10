import numpy as np
import tensorflow.keras
import pickle
import sys
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization, Activation, Bidirectional, LSTM, Convolution1D, MaxPooling1D
from tensorflow.keras import optimizers, regularizers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint


def _inception_module(input_tensor, stride=1, activation='linear'):
        bottleneck_size = 32
        nb_filters = 32
        kernel_size = 41

        input_inception = tensorflow.keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)

        #kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(tensorflow.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, 
                                                 use_bias=False)(input_inception))

        max_pool_1 = tensorflow.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = tensorflow.keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = tensorflow.keras.layers.Concatenate(axis=2)(conv_list)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        x = tensorflow.keras.layers.Activation(activation='relu')(x)
        return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = tensorflow.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias=False)(input_tensor)
    
    shortcut_y = tensorflow.keras.layers.BatchNormalization()(shortcut_y)

    x = tensorflow.keras.layers.Add()([shortcut_y, out_tensor])
    x = tensorflow.keras.layers.Activation('relu')(x)
    return x

def model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape, saved_model_path):
    '''
    load data, compile and train Inception model, apply data shape trasformation for ANN inputs
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
    epochs = 100
    input_layer = tensorflow.keras.layers.Input(input_shape)

    x_train, x_test, x_valid = map(lambda x: get_transformed_input(x), [x_train, x_test, x_valid])

    x = input_layer
    input_res = input_layer
    depth = 6

    for d in range(depth):

        x = _inception_module(input_tensor=x)

        if d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tensorflow.keras.layers.GlobalAveragePooling1D()(x)

    output_layer = tensorflow.keras.layers.Dense(numclasses, activation='softmax')(gap_layer)

    model = tensorflow.keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(),
                      metrics=['accuracy'])

    print(model.summary())
    #add callbacks
    reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
    callbacks = [ModelCheckpoint(filepath=saved_model_path, monitor='categorical_crossentropy'), reduce_lr]

    history = model.fit(x_train, y_train, batch_size=64, validation_data=(x_test, y_test), epochs=epochs, verbose=1, callbacks = callbacks)
    
    return model, history, x_valid


def get_transformed_input(x):
    return np.reshape(x, (x.shape[0], x.shape[1], 1))