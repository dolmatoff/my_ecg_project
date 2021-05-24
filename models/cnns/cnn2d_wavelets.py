import numpy as np
import keras
import pickle
import sys
import argparse
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras import optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint


def model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape):
    '''
    load data, compile and train CNN model (LeNet-5 architecture), apply data shape trasformation for ANN inputs
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
    epochs = 54
    batch_size = 368

    x_train, x_test, x_valid = map(lambda x: get_transformed_input(x), [x_train, x_test, x_valid])
    
    model = Sequential()
    
    # 2 Convolution layer with Max polling
    model.add(Conv2D(32, 5, activation = "relu", padding = 'same', input_shape = input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 5, activation = "relu", padding = 'same', kernel_initializer = "he_normal"))
    model.add(MaxPooling2D())  
    model.add(Flatten())
    
    # 3 Full connected layer
    model.add(Dense(numclasses*1.25, activation = "relu", kernel_initializer = "he_normal"))
    model.add(Dense(numclasses*0.85, activation = "relu", kernel_initializer = "he_normal"))
    model.add(Dense(numclasses, activation = 'softmax'))
    
    # summarize the model
    print(model.summary())

    # compile the model
    model.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    # define callbacks
    callbacks = [ModelCheckpoint(filepath='saved_models/wavelets_based.h5', monitor='categorical_crossentropy')]
    
    # fit the model
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(x_test, y_test))

    return model, history, x_valid

def get_transformed_input(x):
    return np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))
