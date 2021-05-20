import numpy as np
import keras
import pickle
import sys
import argparse
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, LSTM, Convolution2D, GRU
from keras import optimizers, regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint

# LeNet-5 architecture
def model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape):
    '''
    load training data and testing data, compile and train CNN model, return training history
    Parameters
    Input: train_generator, test_generator
    epochs: number of epochs for training
    Output: training history parameters

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
    callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='categorical_crossentropy')]
    
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
