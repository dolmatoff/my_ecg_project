import numpy as np
import tensorflow.keras
import pickle
import sys
import argparse
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers, regularizers
import tensorflow as tf


def model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape, saved_model_path) :
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

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    epochs = 20

    x_train, x_test, x_valid = map(lambda x: get_transformed_input(x), [x_train, x_test, x_valid])

    x = Input(name='the_input', shape=input_shape, dtype='float32')

    c1 = L.Conv2D(4, (4, 4), activation='relu', padding='same')(x) #8
    p1 = L.MaxPooling2D((2, 2))(c1)

    c2 = L.Conv2D(8, (2, 2), activation='relu', padding='same')(c1) #13
    p2 = L.MaxPooling2D((2, 2))(c2)

    c3 = L.Conv2D(8, (2, 2), activation='relu', padding='same')(c2) #13
    p3 = L.MaxPooling2D((2, 2))(c3)

    fl = L.Flatten()(p3)
    dr = L.Dropout(0.45)(fl)
    dense = L.Dense(numclasses*1.5, activation = 'relu')(dr)

    output = L.Dense(numclasses, activation='softmax')(dense)

    model = Model(inputs=[x], outputs=[output], name='ConvSpeechModel')

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    callbacks = [ModelCheckpoint(filepath=saved_model_path, monitor='categorical_crossentropy')]

    history = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=epochs, 
              verbose=1, 
              batch_size=8,
              callbacks = callbacks)

    return model, history, x_valid


def get_transformed_input(x):
    return np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))