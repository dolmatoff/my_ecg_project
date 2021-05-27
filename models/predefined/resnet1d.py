import numpy as np
import tensorflow.keras as keras
import pickle
import sys
from tensorflow.keras import optimizers, regularizers
from models.predefined.ResNetModel1d import build_resnet
from tensorflow.keras.callbacks import ModelCheckpoint


def model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape, saved_model_path):
    '''
    load data, compile and train ResNet1d model, apply data shape trasformation for ANN inputs
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

    x_train, x_test, x_valid = map(lambda x: get_transformed_input(x), [x_train, x_test, x_valid])

    batch_size = min(x_train.shape[0]/10, 16)

    x, y = build_resnet(x_train.shape[1:], 64, numclasses)

    model = keras.models.Model(inputs=x, outputs=y)

    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # define callbacks
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                      patience=50, min_lr=0.0001) 
    callbacks = [ModelCheckpoint(filepath=saved_model_path, monitor='categorical_crossentropy'), reduce_lr]
    # train model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test), callbacks = callbacks)

    return model, history, x_valid



def get_transformed_input(x):
    return np.reshape(x, (x.shape[0], x.shape[1], 1, 1))


