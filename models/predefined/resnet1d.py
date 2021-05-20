import numpy as np
import keras
import pickle
import sys
from keras import optimizers, regularizers
from models.predefined.ResNetModel1d import build_resnet


def model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape):

    epochs = 100

    x_train, x_test, x_valid = map(lambda x: get_transformed_input(x), [x_train, x_test, x_valid])

    batch_size = min(x_train.shape[0]/10, 16)

    x, y = build_resnet(x_train.shape[1:], 64, numclasses)

    model = keras.models.Model(inputs=x, outputs=y)

    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
      
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                      patience=50, min_lr=0.0001) 
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test), callbacks = [reduce_lr])

    return model, history, x_valid



def get_transformed_input(x):
    return np.reshape(x, (x.shape[0], x.shape[1], 1, 1))


