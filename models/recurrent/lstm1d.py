import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM
from tensorflow.keras.layers import BatchNormalization, Activation, Bidirectional, LSTM, Convolution1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger


def model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape, saved_model_path):
    '''
    load data, compile and train Recurrent model, apply data shape trasformation for ANN inputs
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
    epochs = 150

    x_train, x_test, x_valid = map(lambda x: get_transformed_input(x), [x_train, x_test, x_valid])

    model = Sequential()
    model.add(LSTM(2048, return_sequences=True, input_shape=(1, 200)))
    model.add(LSTM(1024, return_sequences=True))
    model.add(LSTM(1024, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))

    #model.add(LSTM(512, return_sequences=True, input_shape=(1, check)))
    #model.add(LSTM(256, return_sequences=True))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(LSTM(64, return_sequences=True))
    #model.add(LSTM(32))

    # Linear classifier
    model.add(Dense(numclasses, activation='softmax'))

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    print(model.summary())

    #stopping = EarlyStopping(monitor='accuracy', patience=3)
    #csv_logger = CSVLogger('training_lstm1d.log', separator=',', append=True)

    reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
    callbacks = [ModelCheckpoint(filepath=saved_model_path, monitor='categorical_crossentropy'), reduce_lr]

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=1, callbacks = callbacks)
    
    return model, history, x_valid


def get_transformed_input(x):
    return np.reshape(x, (x.shape[0], 1, x.shape[1]))
    


