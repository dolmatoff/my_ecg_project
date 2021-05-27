import numpy as np
import keras
import pickle
import sys

from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.pyplot import imread, imshow, subplots, show
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from keras import layers as L
from keras.models import Model, Input
from keras import optimizers, regularizers
from keras.utils import np_utils
from sklearn.preprocessing import normalize
from keras.layers.merge import add, concatenate



def model_fit(x_train, y_train, x_test, y_test, epochs, numclasses):

    input_shape = (26, 26, 1)

    x = keras.layers.Input(input_shape)
#    drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv2D(128, 8, 1, padding='same')(x)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)
    
#    drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv2D(256, 5, 1, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    
#    drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    
    full = keras.layers.GlobalAveragePooling2D()(conv3)
    out = keras.layers.Dense(numclasses, activation='softmax')(full)

    model = keras.models.Model(inputs=x, outputs=out)

    model.summary()

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                      patience=50, min_lr=0.0001) 

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=epochs, verbose=1, callbacks = [reduce_lr])

    return history, model


if __name__ == '__main__':

    filename = 'processed_data/processed_data_spectrograms_25.pkl'
    epochs = 250
    evaluation_result_path = 'model_crnn2d_results.txt'
    numclasses = 1145

    x_train, x_test, x_valid, y_train, y_test, y_valid = load_data_2d(filename)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    y_train = np_utils.to_categorical(y_train, numclasses)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    y_test = np_utils.to_categorical(y_test, numclasses)

    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1))
    y_valid_ = np_utils.to_categorical(y_valid, numclasses)

    history, model = model_fit(x_train, y_train, x_test, y_test, epochs, numclasses)
    model.save('models/crnn2d_model_new_.h5')
    #model = keras.models.load_model('models/lstm1d_model_new.h5')

    # evaluate model on validation data
    print('Evaluation...')
    score = model.evaluate(x_valid, y_valid_, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # predict classes
    y_predict = model.predict(x_valid) 
    y_predict = y_predict.argmax(axis=-1)
    cf = confusion_matrix(y_valid, y_predict)

    #fig = plt.figure()
    #plt.matshow(cf)
    #plt.title(' Confusion Matrix ECG Recognition ')
    #plt.colorbar()
    #plt.ylabel('True Label')
    #plt.xlabel('Predicated Label')
    #plt.savefig('confusion_matrix_lstm1d_new_.jpg')

    report = classification_report(y_valid, y_predict)
    print(report)

    # Print history
    print(history.history.keys())

    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()