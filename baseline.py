import numpy as np
import keras
import pickle
import sys
import argparse

from sklearn.metrics import confusion_matrix, classification_report

from keras.utils import np_utils
import matplotlib.pyplot as plt

from data_processing.processing import *
from models.cnns import cnn1d, cnn2d, cnn2d_2, cnn2d_wavelets
from models.recurrent import lstm1d, lstm1d_2
from models.predefined import inception1d, resnet1d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-1d-file', type=str, default='data/processed_cpsc/cpsc_1431_25.pkl', help='Data directory')
    parser.add_argument('--data-2d-file', type=str, default='data/processed_cpsc/cpsc_1431_25_cwt.pkl', help='Data directory')
    #parser.add_argument('--data-1d-file', type=str, default='ptb_xl_data/ptb_xl_6480_14.pkl', help='Data directory')
    #parser.add_argument('--data-2d-file', type=str, default='ptb_xl_data/ptb_xl_6480_14_spectrograms.pkl', help='Data directory')
    
    parser.add_argument('--model-1d-name', type=str, default='inseption1d', help='Model name')
    parser.add_argument('--model-2d-name', type=str, default='cnn2d_wavelets', help='Model name')
    parser.add_argument('--dimension', type=int, default='2', help='Data dimension')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    
    # numerical constants
    cpsc_num = 1145
    ptbxl_num = 6480

    # assign parameters
    data = args.data_1d_file if (args.dimension == 1) else args.data_2d_file
    isCPSC = True if('cpsc' in data) else False
    model_name = args.model_1d_name if (args.dimension == 1) else args.model_2d_name
    numclasses = cpsc_num if isCPSC else ptbxl_num

    saved_model_path = 'saved_models/' + ('cpsc_' if isCPSC else 'ptbxl_') + model_name + '.h5'
    saved_hist_path = 'saved_history/' + ('cpsc_' if isCPSC else 'ptbxl_') + model_name

    # load the data and split into train/test/validation sets
    x_train, x_test, x_valid, y_train, y_test, y_valid = load_train_test_data(data, isCPSC)

    y_train = np_utils.to_categorical(y_train, numclasses)
    y_test = np_utils.to_categorical(y_test, numclasses)
    y_valid_ = np_utils.to_categorical(y_valid, numclasses)

    input_shape = ((x_train.shape[1], x_train.shape[2], 1) if (args.dimension == 2) else (x_train.shape[1], 1))

    # start training
    model, history, x_valid = locals()[model_name].model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, input_shape)
    # save the model
    model.save(saved_model_path)
    
    # evaluate model on validation data
    print('Evaluation...')
    score = model.evaluate(x_valid, y_valid_, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    try:
        y_predict = model.predict_classes(x_valid)
    except:
        y_predict = model.predict(x_valid)
        y_predict = np.argmax(y_predict, axis=1)

    report = classification_report(y_valid, y_predict)
    cf = confusion_matrix(y_valid, y_predict)
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

    # Confusion matrix
    fig = plt.figure()
    plt.matshow(cf)
    plt.title(' Confusion Matrix ECG Recognition ')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig('confusion_matrix_' + model_name + '.jpg')

    # save the history
    with open(saved_hist_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
