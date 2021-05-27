import numpy as np
import tensorflow.keras
import pickle
import sys
import argparse

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from data_processing.processing import *
from models.cnns import cnn1d, cnn2d, cnn2d_2, cnn2d_wavelets
from models.recurrent import lstm1d, lstm1d_2
from models.predefined import inception1d, resnet1d

#GLOBALS
# numerical constants
cpsc_num = 1145
ptbxl_num = 6480


def parse_args():
    """
        Gets all configurations to set all parameters and proceed
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='cpsc2018/', help='Preprocessed data directory')
    parser.add_argument('--data-1d-file', type=str, default='cpsc2018/cpsc_1145_25.pkl', help='1d data filename')
    parser.add_argument('--data-2d-file', type=str, default='cpsc2018/cpsc_1145_25_cwt.pkl', help='2d data filename')
    parser.add_argument('--data-prefix', type=str, default='cwt', help='Prefix to distiguish between the sets of different dimensionality') #cwt
    #parser.add_argument('--data-1d-file', type=str, default='ptb_xl_data/ptb_xl_6480_14.pkl', help='Data directory')
    #parser.add_argument('--data-2d-file', type=str, default='ptb_xl_data/ptb_xl_6480_14_spectrograms.pkl', help='Data directory')
    
    parser.add_argument('--model-1d-name', type=str, default='resnet1d', help='Model name')
    parser.add_argument('--model-2d-name', type=str, default='cnn2d_wavelets', help='Model name')
    parser.add_argument('--dimension', type=int, default='2', help='Data dimension')
    return parser.parse_args()

def get_based_parameters() :
    """
        Returns 6 parameters: 
            dataset file
            path to the dataset
            number of classes
            the model name
            the saved model path
            the data prefix
    """
    args = parse_args()
    data = args.data_1d_file if (args.dimension == 1) else args.data_2d_file
    isCPSC = True if('cpsc' in data) else False
    model_name = args.model_1d_name if (args.dimension == 1) else args.model_2d_name
    numclasses = cpsc_num if isCPSC else ptbxl_num
    saved_model_path = 'saved_models/' + ('cpsc_' if isCPSC else 'ptbxl_') + model_name + '.h5'
    return data, args.data_path, numclasses, model_name, saved_model_path, args.data_prefix


if __name__ == '__main__':

    args = parse_args()
    # assign parameters
    data, path, numclasses, model_name, saved_model_path, prefix = get_based_parameters()
    
    isCPSC = True if('cpsc' in data) else False
    saved_hist_path = 'saved_history/' + ('cpsc_' if isCPSC else 'ptbxl_') + model_name

    # load the data and split into train/test/validation sets
    x_train, x_test, x_valid, y_train, y_test, y_valid = load_train_test_data(path, args.data_prefix, data)

    y_train = to_categorical(y_train, numclasses)
    y_test = to_categorical(y_test, numclasses)
    y_valid_ = to_categorical(y_valid, numclasses)

    input_shape = ((x_train.shape[1], x_train.shape[2], 1) if (args.dimension == 2) else (x_train.shape[1], 1))
    # start training
    model, history, x_valid = locals()[model_name].model_fit(x_train, y_train, x_test, y_test, x_valid, numclasses, \
        input_shape, saved_model_path)
    
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
    plt.savefig('confusion_matrix_' + ('cpsc_' if isCPSC else 'ptbxl_') + model_name + '.jpg')

    # save the history
    with open(saved_hist_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
