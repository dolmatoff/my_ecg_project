from keras.models import load_model
import numpy as np
import wfdb
import sys
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report

from data_processing.processing import *
from models.cnns import cnn1d, cnn2d, cnn2d_2
from models.recurrent import lstm1d, lstm1d_2
from models.predefined import inseption1d, resnet1d


def predict(model, x):
    '''
    Make prediction of signals
    Input: model, processed signals
    Output: probabities of each class
    '''
    pred_prob = model.predict(x)
    return pred_prob


if __name__ == '__main__':

    model_file = 'saved_models/ptbxl_resnet1d.h5'
    signal_file = 'ptb_xl_data/processed_ptb_xl_.pkl'
    numclasses = 6480
    model_name = 'resnet1d'

    model = load_model(model_file)
    x_train, x_test, x_valid, y_train, y_test, y_valid = load_data(signal_file)

    #prepare data
    y_train = np_utils.to_categorical(y_train, numclasses)
    y_test = np_utils.to_categorical(y_test, numclasses)
    y_valid_ = np_utils.to_categorical(y_valid, numclasses)

    model_space = locals()[model_name]
    x_train, x_test, x_valid = map(lambda x: model_space.get_transformed_input(x), [x_train, x_test, x_valid])

    #load model
    model = load_model(model_file)
    #pred_prob = predict(model, x)

    try:
        y_predict = model.predict_classes(x_valid)
    except:
        y_predict = model.predict(x_valid)
        y_predict = np.argmax(y_predict, axis=1)

    report = classification_report(y_valid, y_predict)
    cf = confusion_matrix(y_valid, y_predict)
    print(report)

