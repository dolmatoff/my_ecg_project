from keras.models import load_model
import numpy as np
import wfdb
import sys
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

from data_processing.processing import *
from models.cnns import cnn1d, cnn2d, cnn2d_2, cnn2d_wavelets
from models.recurrent import lstm1d, lstm1d_2
from models.predefined import inception1d, resnet1d


if __name__ == '__main__':
    
    model_file = 'saved_models/wavelets_based.h5'
    signal_file = 'cpsc2018/cpsc_1145_25_cwt.pkl'
    signal_1d_file = 'cpsc2018/cpsc_1145_25.pkl'
    numclasses = 1145 #6480
    model_name = 'cnn2d_wavelets'
    samples_per_class = 25

    model = load_model(model_file)
    x_train, x_test, x_valid, y_train, y_test, y_valid = load_train_test_data(signal_file, True)

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

    #diff = np.unique(list((counter(y_predict.flatten()) - counter(y_valid.flatten())).elements()))
    #print(diff)
    #x, y = load_preprocessed_1d_data(signal_1d_file)
    #misclassified_y = [ i for i, x in enumerate(y) if x in diff ]
    #misclassified_x = x[misclassified_y]
    #chunks = np.array_split(misclassified_x, samples_per_class)
    #for ch in chunks :
    #    plt.plot(np.linspace(0, 199, 200), ch.t)
    #    plt.show()

    cf = confusion_matrix(y_valid, y_predict)

    ###########################################################################################################################
    TP = np.diag(cf)
    TP_SUM = TP.sum()

    FN_FP = np.fromiter(map(lambda x, y: x - y, np.apply_along_axis(sum, axis=0, arr=cf), TP), dtype=np.int)

    FN_SUM = FN_FP.sum()
    FP_SUM = FN_FP.sum()

    TN_SUM = y_predict.shape[0] - TP_SUM - FN_SUM

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP_SUM/(TP_SUM + FN_SUM)
    # Specificity or true negative rate
    TNR = TN_SUM/(TN_SUM + FP_SUM) 
    # Precision or positive predictive value
    PPV = TP_SUM/(TP_SUM + FP_SUM)
    # Negative predictive value
    NPV = TN_SUM/(TN_SUM + FN_SUM)
    # Fall out or false positive rate
    FPR = FP_SUM/(FP_SUM + TN_SUM)
    # False negative rate
    FNR = FN_SUM/(TP_SUM + FN_SUM)
    # False discovery rate
    FDR = FP_SUM/(TP_SUM + FP_SUM)

    # Overall accuracy
    ACC = (TP_SUM + TN_SUM)/(TP_SUM + FP_SUM + TN_SUM)

    print('TP   TN    FP    FN')
    print((TP_SUM, TN_SUM, FP_SUM, FN_SUM))
    print('Accuracy:')
    print(ACC)
    ##########################################################################################################################

    plt.close()
    fig = plt.figure()
    plt.matshow(cf)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig('confusion_matrix_' + model_name + '.png', dpi=500)
