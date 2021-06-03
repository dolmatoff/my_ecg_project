from tensorflow.keras.models import load_model
import numpy as np
import wfdb
import sys
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import time

from data_processing.processing import *
from models.cnns import cnn1d, cnn2d, cnn2d_2, cnn2d_wavelets
from models.recurrent import lstm1d, lstm1d_2
from models.predefined import inception1d, resnet1d
from baseline import parse_args, get_based_parameters
from data_processing.raw_data_processing import samples

if __name__ == '__main__':

    args = parse_args()
    data, path, numclasses, model_name, saved_model_path, prefix = get_based_parameters()
    signal_1d_file = args.data_1d_file #'cpsc2018/cpsc_1145_25.pkl'

    ## Use PTB_XL data as unknown classes for CPSC 2018 dataset
    unknown_file = 'ptb_xl_data/ptb_xl_75_25_cwt.pkl'
    unknown_classes = 75
    unknown_num = unknown_classes * samples

    #load model
    model = load_model(saved_model_path)
    x_train, x_test, x_valid, y_train, y_test, y_valid = load_train_test_data(path, prefix, data)

    #prepare data
    y_train = to_categorical(y_train, numclasses)
    y_test = to_categorical(y_test, numclasses)
    y_valid_ = to_categorical(y_valid, numclasses)

    #ux, uy = load_preprocessed_data(unknown_file)

    # test the classificator
    unknown_idx = random.randint(1, unknown_num)

    model_space = locals()[model_name]
    x_train, x_test, x_valid = map(lambda x: model_space.get_transformed_input(x), [x_train, x_test, x_valid])

    #get metrics
    try:
        y_predict = model.predict_classes(x_valid)
    except:
        y_predict = model.predict(x_valid)
        y_predict = np.argmax(y_predict, axis=1)

    report = classification_report(y_valid, y_predict)

    diff = np.unique(list((Counter(y_predict.flatten()) - Counter(y_valid.flatten())).elements()) + \
        list((Counter(y_valid.flatten()) - Counter(y_predict.flatten())).elements()))

    print(diff)
    x, y = load_preprocessed_data(signal_1d_file)
    misclassified_y = [ i for i, x in enumerate(y) if x in diff ]
    misclassified_x = x[misclassified_y]
    chunks = np.array_split(misclassified_x, samples)
    plt.plot(np.linspace(0, 199, 200), misclassified_x.T)
    plt.show()
    #for ch in chunks :
        #plt.plot(np.linspace(0, 199, 200), ch.T)
        #plt.show()

    
    ###########################################################################################################################
    cf = confusion_matrix(y_valid, y_predict)
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
