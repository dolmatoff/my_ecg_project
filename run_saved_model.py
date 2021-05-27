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
    y_train = np_utils.to_categorical(y_train, numclasses)
    y_test = np_utils.to_categorical(y_test, numclasses)
    y_valid_ = np_utils.to_categorical(y_valid, numclasses)

    ux, uy = load_preprocessed_data(unknown_file)
    unknown_idx = random.randint(1, unknown_num)

    model_space = locals()[model_name]
    x_train, x_test, x_valid, ux = map(lambda x: model_space.get_transformed_input(x), [x_train, x_test, x_valid, ux])

    # check on the known and unknown data
    ch = model.predict(x_valid)
    errs = np.asarray([a for a in ch if max(a) < 0.98])
    print(errs.argmax(axis=-1))
    ch2 = model.predict(ux)
    errs2 = np.asarray([a for a in ch2 if max(a) < 0.98])
    print(errs2.argmax(axis=-1))

    #check on the one randomly selected element
    s_idx = random.randint(1, x_valid.shape[0])
    s_x = x_valid[s_idx]
    s_x = s_x.reshape(1, s_x.shape[0], s_x.shape[1], 1)
    s_result = model.predict(s_x)
    if(s_result.max() > 0.99) : 
        s_predicted = s_result.argmax(axis=-1)
        if(y_valid[s_idx] == s_predicted) :
            print('An existent unseen class was classified correctly.')
        else :
            print('An existent unseen class was classified erroneously.')
        print('Predicted label:')
        print(s_result.argmax(axis=-1))
        print('Real label:')
        print(y_valid[s_idx])
    else :
        print('An existent unseen class with the label '+ str(y_valid[s_idx]) + ' was rejected by mistake')
    

    u_x = ux[random.randint(1, unknown_num)]
    u_x = u_x.reshape(1, u_x.shape[0], u_x.shape[1], 1)
    u_result = model.predict(u_x)
    if(u_result.max() > 0.99) : 
        print('Error unknown class label: ')
        print(u_result.argmax(axis=-1))
    else :
        print('Unknown class was rejected')

    #get metrics
    try:
        y_predict = model.predict_classes(x_valid)
    except:
        y_predict = model.predict(x_valid)
        y_predict = np.argmax(y_predict, axis=1)

    report = classification_report(y_valid, y_predict)

    #diff = np.unique(list((Counter(y_predict.flatten()) - Counter(y_valid.flatten())).elements()))
    #print(diff)
    #x, y = load_preprocessed_data(signal_1d_file)
    #misclassified_y = [ i for i, x in enumerate(y) if x in diff ]
    #misclassified_x = x[misclassified_y]
    #chunks = np.array_split(misclassified_x, samples)
    #for ch in chunks :
    #    plt.plot(np.linspace(0, 199, 200), ch.T)
    #    plt.show()

    
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
