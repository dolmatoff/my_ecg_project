from sklearn.neighbors import LocalOutlierFactor
from data_processing.processing import load_preprocessed_data
import numpy as np

def predict_LOF(x_train, x_test, x_valid, dim) :

    def get_2d_input(x): return np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
    def get_1d_input(x): return np.reshape(x, (x.shape[0], x.shape[1]))

    clf = LocalOutlierFactor(n_neighbors=1, contamination=0.5, novelty=True, n_jobs=5)
    
    if(dim == 2) :
        ux_, uy_ = load_preprocessed_data('ptb_xl_data/ptb_xl_3490_15_cwt.pkl') #'ptb_xl_data/ptb_xl_75_25_cwt.pkl'
        x_train_, x_test_, x_valid_, ux_ = map(lambda x: get_2d_input(x), [x_train, x_test, x_valid, ux_])
    else :
        ux_, uy_ = load_preprocessed_data('ptb_xl_data/ptb_xl_3490_15.pkl') #'ptb_xl_data/ptb_xl_6480_14.pkl'
        x_train_, x_test_, x_valid_, ux_ = map(lambda x: get_1d_input(x), [x_train, x_test, x_valid, ux_])

    x_all = np.concatenate((x_train_, x_test_), axis=0)
    # fit the model for outlier detection (default)
    clf.fit(x_all)

    y_pred = clf.predict(x_train_)
    errs_train = sum(y_pred == -1)

    y_pred_ted = clf.predict(x_test_)
    errs_test = sum(y_pred_ted == -1)

    y_pred_val = clf.predict(x_valid_)
    errs_val = sum(y_pred_val == -1)

    # test on the unknown data
    y_pred_ud = clf.predict(ux_)
    # 75 classes: all 1875 - 120 errors (2d, 6.4%), 47 errors (1d, 2.51%)
    # 2d : 3175 from 52350 (6,06%), 1365 from 52350 (2.61%)
    errs_un = sum(y_pred_ud == 1) 

    return errs_train, errs_test, errs_un
