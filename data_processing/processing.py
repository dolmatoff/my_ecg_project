"""Data parser

This script allows the user to prepare partially processed data to proceed. 
Multiple methods here aim to normalize, filter, transform data to appropriate forms.
The final version is saved in .pkl file, where there are X and Y sets containing selected qrs-segments and their labels.

"""

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import scipy
import pickle
import os
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
import random
from scipy.signal import savgol_filter
import pywt
import skimage
from skimage.transform import resize
from sklearn.preprocessing import normalize
from data_processing.raw_data_processing import samples


def apply_smooth_filter(x) :
    """
        applies savitzky golay filter in order to make the curves x smoother
    """
    def sm_filter(x):
        return savgol_filter(x, 11, 1)

    xx = pd.DataFrame(x).apply(sm_filter)
    return xx.to_numpy()


def load_preprocessed_data(filename) :
    """
        reads and returns X and Y sets from .pkl file
    """
    with open(filename, 'rb') as f:
        x, y = pickle.load(f)

    return np.array(x), np.array(y)

def load_train_test_data(path, prefix, filename) :
    '''
    Load data from pickle file, then split data to training set, test set, validation sets.
    Input: 
        path to the preprocessed data
        prefix to identify the target sets
        pickle filename
    Output: 
        training data, testing data, validation set (randomly shuffled)

    '''
    def getTargetName(mainPart, prefix) :
        return os.path.join(path, mainPart + prefix + '.pkl')

    if os.path.exists(getTargetName('train_', prefix)) and \
       os.path.exists(getTargetName('test_', prefix )) and \
       os.path.exists(getTargetName('validation_', prefix)) :
        with open(getTargetName('train_', prefix), 'rb') as f :
            x_train, y_train = pickle.load(f)
        with open(getTargetName('test_', prefix ), 'rb') as f :
            x_test, y_test = pickle.load(f)
        with open(getTargetName('validation_', prefix), 'rb') as f :
            x_valid, y_valid = pickle.load(f)
    else :
        x, y = load_preprocessed_data(filename)
        y = y.astype(int).flatten()
        #declare empty lists
        x_train = np.empty((0, x.shape[1])) if prefix == '1d' else np.empty((0, x.shape[1], x.shape[2]))
        x_test = np.empty((0, x.shape[1])) if prefix == '1d' else np.empty((0, x.shape[1], x.shape[2]))
        x_valid=np.empty((0, x.shape[1])) if prefix == '1d' else np.empty((0, x.shape[1], x.shape[2])) 
        y_train=[]; y_test=[]; y_valid=[]

        rs_valid = ShuffleSplit(n_splits=2, test_size=0.04, train_size=0.96, random_state=42)
        rs_test = ShuffleSplit(n_splits=2, test_size=0.125, train_size=0.875, random_state=42)

        for i in range(0, y.shape[0], samples) :
            x_row = x[i:i+samples,:] if prefix == '1d' else x[i:i+samples,:,:]
            y_row = y[i:i+samples]
            _, train_valid_idx = rs_valid.split(y_row)
            _, train_test_idx = rs_test.split(y_row[train_valid_idx[0]])

            x_valid = np.vstack((x_valid, x_row[train_valid_idx[1][0]]))
            y_valid.append(y_row[train_valid_idx[1][0]])
            x_test = np.vstack((x_test, x_row[train_test_idx[1]]))
            y_test.extend(y_row[train_test_idx[1]])
            x_train = np.vstack((x_train, x_row[train_test_idx[0]]))
            y_train.extend(y_row[train_test_idx[0]])
    
        x_train, y_train = shuffle(np.asarray(x_train), np.asarray(y_train))
        x_test, y_test = shuffle(np.asarray(x_test), np.asarray(y_test))
        x_valid, y_valid = shuffle(np.asarray(x_valid), np.asarray(y_valid))

        x_train, x_test, x_valid = map(lambda t: t.reshape(t.shape[0], t.shape[1], 1) if prefix == '1d' \
            else t.reshape(t.shape[0], t.shape[2], t.shape[2]), [x_train, x_test, x_valid])

        with open(getTargetName('train_', prefix), 'wb') as f:
            pickle.dump((x_train, y_train), f)
        with open(getTargetName('test_', prefix), 'wb') as f:
            pickle.dump((x_test, y_test), f)
        with open(getTargetName('validation_', prefix), 'wb') as f:
            pickle.dump((x_valid, y_valid), f)


    return x_train, x_test, x_valid, y_train, y_test, y_valid


  
def process_cpsc_data(filename):
    """
        reads X and Y sets from .pkl file, smooths the curves from X and saves that again
    """
    # load data
    x = pd.read_csv(os.path.join('cpsc2018', 'x_cpsc.csv')).to_numpy()
    y = pd.read_csv(os.path.join('cpsc2018', 'y_cpsc.csv')).to_numpy()

    # apply the filter to make the signals smoother
    x = apply_smooth_filter(x)

    #plt.plot(np.linspace(0, 199, 200), x.T)
    #plt.show()

    with open(filename, 'wb') as f:
        pickle.dump((x, y), f)


def filter_qrs_ptb_xl() :
    """
        reads X and Y sets from .pkl file
        -smooths the curves from X
        -normalizes thos curves
        -selects big enough data and extract equal parts
        and saves that again
    """
    with open(os.path.join('ptb_xl_data', 'processed_ptb_xl_25.pkl'), 'rb') as f :
        x, y = pickle.load(f)

    # apply the filter to make the signals smoother
    x = apply_smooth_filter(x)
    x = normalize(x, axis=1, norm='l2')

    # make sure there are exactly 14 qrs segment per each class
    x_bal = []
    y_bal = []
    y = np.array(y)
    unique, counts = np.unique(y, return_counts=True)
    d = dict(zip(unique, counts))
    
    samples = 25 # may vary
    count = 0
    for i in unique:
        occrs = d[i]
        if occrs >= samples :
            indices = [k for k, val in enumerate(y) if val == i]
            if len(indices) >= samples :
                x_values = x[indices]
                y_values = y[indices]
                count += 1
                for j in range(0, samples) :
                    x_bal.append(x_values[j])
                    y_bal.append(count)

    plt.plot(np.linspace(0, 199, 200), x.T)
    plt.show()

    with open('ptb_xl_data/processed_ptb_xl_' + str(samples) + '.pkl', 'wb') as f:
        pickle.dump((x_bal, y_bal), f)


######################################################## DATA TRANSITIONS ########################################################
def to_spectrograms(signals):
    Sxx = []
    nfft = 50
    _noverlap = 44

    for d in signals:
        spectrum, freqs, t, im = plt.specgram(d, Fs=500, NFFT=nfft, window=None, mode='magnitude', noverlap=_noverlap)
        Sxx.append(spectrum)
        #plt.xlabel('Time')
        #plt.ylabel('Frequency')
        #plt.show() 
    
    return Sxx

def _cwt(data):
    Sxx = []
    rescale_size = 64

    for d in data:
        # apply  PyWavelets continuous wavelet transfromation function
        coeffs, freqs = pywt.cwt(d, np.arange(1, 65), wavelet = 'morl')
        rescale_coeffs = resize(coeffs, (rescale_size, rescale_size), mode = 'constant')
        # show result
        #plt.imshow(rescale_coeffs, cmap = 'coolwarm', aspect = 'auto')
        #plt.show()

        Sxx.append(rescale_coeffs)
        
    return Sxx
######################################################## DATA TRANSITIONS ########################################################

if __name__ == '__main__' :
   
    filename = 'ptb_xl_data/ptb_xl_75_25.pkl'
    targetFilename = 'ptb_xl_data/ptb_xl_75_25_cwt.pkl'
    
    x, y = load_preprocessed_data(filename)
    # convert 1d qrs segments to 2d spatial representation
    x_s = _cwt(x)

    # save transformed data into .pkl file
    with open(targetFilename, 'wb') as f:
        pickle.dump((x_s, y), f)

    #filter_qrs_ptb_xl() 

    #normalize_xy_data()

