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
from sklearn.model_selection import StratifiedShuffleSplit
import random
from scipy.signal import savgol_filter
import pywt
import skimage
from skimage.transform import resize
from sklearn.preprocessing import normalize


def apply_smooth_filter(x) :
    """
        applies savitzky golay filter in order to make the curves x smoother
    """
    def sm_filter(x):
        return savgol_filter(x, 11, 1)

    xx = pd.DataFrame(x).apply(sm_filter)
    return xx.to_numpy()


def load_preprocessed_1d_data(filename) :
    """
        reads and returns X and Y sets from .pkl file
    """
    with open(filename, 'rb') as f:
        x, y = pickle.load(f)

    return np.array(x), np.array(y)

def load_train_test_data(filename, split_cpsc = False) :
    '''
    Load data from pickle file, then split data to training set, test set, validation set using StratifiedShuffleSplit class.
    Input: 
        pickle filename
        split_cpsc - defines the share of validation set
    Output: 
        training data, testing data, validation set

    '''
    
    x, y = load_preprocessed_1d_data(filename)
    valid_part = 0.05 if split_cpsc else 0.075
    test_part = 0.15

    train_valid = StratifiedShuffleSplit(n_splits=2, test_size=valid_part, random_state=42)
    train_test = StratifiedShuffleSplit(n_splits=2, test_size=test_part, random_state=42)

    for train_index, val_index in train_valid.split(x, y):
        x_train, x_valid = x[train_index], x[val_index]
        y_train, y_valid = y[train_index], y[val_index]
    
    for train_index, test_index in train_test.split(x_train, y_train):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return x_train, x_test, x_valid, y_train.astype(int), y_test.astype(int), y_valid.astype(int)


  
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
   
    #filename = 'data/processed_cpsc/cpsc_1431_25.pkl'
    #targetFilename = 'data/processed_cpsc/cpsc_1431_25_cwt.pkl'
    
    #x, y = load_preprocessed_1d_data(filename)
    ## convert 1d qrs segments to 2d spatial representation
    #x_s = _cwt(x)

    ## save transformed data into .pkl file
    #with open(targetFilename, 'wb') as f:
    #    pickle.dump((x_s, y), f)

    filter_qrs_ptb_xl() 

    #normalize_xy_data()

