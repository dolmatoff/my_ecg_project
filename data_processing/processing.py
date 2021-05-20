import numpy as np
import pandas as pd
import sys
import wfdb
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import scipy
from scipy.signal import spectrogram
import keras
import pickle
from time import time
import os
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.signal import savgol_filter
import neurokit2 as nk
import random
from random import randint
import operator
from sklearn.preprocessing import normalize
import scipy.signal as signal
import pywt
import skimage
from skimage.transform import resize


def apply_smooth_filter(x) :
    def sm_filter(x):
        return savgol_filter(x, 11, 1)

    xx = pd.DataFrame(x).apply(sm_filter)
    return xx.to_numpy()

def load_preprocessed_1d_data(filename) :

    with open(filename, 'rb') as f:
        x, y = pickle.load(f)

    return np.array(x), np.array(y)

def load_train_test_data(filename, split_cpsc = False) :
    '''
    Load data from pickle file, then split data to training set and test set.
    Input: pickle filename
    Output: training data and testing data

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

###############################################################################################
def filter_qrs_ptb_xl() :
    with open(os.path.join('ptb_xl_data', 'processed_ptb_xl_all.pkl'), 'rb') as f :
        x, y = pickle.load(f)

    #x = normalize(x, axis=1, norm='l2')

    # apply the filter to make the signals smoother
    #x = apply_smooth_filter(x)

    #plt.plot(np.linspace(0, 99, 100), x.T)
    #plt.show()

    # make sure there are exactly 14 qrs segment per each class
    x_bal = []
    y_bal = []
    y = np.array(y)
    unique, counts = np.unique(y, return_counts=True)
    d = dict(zip(unique, counts))
    
    samples = 15 # may vary
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
      
    with open('ptb_xl_data/processed_ptb_xl_' + str(samples) + '.pkl', 'wb') as f:
        pickle.dump((x_bal, y_bal), f)

# reassign classes labels from 1 to 18885 (instead of 302-21797)
def sort_ptb_xl_labels() :

     with open(os.path.join('ptb_xl_data', 'processed_ptb_xl.pkl'), 'rb') as f:
        x, y = pickle.load(f)

     y_u = set(y)
     L = sorted(zip(x,y), key=operator.itemgetter(1))
     new_x, new_y = zip(*L)
     y_n = []

     count = 1
     ln = len(new_y)
     for i, val in enumerate(new_y) :
         y_n.append(count)
         if i+1 < ln and (val != new_y[i+1]) :
             count += 1

     with open('ptb_xl_data/processed_ptb_xl.pkl', 'wb') as f:
        pickle.dump((new_x, y_n), f)

     return new_x, y_n
     
def process_cpsc_data(filename):
   
    # load data
    x = pd.read_csv(os.path.join('cpsc2018', 'x_cpsc.csv')).to_numpy()
    y = pd.read_csv(os.path.join('cpsc2018', 'y_cpsc.csv')).to_numpy()

    # apply the filter to make the signals smoother
    x = apply_smooth_filter(x)

    #plt.plot(np.linspace(0, 199, 200), x.T)
    #plt.show()

    with open(filename, 'wb') as f:
        pickle.dump((x, y), f)



######################################################## DATA TRANSITIONS ########################################################
def to_spectrograms(signals, isCPSC):
    Sxx = []
    nfft = 50
    _noverlap = 44 if isCPSC else 48

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
        # show scalogram
        #plt.imshow(rescale_coeffs, cmap = 'coolwarm', aspect = 'auto')
        #plt.show()

        Sxx.append(rescale_coeffs)
        
    return Sxx


if __name__ == '__main__' :
   
    # convert 1d qrs segments to 2d matrix of spectral coefficients
    #x, y = load_preprocessed_1d_data('ptb_xl_data/ptb_xl_6480_14.pkl')
    #x_s = to_spectrograms(x, False)

    x, y = load_preprocessed_1d_data('cpsc2018/cpsc_1145_25.pkl')
    x_s = _cwt(x)

    # save spectrograms into .pkl file
    with open('cpsc2018/cpsc_1145_25_cwt.pkl', 'wb') as f:
        pickle.dump((x_s, y), f)

    #filter_qrs_ptb_xl()

