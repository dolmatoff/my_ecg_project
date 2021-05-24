"""Raw data parser

This script allows the user to prepare raw data to proceed. 
Multiple methods here aim to generate some intermediate files that must be used to get the final dataset.

"""

import argparse
import numpy as np
import pandas as pd
import sys
from scipy.signal import savgol_filter
import os
import sys
import matplotlib.pyplot as plt
import pickle
import operator
from data_processing.QRSDetectorOffline import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='raw_data/CPSC', help='Raw data (.mat, .hea, .dat) directory, where it should be unpacked')
    parser.add_argument('--data-processed-dir', type=str, default='cpsc2018', help='Preprocessed data directory')
    return parser.parse_args()

####### GLOBALS
qrs_threshold = 100
qrs_duration = 200
samples = 25

def generate_features_csv(features_csv, data_dir, patient_ids) :
    print('Generating expert features...')
    ecg_features = []
    damaged_ecgs = []

    for patient_id in tqdm(patient_ids):
        try:
            ecg_data, _ = wfdb.rdsamp(os.path.join(data_dir, patient_id))
            ecg_features.append(extract_features(ecg_data[:,0], ecg_id=patient_id))
        except:
            damaged_ecgs.append(patient_id)
            continue
    
    valid_ecgs = sorted(set(patient_ids).difference(set(damaged_ecgs)))
    df = pd.DataFrame(ecg_features, index=valid_ecgs, dtype='int64')
    df.index.name = 'patient_id'
    df.to_csv(features_csv)
    return df

def process_cpsc_data() :
    args = parse_args()
    data_dir = args.data_raw_dir
    processed_dir = args.data_processed_dir

    features_csv = os.path.join(data_dir, 'features.csv')
    labels_csv = os.path.join(data_dir, 'labels.csv')

    df_labels = pd.read_csv(labels_csv)
    patient_ids = df_labels['patient_id'].tolist()

    if not os.path.exists(features_csv):
        df_X = generate_features_csv(features_csv, data_dir, patient_ids)
    else:
        df_X = pd.read_csv(features_csv)

    df_X = df_X.merge(df_labels[['patient_id', 'fold']], on='patient_id')
    
    feature_cols = df_X.columns[1:-1] # remove patient id and fold

    X = []
    Y = []
    Y_real = []
    count = 0
    qrs_threshold = 100
    qrs_duration = 200
    samples = 25
    max_heartrate = 130

    # load data
    if os.path.exists(os.path.join(processed_dir, 'qrs_segments.csv')) and \
       os.path.exists(os.path.join(processed_dir, 'qrs_labels.csv')):
        X = pd.read_csv(os.path.join(processed_dir, 'qrs_segments.csv')).to_numpy()
        Y = pd.read_csv(os.path.join(processed_dir, 'qrs_labels.csv')).to_numpy()
    ###############################################################################################
    # or fill data
    else :
        # start to fill dataset
        for i in range(len(df_X)): # 6877 patients
            patient_id = df_X.values[i][0]
            ecg_data, _ = wfdb.rdsamp(os.path.join(data_dir, df_X.values[i][0]))
            p_data = ecg_data.transpose()
            r = p_data[0,:] # take the first row (I lead)

            try:
                peaks_indices = [x for x in df_X.values[i][1:263] if str(x) != 'nan']
                # take the records longer or equal 15s 
                if len(peaks_indices) >= samples and len(peaks_indices) < max_heartrate and \
                   r.shape[0] >= 7500 and min(r) >= -1 and max(r) <= 1: 
                    peak_idx = 0
                    xtemp = []
                    ytemp = []
                    # iterate by peaks indices
                    for peak_idx in peaks_indices :
                        peak_idx = int(peak_idx)
                        row = r[(peak_idx-qrs_threshold):(peak_idx+qrs_threshold)] #extract qrs-segment
                        if row.shape[0] == qrs_duration and abs(row[50]) <= 0.65 and abs(row[150]) <= 0.65: 
                            xtemp.append(row)   # qrs data

                        # add candidates with at least 25 qrs segments
                        if(len(xtemp) >= samples):
                            X += xtemp[:samples]
                            ytemp = np.full(samples, count)
                            Y = np.hstack((Y, ytemp))
                            Y_real.append(i)
                            count += 1
                            break
                     
            except:
                continue  
                 

        pd.DataFrame(np.vstack(X)).to_csv(os.path.join(processed_dir, 'qrs_segments.csv'), index=False)
        pd.DataFrame(np.vstack(Y)).to_csv(os.path.join(processed_dir, 'qrs_labels.csv'), index=False)
        pd.DataFrame(np.vstack(Y_real)).to_csv(os.path.join(processed_dir, 'qrs_real_labels.csv'), index=False)
    ###############################################################################################
    # number of classes 1145 - 25 samples per class !!!


################################################ PTB-XL PREPROCESSING ###############################################
path = 'data/ptbxl'
sampling_rate = 500

def load_dataset():
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path)

    pickle.dump(X, open('ptb_xl_data/X.pkl', 'wb'))
    pickle.dump(Y, open('ptb_xl_data/Y.pkl', 'wb'))
    return X, Y


def load_raw_data_ptbxl(df):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data_ = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = []
            for signal, meta in data_ :
                data.append(signal)
            #data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'))
    return data

# reassign classes labels from 1 to 18885 (instead of 302-21797)
def sort_ptb_xl_labels() :

     with open(os.path.join('ptb_xl_data', 'processed_ptb_xl_25.pkl'), 'rb') as f:
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

     with open('ptb_xl_data/processed_ptb_xl_25.pkl', 'wb') as f:
        pickle.dump((new_x, y_n), f)

     return new_x, y_n


def process_ptb_xl_data() :

     ecg_features = []
     damaged_ecgs_ids = []
     features_csv = os.path.join('ptb_xl_data', 'ptb_xl_features.csv')
     filtered_xy = os.path.join('ptb_xl_data' , 'valid_ptb_xl_data.pkl')

     if not os.path.exists(filtered_xy) : # load raw ecg data with labels
         with open(os.path.join('ptb_xl_data', 'X.pkl'), 'rb') as f:
            x = pickle.load(f)
         with open(os.path.join('ptb_xl_data', 'Y.pkl'), 'rb') as f:
            y = pickle.load(f)
     else :
         with open(filtered_xy, 'rb') as f:
            x, y = pickle.load(f)

     #import matplotlib.pyplot as plt
     #plt.plot(np.linspace(0, 4999, 5000), x.T)
     #plt.show()

     if not os.path.exists(features_csv) :
        for i, val in enumerate(y.astype(int)) :
            try:
                ecg_features.append(extract_features(x[i], ecg_id=val))
            except:
                damaged_ecgs_ids.append(i) # no damaged ecgs
                continue

        x = [v for i, v in enumerate(x) if i not in damaged_ecgs_ids]
        y = [v for i, v in enumerate(y) if i not in damaged_ecgs_ids]

        with open(filtered_xy, 'wb') as f: # save filtered ecgs
            pickle.dump((x, y), f)

        df = pd.DataFrame(ecg_features, dtype='int64') # save ecgs features
        df.to_csv(features_csv)

     else :
         ecg_features = pd.read_csv(features_csv).to_numpy().astype(int)
     
     qrs_segments = []
     classes = []

     processed_xy = os.path.join('ptb_xl_data' , 'processed_ptb_xl_data_' + str(samples) + '.pkl')

     for i, val in enumerate(map(int, y)) :
         peaks_indices = [x for x in ecg_features[i] if str(x) != 'nan' and x >= qrs_threshold ]
         ecg_data = x[i]
         cnt = 0
         q_tmp = []
         c_tmp = []
         if(len(peaks_indices) >= samples) :
             for idx in peaks_indices :
                if idx > qrs_threshold and idx < 4900 :
                    data = ecg_data[idx-qrs_threshold:idx+qrs_threshold]
                    if len(data) == qrs_duration :
                        q_tmp.append(data)
                        c_tmp.append(val)
                        cnt += 1
                    if cnt == samples :
                        qrs_segments += q_tmp[0:samples]
                        classes += c_tmp[0:samples]
                        break
     
     with open(processed_xy, 'wb') as f: # save processed qrs segments
        pickle.dump((qrs_segments, classes), f)

     
     plt.plot(np.linspace(0, 199, 200), np.asarray(qrs_segments).T)
     plt.show()
################################################ PTB-XL PREPROCESSING ###############################################


if __name__ == '__main__' :

    #process_ptb_xl_data()
    sort_ptb_xl_labels()
