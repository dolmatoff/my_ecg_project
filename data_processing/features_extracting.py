import argparse
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb

from QRSDetectorOffline import QRSDetectorOffline

import pickle

warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-cpsc-dir', type=str, default='data/CPSC', help='Raw CPSC2018 data directory')
    parser.add_argument('--data-ptbxl-dir', type=str, default='data/ptbxl', help='Raw PTB-XL data directory')
    return parser.parse_args()


# split each reacord into qrs segments
def generate_features_csv(features_csv, data_dir, patient_ids):
    
    print('Generating expert features...')

    ecg_features = []
    damaged_ecgs = []

    for patient_id in tqdm(patient_ids):
        try:
            ecg_data = wfdb.rdsamp(os.path.join(data_dir, patient_id))
            ecg_features.append(extract_features(ecg_data, ecg_id=patient_id)) #ecg_data[:,0]
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
    data_dir = args.data_cpsc_dir

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

    # load data
    if os.path.exists(os.path.join('cpsc2018', 'x_cpsc.csv')) and \
       os.path.exists(os.path.join('cpsc2018', 'y_cpsc.csv')):
        X = pd.read_csv(os.path.join('cpsc2018', 'x_cpsc.csv')).to_numpy()
        Y = pd.read_csv(os.path.join('cpsc2018', 'y_cpsc.csv')).to_numpy()
    ###############################################################################################
    else :
        # start to fill dataset
        for i in range(len(df_X)): # 6877 patients
            #patient_id = df_X.values[i][0]
            ecg_data, meta = wfdb.rdsamp(os.path.join(data_dir, patient_id)) # read all 12 leads data

            try:
                p_data = ecg_data.transpose()
                r = p_data[0,:] # take the first row (I lead)
                
                # take the records longer or equal 15s 
                if r.shape[0] >= 7500 and min(r) >= -1 and max(r) <= 1 :
                    peak_idx = 0
                    xtemp = []
                    ytemp = []
                    # iterate by peaks indices
                    for peak_idx in df_X.values[i][1:] :
                        peak_idx = int(peak_idx)
                        row = r[(peak_idx-100):(peak_idx+100)] # take the qrs segment
                        if row.shape[0] == 200 and abs(row[50]) <= 0.65 and abs(row[150]) <= 0.65 : # brute amplitude filter 
                            xtemp.append(row)   # qrs data

                        # add candidates with at least 25 qrs segments
                        samples = 25
                        if(len(xtemp) >= samples):
                            X += xtemp[:samples]
                            ytemp = np.full(samples, count)
                            Y = np.hstack((Y, ytemp))
                            Y_real.append(i)
                            count += 1
                            break
                     
            except:
                continue  
                 
        # dump the data to the disk
        pd.DataFrame(np.vstack(X)).to_csv(os.path.join('cpsc2018', 'x_cpsc.csv'), index=False)
        pd.DataFrame(np.vstack(Y)).to_csv(os.path.join('cpsc2018', 'y_cpsc.csv'), index=False)
        pd.DataFrame(np.vstack(Y_real)).to_csv(os.path.join('cpsc2018', 'y_real_cpsc.csv'), index=False)
    ###############################################################################################
    # number of classes 1145 - 25 samples per class

def load_raw_data_ptbxl(df, sampling_rate, path):
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


# split each reacord into qrs segments
def process_ptb_xl_data() :
     parse_args()
     data_dir = args.data_ptbxl_dir

     ecg_features = []
     damaged_ecgs_ids = []

     features_csv = os.path.join(data_dir, 'ptb_xl_features.csv')
     filtered_xy = os.path.join(data_dir, 'valid_ptb_xl_data.pkl')
     processed_xy = os.path.join(data_dir , 'processed_ptb_xl_data.pkl')

     if not os.path.exists(filtered_xy) : # load raw ecg data with labels
         with open(os.path.join(data_dir, 'X.pkl'), 'rb') as f:
            x = pickle.load(f)
         with open(os.path.join(data_dir, 'Y.pkl'), 'rb') as f:
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

     p_data = []
     c = []

     for i, val in enumerate(map(int, y)) :
         peaks_indices = ecg_features[i]
         ecg_data = x[i]
         for idx in peaks_indices :
            if idx > 0 and idx < 5000 :
                data = ecg_data[idx-50:idx+50]
                if len(data) == 100 :
                    qrs_segments.append(data)
                    classes.append(val)

     samples = 11 # qrs samples per class
     for val in set(y) :
        indices = [index for index, value in enumerate(classes) if value == int(val)]
        if len(indices) > samples :
            for i in indices :
                p_data.append(qrs_segments[i])
                c.append(classes[i])
      
     with open(processed_xy, 'wb') as f: # save processed qrs segments
        pickle.dump((p_data, c), f)
     #######################################################################################################
     # number of classes 11532 - 11 samples per class

     #import matplotlib.pyplot as plt
     #plt.plot(np.linspace(0, 99, 100), np.asarray(p_data).T)
     #plt.show()

##########################################################################################################################################
def cal_entropy(coeff):
    # calculate shannon entropy
    coeff = pd.Series(coeff).value_counts()
    e = entropy(coeff)
    return e / 10


def extract_stats(channel):
    # extract statistic features
    n5 = np.percentile(channel, 5)
    n25 = np.percentile(channel, 25)
    n75 = np.percentile(channel, 75)
    n95 = np.percentile(channel, 95)
    median = np.percentile(channel, 50)
    mean = np.mean(channel)
    std = np.std(channel)
    return [n5, n25, n75, n95, median, mean, std]


def extract_wavelet_features(channel):
    # extract wavelet coeff features
    coeffs = pywt.wavedec(channel, 'db1', level=3)
    return coeffs[0] + [cal_entropy(coeffs[0])]


def extract_heart_rates(ecg_data, sampling_rate=500, ecg_id=0):
    # extract instant heart rates
    qrs_detector = QRSDetectorOffline(ecg_data, frequency=sampling_rate, ecg_id=ecg_id)
    #index, heart_rates = tools.get_heart_rate(qrs_detector.detected_peaks_indices, sampling_rate=sampling_rate)
    return qrs_detector.detected_peaks_indices


def extract_hos(channel):
    # extract higher order statistics features
    return []


def extract_channel_features(channel):
    stats_features = extract_stats(channel)
    wavelet_features = extract_wavelet_features(channel)
    #hos_features = extract_hos(channel)
    return np.array(list(stats_features) + list(wavelet_features)) #+ hos_features

   
def extract_features(ecg_data, sampling_rate=500, ecg_id=0):
    # extract expert features for 12-lead ECGs
    all_features = []
    all_features = extract_heart_rates(ecg_data, sampling_rate=sampling_rate, ecg_id=ecg_id)
    #channel_features = extract_channel_features(ecg_data)
    #all_features = np.array(list(all_features) + channel_features)
    return all_features


if __name__ == "__main__":
    
    #process_cpsc_data()
    process_ptb_xl_data()

    
