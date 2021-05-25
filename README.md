# my_ecg_project
This project aims to build and test several ANN models for ECG subject identification. This is a multiclass-classification task and CPSC 2018 ECG dataset 
was chosen as the main source of ECG records (that is publicly available for download at  https://disk.yandex.ru/d/oSItWrcWIF2qIA )
(and also PTB_XL dataset at https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip )

The basic steps:
1. Data preprocessing
2. Model training and evaluation

# Data preprocessing
In order to process data, we have to take only first lead from 12 leads and perform denoising, R-peaks detection algorithm, smoothing, normalization and balance classes.
