# my_ecg_project
This project aims to build and test several ANN models for ECG subject identification. This is a multiclass-classification task and CPSC 2018 ECG dataset 
was chosen as the main source of ECG records (that is publicly available for download at  https://disk.yandex.ru/d/oSItWrcWIF2qIA )
(and also PTB_XL dataset at https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip )

The basic steps:
1. Data preprocessing
2. Model training and evaluation

# Data preprocessing
In order to process data, we have to take only first lead from 12 leads and perform denoising, R-peaks detection algorithm, smoothing, normalization and balance classes.

All preprocessed data used in this project are available at https://disk.yandex.ru/d/2pRn8pO0tuiqIQ
Please, unpack them into project directory. Or if you want to reproduce all steps to get that, download and unpack raw data from the previous links and run 
raw_data_processing.py script's methods.

# Model training and evaluation
There are three types of ANNs were taken: convolutional, recurrent, predefined. 
In this case selected ANN acts as a feature extractor and a classifier: Softmax turn logits (numeric output of the last linear layer of a multi-class classification neural network) into probabilities by take the exponents of each output and then normalize each number by the sum of those exponents so the entire output vector adds up to one — all probabilities should add up to one.

You can run model training process from baseline.py script assigning required parameters or you can just rely on previously saved models, but first download them at https://disk.yandex.ru/d/DNUI9EzwUtmmSQ  unpack and then execute run_saved_model.py script specifying the model's name and appropriate dataset.
If you have a GPU you can make use of its processing capability to improve the training step.





