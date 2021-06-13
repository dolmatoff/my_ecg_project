# my_ecg_project
This project aims to build and test several ANN models for ECG-based subject identification. This is a multiclass-classification task and CPSC 2018 ECG dataset 
was chosen as the main source of ECG records (that is publicly available for download at  https://disk.yandex.ru/d/oSItWrcWIF2qIA )
(and also PTB_XL dataset at https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip )

The basic steps:
1. Data preprocessing
2. Model training and evaluation

# Data preprocessing
In order to process data, we have to take only first lead from 12 leads and perform denoising, R-peaks detection algorithm, smoothing, normalization and balance classes.

All preprocessed data used in this project are available at 
https://drive.google.com/file/d/1SiDIYOY-5xRVZVm0sfyXGbPQD7QGRggn/view?usp=sharing
Please, unpack them into project directory. Or if you want to reproduce all steps to get that, download and unpack raw data from the previous links and run 
raw_data_processing.py script's methods.

# Model training and evaluation
There are three types of ANNs were taken as classifiers: convolutional, recurrent, predefined. 
In this case selected ANN acts as a feature extractor and a classifier: Softmax turns logits (numeric output of the last linear layer of a multi-class classification neural network) into probabilities by taking the exponents of each output and then normalize each number by the sum of those exponents so the entire output vector adds up to one — all probabilities should add up to one.

The second problem is outlier detection. Unsupervised Outlier Detection using Local Outlier Factor (LOF) from sklearn library was tested for this purpose in run_FOF.py.

You can run model training process from baseline.py script assigning required parameters or you can just rely on the previously saved models, but first download them at https://disk.yandex.ru/d/U-dlOLbFhTDXDw  unpack and then execute run_saved_model.py script specifying the model's name and the appropriate dataset.
If you have a GPU you can make use of its processing capability to improve the training step.

# Results
The overall classification accuracy obtained using the CNN2D LeNet5 architecture was 95% on the validation set:
|Accuracy  | Loss  | Training time | Epochs |
| :--------| :---- | :------------ | -----: |
|  95%     | 0.086 | 20 min        | 54     |





