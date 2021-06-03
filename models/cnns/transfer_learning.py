import numpy as np
import os
import time
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from collections import Counter
from sklearn.metrics import classification_report
from data_processing.processing import * 
from tensorflow.keras.utils import to_categorical
from baseline import cpsc_num as numclasses
from data_processing.raw_data_processing import samples


def train_existing_model() :

    model = load_model('saved_models/cpsc_cnn2d_wavelets.h5')

    for layer in model.layers[:-1] :
        layer.trainable = False

    model.layers[6].trainable = True
    model.layers[7].trainable = True

    model.summary()

    replacedFile = 'cpsc2018/replaced_cwt.pkl'
    x, y = load_preprocessed_data('cpsc2018/cpsc_1145_25_cwt.pkl')
    if not os.path.exists(replacedFile) : 
        x_new, y_new = load_preprocessed_data('ptb_xl_data/ptb_xl_75_25_cwt.pkl')
        # replace first 75*25 records
        x[0:x_new.shape[0],:,:] = x_new
        with open(replacedFile, 'wb') as f:
            pickle.dump((x, y), f)
    else :
        with open(replacedFile, 'rb') as f:
            x, y = pickle.load(f)

    x_train, x_test, x_valid, y_train, y_test, y_valid = load_train_test_data('cpsc2018', 'cwt_', replacedFile)
    x_train, x_test, x_valid = map(lambda x: get_transformed_input(x), [x_train, x_test, x_valid])
    y_train = to_categorical(y_train, numclasses)
    y_test = to_categorical(y_test, numclasses)
    y_valid_ = to_categorical(y_valid, numclasses)

    if not os.path.exists('retrained1.h5') :
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        t = time.time()
        hist = model.fit(x=x_train,
                         y=y_train,
                         batch_size=368,
                         epochs=10,
                         verbose=1,
                         validation_data=(x_test, y_test))
        print('training time: %s' % (t - time.time()))
        model.save('retrained1.h5')
    else :
        model = load_model('retrained1.h5')
        evaluate_model(model, x_valid, y_valid_, y_valid)
    

    # remove the last 145 classes (3625 records)
    replacedFile2 = 'cpsc2018/replaced_cwt2.pkl'
    removed_classes = 145
    numclasses2 = numclasses - removed_classes
    if not os.path.exists(replacedFile) : 
        ln = x.shape[0] - removed_classes*samples
        x = x[0:ln-1,:,:]
        y = y[0:ln-1,:]
        with open(replacedFile2, 'wb') as f:
            pickle.dump((x, y), f)
    else :
        with open(replacedFile2, 'rb') as f:
            x, y = pickle.load(f)

    x_train, x_test, x_valid, y_train, y_test, y_valid = load_train_test_data('cpsc2018', 'cwt__', replacedFile2)
    x_train, x_test, x_valid = map(lambda x: get_transformed_input(x), [x_train, x_test, x_valid])
    y_train = to_categorical(y_train, numclasses2)
    y_test = to_categorical(y_test, numclasses2)
    y_valid_ = to_categorical(y_valid, numclasses2)

    # change the model
    model.layers.pop()
    x = model.layers[6].output
    o = Dense(1000, activation='softmax', name='dense_2_changed')(x)

    model2 = Model(inputs=model.input, outputs=[o])
    model2.summary()

    if not os.path.exists('retrained2.h5') :
        model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist2 = model2.fit(x=x_train,
                         y=y_train,
                         batch_size=368,
                         epochs=10,
                         verbose=1,
                         validation_data=(x_test, y_test))
        model2.save('retrained2.h5')
    else: 
        model2 = load_model('retrained2.h5')
        evaluate_model(model2, x_valid, y_valid_, y_valid)
    
    
    return hist, hist2


def get_transformed_input(x):
    return np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))

def evaluate_model(model, x_valid, y_valid_, y_valid) :
    y_predict = model.predict(x_valid)
    y_predict = np.argmax(y_predict, axis=1)
    diff = np.unique(list((Counter(y_predict.flatten()) - Counter(y_valid.flatten())).elements()) + \
        list((Counter(y_valid.flatten()) - Counter(y_predict.flatten())).elements()))
    print(diff)

    # evaluate model on validation data
    print('Evaluation...')
    score = model.evaluate(x_valid, y_valid_, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
