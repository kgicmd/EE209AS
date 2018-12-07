# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:41:56 2018

@author: kgicmd
"""

import numpy as np
import os
os.chdir('E:\\har')

import pandas as pd

from keras.layers import Input, BatchNormalization, concatenate, Flatten, Dropout, Dense,ReLU, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.utils import normalize

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from util import *
from sklearn.metrics import confusion_matrix
from keras.models import Sequential

from keras.utils import np_utils

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
RANDOM_SEED = 666
LEARNING_RATE = 7e-4


# define baseline model
def autoencoder_model(input_seq, encoding_dim):
	# create model
    encoded1 = Dense(128, activation='relu')(input_seq)
    #encoded1 = LeakyReLU()(encoded1)
    encoded2 = Dense(16, activation='relu')(encoded1)
    #encoded2 = LeakyReLU()(encoded2)
    encoded3 = Dense(8, activation='relu')(encoded2)
    #encoded3 = LeakyReLU()(encoded3)
    encoder_output = Dense(encoding_dim)(encoded3)
    
    decode1 = Dense(16, activation='relu')(encoder_output)
    #decode1 = LeakyReLU()(decode1)
    decode2 = Dense(128, activation='relu')(decode1)
    #decode2 = LeakyReLU()(decode2)
    decode3 = Dense(561, activation='relu')(decode2)
    #decode3 = LeakyReLU()(decode3)
    return decode3


def mask_data_and_normalize(X_train_raw, X_test_raw, option = None):
    '''
    mask data and return normalized data filled w/ 0's
    '''
    if option:
        # mark data with 0
        X_train_raw_0 = mask_source_channel([option_item for option_item in option], X_train_raw, 0)
        X_train_raw_0 = X_train_raw_0.dropna(axis=1)
        
        X_test_raw_0 = mask_source_channel([option_item for option_item in option], X_test_raw, 0)
        X_test_raw_0 = X_test_raw_0.dropna(axis=1)

        # split in 5 folds        
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for train_index, validation_index in kf.split(X_train_raw):
            pass
        
        # split raw data
        X_train = X_train_raw.iloc[train_index]
        X_train_val = X_train_raw.iloc[validation_index]
        # split missed features data
        X_train_0 = X_train_raw_0.iloc[train_index]
        X_train_val_0 = X_train_raw_0.iloc[validation_index]
        
        # normalize data
        X_train_nor = normalize_data(X_train)
        X_train_val_nor = normalize_data(X_train_val)
        X_train_0_nor = normalize_data(X_train_0)
        X_train_0_val_nor = normalize_data(X_train_val_0)
        
        # testing data
        X_test_raw_0_nor = normalize_data(X_test_raw_0)
        X_test_raw_nor = normalize_data(X_test_raw)
        
    else:
        raise('wrong option:{}'.format(option))
    
    return X_train_0_nor, X_train_nor, X_train_0_val_nor, X_train_val_nor, X_test_raw_0_nor, X_test_raw_nor


def autoencoder_test(X_input, X_output, X_val_input, X_val_output, X_test_input, X_test_raw, X_test_output, encoding_dim = 4, autoencoder=None):
    '''
    testing with encoding dim
    '''
    input_dim = X_input.shape[1]
    
    if not autoencoder:
        # sensor data autoencoder
        input_data = Input(shape=(input_dim,))
        output_data = autoencoder_model(input_data, encoding_dim)
        # construct autoencoder
        autoencoder = Model(inputs=input_data, outputs=output_data)
         
        # compile autoencoder  
        autoencoder.compile(optimizer=Adam(lr = LEARNING_RATE), loss='mean_squared_error')
    else:
        pass
    #autoencoder.summary()
    
    history = autoencoder.fit(X_input, X_output,
                                epochs=300,
                                batch_size=20,
                                shuffle=True,
                                validation_data=(X_val_input, X_val_output),
                                callbacks=[TensorBoard(log_dir='./tmp/autoencoder')])
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # predict with normalized data and denormalize
    X_test_raw_0_recover = autoencoder.predict(X_test_input)
    X_test_raw_0_recover_denor = denormalize_data(X_test_raw_0_recover, X_test_raw)
    
    # rmse
    mse = np.mean(np.mean(np.abs((X_test_raw_0_recover_denor - X_test_raw))))
    
    return loss, val_loss, mse, autoencoder

# import and mask data
X_train_raw, y_train_raw, X_test_raw, y_test_raw = import_data()
X_train_0_nor, X_train_nor, X_train_0_val_nor, X_train_val_nor, X_test_raw_0_nor, X_test_raw_nor = mask_data_and_normalize(X_train_raw, X_test_raw, option = ['Acc'])

# train with gyro
_, _, _, autoencoder = autoencoder_test(X_train_0_nor,
                                        X_train_nor,
                                        X_train_0_val_nor,
                                        X_train_val_nor,
                                        X_test_raw_0_nor,
                                        X_test_raw,
                                        X_test_raw_nor,
                                        encoding_dim = 6)


X_train_raw, y_train_raw, X_test_raw, y_test_raw = import_data()
X_train_0_nor, X_train_nor, X_train_0_val_nor, X_train_val_nor, X_test_raw_0_nor, X_test_raw_nor = mask_data_and_normalize(X_train_raw, X_test_raw, option = ['Gyro'])

# train with acc
loss, val_loss, mse, autoencoder = autoencoder_test(X_train_0_nor,
                                        X_train_nor,
                                        X_train_0_val_nor,
                                        X_train_val_nor,
                                        X_test_raw_0_nor,
                                        X_test_raw,
                                        X_test_raw_nor,
                                        encoding_dim = 6,
                                        autoencoder=autoencoder)


X_train_0_nor2, X_train_nor2, X_train_0_val_nor2, X_train_val_nor2, X_test_raw_0_nor2, X_test_raw_nor2 = mask_data_and_normalize(X_train_raw, X_test_raw, option = ['Acc'])

X_train_0_nor3 = np.concatenate((X_train_0_nor, X_train_0_nor2), axis=0)
X_train_nor3 = np.concatenate((X_train_nor, X_train_nor2), axis=0)
X_train_0_val_nor3 = np.concatenate((X_train_0_val_nor, X_train_0_val_nor2), axis=0)
X_train_val_nor3 = np.concatenate((X_train_val_nor, X_train_val_nor2), axis=0)
X_test_raw_0_nor3 = np.concatenate((X_test_raw_0_nor, X_test_raw_0_nor2), axis=0)
X_test_raw_nor3 = np.concatenate((X_test_raw_nor, X_test_raw_nor2), axis=0)
X_test_raw3 = np.concatenate((X_test_raw, X_test_raw), axis=0)

# train with combined
loss, val_loss, mse, autoencoder = autoencoder_test(X_train_0_nor3,
                                        X_train_nor3,
                                        X_train_0_val_nor3,
                                        X_train_val_nor3,
                                        X_test_raw_0_nor3,
                                        X_test_raw3,
                                        X_test_raw_nor3,
                                        encoding_dim = 6,
                                        autoencoder=autoencoder)

X_test_raw_0_recover_denor = denormalize_data(autoencoder.predict(X_test_raw_0_nor2), X_test_raw)

encoding_dim_list = np.arange(2,11,2)

loss_list = []
val_loss_list = []
mse_list = []
for encoding_dim in encoding_dim_list:
    loss, val_loss, mse = autoencoder_test(X_train_0_nor,
                                            X_train_nor,
                                            X_train_0_val_nor,
                                            X_train_val_nor,
                                            X_test_raw_0_nor,
                                            X_test_raw,
                                            X_test_raw_nor,
                                            encoding_dim = encoding_dim)
    loss_list.append(loss[-1])
    val_loss_list.append(val_loss[-1])
    mse_list.append(mse)


def construct_layers(autoencoder_layers, input_dim, dropout):
    model = Sequential()
    flag = 0
    while(len(autoencoder_layers) > 0):
        if flag == 0:
            # if it is the first layer
            model.add(Dense(autoencoder_layers[0], input_shape=(input_dim,)))
            model.add(LeakyReLU())
            if dropout > 0 and len(autoencoder_layers) > 1:
                model.add(Dropout(dropout))
        else:
            model.add(Dense(autoencoder_layers[0]))
            model.add(LeakyReLU())
            if dropout > 0 and len(autoencoder_layers) > 1:
                model.add(Dropout(dropout))
        autoencoder_layers = autoencoder_layers[1:]
    return model


def autoencoder_test2(autoencoder_layers, X_input, X_output, X_val_input, \
                      X_val_output, X_test_nor,X_test_raw,verbose=1,epochs=[100,100],\
                      batch_size=[50,20], dropout=-0.1, autoencoder = None):
    input_dim = X_input.shape[1]
    # construct autoencoder
    if not autoencoder:
        autoencoder = construct_layers(autoencoder_layers, input_dim, dropout)
            # compile autoencoder  
        autoencoder.compile(optimizer=Adam(lr = LEARNING_RATE), loss='mean_squared_error')
         
        autoencoder.summary()
    else:
        pass
    

    
    history = autoencoder.fit(X_input, X_output,
                                epochs=epochs[0],
                                batch_size=batch_size[0],
                                shuffle=True,
                                validation_data=(X_val_input, X_val_output),
                                verbose=verbose)
    
    history = autoencoder.fit(X_input, X_output,
                            epochs=epochs[1],
                            batch_size=batch_size[1],
                            shuffle=True,
                            validation_data=(X_val_input, X_val_output),
                            verbose=verbose)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # predict with normalized data and denormalize
    X_test_raw_0_recover = autoencoder.predict(X_test_nor)
    X_test_raw_0_recover_denor = denormalize_data(X_test_raw_0_recover, X_test_raw)
    
    # rmse
    mse = np.mean(np.mean((X_test_raw_0_recover_denor - X_test_raw)**2))
    
    return loss, val_loss, mse, autoencoder

autoencoder_layers = [561]
autoencoder_layers.extend(list(np.repeat(250,5)))
autoencoder_layers.append(561)
    
# import and mask data
X_train_raw, y_train_raw, X_test_raw, y_test_raw = import_data()
X_train_0_nor, X_train_nor, X_train_0_val_nor, X_train_val_nor, X_test_raw_0_nor, X_test_raw_nor = mask_data_and_normalize(X_train_raw, X_test_raw, option = ['Acc'])

# train with gyro
_, _, _, autoencoder250 = autoencoder_test2(autoencoder_layers,
                                         X_train_0_nor,
                                         X_train_nor,
                                         X_train_0_val_nor,
                                         X_train_val_nor,
                                         X_test_raw_0_nor,
                                         X_test_raw)

X_train_0_nor, X_train_nor, X_train_0_val_nor, X_train_val_nor, X_test_raw_0_nor, X_test_raw_nor = mask_data_and_normalize(X_train_raw, X_test_raw, option = ['Gyro'])

# train with acc
loss, val_loss, mse, autoencoder250 = autoencoder_test2(autoencoder_layers,
                                                     X_train_0_nor,
                                                     X_train_nor,
                                                     X_train_0_val_nor,
                                                     X_train_val_nor,
                                                     X_test_raw_0_nor,
                                                     X_test_raw_nor,
                                                     autoencoder=autoencoder250)


X_train_0_nor2, X_train_nor2, X_train_0_val_nor2, X_train_val_nor2, X_test_raw_0_nor2, X_test_raw_nor2 = mask_data_and_normalize(X_train_raw, X_test_raw, option = ['Acc'])

X_train_0_nor3 = np.concatenate((X_train_0_nor, X_train_0_nor2), axis=0)
X_train_nor3 = np.concatenate((X_train_nor, X_train_nor2), axis=0)
X_train_0_val_nor3 = np.concatenate((X_train_0_val_nor, X_train_0_val_nor2), axis=0)
X_train_val_nor3 = np.concatenate((X_train_val_nor, X_train_val_nor2), axis=0)
X_test_raw_0_nor3 = np.concatenate((X_test_raw_0_nor, X_test_raw_0_nor2), axis=0)
X_test_raw_nor3 = np.concatenate((X_test_raw_nor, X_test_raw_nor2), axis=0)
X_test_raw3 = np.concatenate((X_test_raw, X_test_raw), axis=0)

# train with combined
loss, val_loss, mse, autoencoder250 = autoencoder_test2(autoencoder_layers,
                                                       X_train_0_nor3,
                                                       X_train_nor3,
                                                       X_train_0_val_nor3,
                                                       X_train_val_nor3,
                                                       X_test_raw_0_nor3,
                                                       X_test_raw3,
                                                       autoencoder=autoencoder250)


X_test_raw_0_recover_denor = denormalize_data(autoencoder250.predict(X_test_raw_0_nor), X_test_raw)

### deeep test
loss_list = []
val_loss_list = []
mse_list = []
epochs = [100,100]
batch_size = [50,20]

for rep_times in np.arange(1,20,2):
    start = X_train_0_nor.shape[1]
    end = 561
    autoencoder_layers = [start]
    autoencoder_layers.extend(list(np.repeat(250,rep_times)))
    autoencoder_layers.append(end)
    loss, val_loss, mse = autoencoder_test2(autoencoder_layers,
                                            X_train_0_nor,
                                            X_train_nor,
                                            X_train_0_val_nor,
                                            X_train_val_nor,
                                            X_test_raw_0_nor,
                                            X_test_raw,
                                            verbose=0,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            dropout=0.2)
    print('{} layers\n'.format(len(autoencoder_layers)))
    print('loss:{}, val_loss:{},mse:{}'.format(loss[-1], val_loss[-1], mse))
    loss_list.append(loss[-1])
    val_loss_list.append(val_loss[-1])
    mse_list.append(mse)

# visualize
plt.figure(figsize=(8,6))
plt.plot(np.arange(1,20,2),loss_list, label='loss')
plt.plot(np.arange(1,20,2),val_loss_list, label='validation loss')
plt.xlabel('layers')
plt.ylabel('loss and val loss')

plt.legend()
plt.savefig('ss.png')
plt.show()


### neuron size test
loss_list = []
val_loss_list = []
mse_list = []
epochs = [100,50]
batch_size = [100,20]
for neuron_size in np.arange(1,2000,50):
    start = X_train_0_nor.shape[1]
    end = 561
    autoencoder_layers = [start]
    autoencoder_layers.extend(list(np.repeat(neuron_size,1)))
    autoencoder_layers.append(end)
    loss, val_loss, mse = autoencoder_test2(autoencoder_layers,
                                            X_train_0_nor,
                                            X_train_nor,
                                            X_train_0_val_nor,
                                            X_train_val_nor,
                                            X_test_raw_0_nor,
                                            X_test_raw,
                                            verbose=0,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            dropout=0.2)
    print('{} layers\n'.format(len(autoencoder_layers)))
    print('loss:{}, val_loss:{},mse:{}'.format(loss[-1], val_loss[-1], mse))
    loss_list.append(loss[-1])
    val_loss_list.append(val_loss[-1])
    mse_list.append(mse)

# visualize
plt.figure(figsize=(8,6))
plt.plot(np.arange(1,2000,50),loss_list, label='loss')
plt.plot(np.arange(1,2000,50),val_loss_list, label='validation loss')
plt.xlabel('number of neurons')
plt.ylabel('loss and val loss')

plt.legend()
plt.savefig('ss2.png')
plt.show()



# predict model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(512, input_dim=561, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(6, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# encode categorical data
encoder = LabelEncoder()
encoder.fit(y_train_raw)
encoded_Y = encoder.transform(y_train_raw)
dummy_y = np_utils.to_categorical(encoded_Y)

estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=20, verbose=1)
estimator.fit(X_train_raw, dummy_y, epochs=20, batch_size=20, verbose=1)
ynew = estimator.predict(X_test_raw)
ynew = ynew + 1

acc_tests = np.trace(confusion_matrix(y_test_raw, ynew)) / len(ynew)
acc_tests

ynew_recor = estimator.predict(X_test_raw_0_recover_denor)
ynew_recor = ynew_recor + 1

acc_tests_recor = np.trace(confusion_matrix(y_test_raw, ynew_recor)) / len(ynew_recor)
acc_tests_recor