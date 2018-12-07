# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:00:03 2018

@author: kgicmd
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:04:21 2018

@author: kgicmd
"""

# using autoencoder to refill missing data
import numpy as np
import os
os.chdir('E:\\har')

from util import *

import pandas as pd
from util import plot_confusion_matrix, import_data, mask_source_channel
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

# implement autoencoder
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

LEARNING_RATE = 7e-4

def test_auto_encoder_with_data(control_seq, estimator):
    """
    test data with control seq missed
    """
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = import_data()
    
    X_train_raw_0 = mask_source_channel(control_seq, X_train_raw, 1)
    X_test_raw_0 = mask_source_channel(control_seq, X_test_raw, 1)
    
    # split train data into 5 folds
    kf = KFold(n_splits=5, shuffle=True)
    
    for train_index, validation_index in kf.split(X_train_raw):
        pass
    # raw data
    X_train_raw_train = X_train_raw.iloc[train_index]
    X_train_raw_validation = X_train_raw.iloc[validation_index]
    # missed features data
    X_train_raw_0_train = X_train_raw_0.iloc[train_index]
    X_train_raw_0_validation = X_train_raw_0.iloc[validation_index]
    
    # drop na
    X_train_raw_0_train_dropna = X_train_raw_0_train.dropna(axis=1)
    X_test_raw_0_dropna = X_test_raw_0.dropna(axis=1)
    
    # normalize data
    X_train_all_dropna_normalize = normalize_data(X_train_raw_0_train_dropna, X_train_raw_train)
    X_train_raw_train_normalize = normalize_data(X_train_raw_train, X_train_raw_train)
    
    X_test_raw_0_dropna_normalize = normalize_data(X_test_raw_0_dropna, X_test_raw)
    
    
    # this is the size of our encoded representations
    # we know that there are 561 features...
    # reduce to 28 dim (compr rate = 20)
    encoding_dim = 4
      
    # this is our input placeholder  
    input_seq = Input(shape=(X_train_raw_0_train_dropna.shape[1],))  
      
    # encode layer   
    encoded = Dense(128, activation='relu')(input_seq)
    #encoded = Dense(64, activation='relu')(encoded) 
    encoded = Dense(16, activation='relu')(encoded)
    encoded = Dense(8, activation='relu')(encoded) 
    encoder_output = Dense(encoding_dim)(encoded)  
      
    # decode layer 
    decoded = Dense(16, activation='relu')(encoder_output)  
    #decoded = Dense(64, activation='relu')(decoded)  
    decoded = Dense(128, activation='relu')(decoded)  
    decoded = Dense(561, activation='sigmoid')(decoded)  
      
    # construct autoencoder
    autoencoder = Model(inputs=input_seq, outputs=decoded)
     
    # compile autoencoder  
    autoencoder.compile(optimizer=Adam(lr = LEARNING_RATE), loss='mean_squared_error')
     
    autoencoder.summary()
     
    # training  
    from keras.callbacks import TensorBoard
    
    autoencoder.fit(X_train_all_dropna_normalize, X_train_raw_train_normalize,
                    epochs=300,
                    batch_size=100,
                    #shuffle=True,
                    #validation_data=(X_train_raw_0_validation, X_train_raw_validation),
                    callbacks=[TensorBoard(log_dir='./tmp/autoencoder')])
    # tensorboard --logdir=E:\har\tmp\autoencoder
    
    # predict and denormalize data
    X_test_raw_refilled = autoencoder.predict(X_test_raw_0_dropna_normalize)
    X_test_raw_refilled_denor = denormalize_data(X_test_raw_refilled, X_test_raw)
    
    
    ynew = estimator.predict(X_test_raw_refilled_denor)
    ynew = ynew + 1
    
    acc_tests = np.trace(confusion_matrix(y_test_raw, ynew)) / len(ynew)
    return acc_tests

class_names = ['WALKING',
           'WALKING_UPSTAIRS',
           'WALKING_DOWNSTAIRS',
           'SITTING',
           'STANDING',
           'LAYING']

plot_confusion_matrix(confusion_matrix(y_test_raw, ynew), classes=class_names)