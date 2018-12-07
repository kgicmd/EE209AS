# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:22:10 2018

@author: kgicmd
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

def import_data():
    # import data
    X_train_raw = pd.read_csv('E:\\har\\train\\X_train.txt', delimiter=r"\s+", header=None)
    y_train_raw = pd.read_csv('E:\\har\\train\\y_train.txt', sep=' ',header=None)
    
    X_test_raw = pd.read_csv('E:\\har\\test\\X_test.txt', delimiter=r"\s+", header=None)
    y_test_raw = pd.read_csv('E:\\har\\test\\y_test.txt', sep=' ',header=None)
    
    
    # read feature name
    feature_list = []
    feature_name_file = open('E:\\har\\features.txt')
    for line in feature_name_file.readlines():
        # '1 tBodyAcc-mean()-X\n'
        line.rstrip()
        feature_list.append(line.split(' ')[1])
    
    feature_name_file.close()
    
    X_train_raw.columns = feature_list
    X_test_raw.columns = feature_list
    
    return X_train_raw, y_train_raw, X_test_raw, y_test_raw

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Quoted from: sklearn example
        
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def test_k_fold(X_transform , y, clf):
    """
    Input:  X_transform: transformed train data
            y : 1-d label array
            clf : classifier
    Output : acc list
             fitted clf
    """
    kf = KFold(n_splits=5, shuffle=True)
    acc = []
    for train_index, test_index in kf.split(X_transform):
        X_train, X_test = X_transform[train_index], X_transform[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_used = clf
        clf_used.fit(X_train, y_train)
        y_pred = clf_used.predict(X_test)
        # calculate accuracy
        acc_fld = np.trace(confusion_matrix(y_test, y_pred)) / len(y_pred)
        acc.append(acc_fld)
    return acc, clf_used

def test_clf(X_test_transform, y_test, clf):
    """
    Input:  X_test_transform: transformed test data
            y_test : 1-d label array
            params : parameter dictionary
            clf : classifier
    Output: cm: confusion matrix
            acc_test: testing accuracy 
    """
    y_pred_test = clf.predict(X_test_transform)
    # calculate accuracy
    cm = confusion_matrix(y_test, y_pred_test)
    acc_test = np.trace(cm) / len(y_pred_test)
    
    return cm, acc_test

def mask_source_channel(channel, X_test_data, option=0):
    """
    Input : channel[0] : acc / gyro
            channel[1] : -X, -Y, -Z (or '')
    Output: masked dataset (replace with 0)
    """
    X_test_data2 = X_test_data.copy()
    col_list = X_test_data2.columns
    
    del_list_index = []
    if len(channel) != 0:
        for ind in np.arange(len(col_list)):
            # if delete an axis of a sensor
            if (len(channel) == 2):
                # find device and axis
                if col_list[ind].find(channel[0]) != -1:
                    if col_list[ind].find(channel[1]) != -1:
                        # record in delete list
                        del_list_index.append(ind)
            else:
            # else, delete a sensor
                if col_list[ind].find(channel[0]) != -1:
                    del_list_index.append(ind)
        
        
        if(option == 0):
            X_test_data2.iloc[:,del_list_index] = 0
        else:
            X_test_data2.iloc[:,del_list_index] = np.nan
        
        return X_test_data2
    else:
        return X_test_data2

def normalize_data(data):
    """
    normalize data to 0-1
    """
    data = np.array(data)
    max_value = np.max(data)
    min_value = np.min(data)
    
    new_data = (data - min_value) / (max_value - min_value)
    return new_data

def denormalize_data(data, original_data):
    """
    map data to 0-1 to original
    """
    data = np.array(data)
    original_data = np.array(original_data)
    
    max_value = np.max(original_data)
    min_value = np.min(original_data)
    
    new_data = data * (max_value - min_value) + min_value
    return new_data    