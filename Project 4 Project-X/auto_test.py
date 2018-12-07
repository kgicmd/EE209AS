# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:47:03 2018

@author: kgicmd
"""

import numpy as np
import os
os.chdir('E:\\har')

import pandas as pd
from util import plot_confusion_matrix

# feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.validation import column_or_1d

# grid cv
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA
# utils
from util import *

# GBDT
from sklearn.ensemble import GradientBoostingClassifier
# RF
from sklearn.ensemble import RandomForestClassifier
# CART
from sklearn.tree import DecisionTreeClassifier
# SVM
from sklearn.svm import SVC
# NB
from sklearn.naive_bayes import GaussianNB
# ANN
from sklearn.neural_network import MLPClassifier

def gbdt_test(X_train_raw, y_train_raw, X_test_raw, y_test_raw):
    # select features with GBDT
    y = column_or_1d(y_train_raw, warn=False)
    clf_gbdt = GradientBoostingClassifier()
    sel = SelectFromModel(clf_gbdt)
    sel.fit(X_train_raw, y)
    
    # transform data
    X_transform = sel.transform(X_train_raw)
    X_test_transform = sel.transform(X_test_raw)
    
    # optimum params
    params = {'n_estimators': 100,
              'max_leaf_nodes': 4,
              'max_depth': None,
              'random_state': 2,
              'min_samples_split': 5}
    clf_gbdt = GradientBoostingClassifier(**params)
    # fit
    clf_gbdt.fit(X_transform, y)
    
    y_pred_test = clf_gbdt.predict(X_test_transform)
    # calculate accuracy
    acc_test = np.trace(confusion_matrix(y_test_raw, y_pred_test)) / len(y_pred_test)
    
    return acc_test

def rf_test(X_train_raw, y_train_raw, X_test_raw, y_test_raw):
    # select features with RF
    y = column_or_1d(y_train_raw, warn=False)
    clf_rf = RandomForestClassifier()
    # find most fitted features
    sel = SelectFromModel(clf_rf)
    sel.fit(X_train_raw, y)
    
    # transform data
    X_transform = sel.transform(X_train_raw)
    X_test_transform = sel.transform(X_test_raw)
    y_test_1d = column_or_1d(y_test_raw, warn=False)
    
    # model
    best_params = {'n_estimators': 100,
                   'max_depth' : 32,
                   'max_features': 2}
    clf_rf_opt = RandomForestClassifier(**best_params)
    clf_rf_opt.fit(X_transform, y)
    
    # test
    y_pred_test = clf_rf_opt.predict(X_test_transform)
    # calculate accuracy
    acc_test = np.trace(confusion_matrix(y_test_1d, y_pred_test)) / len(y_pred_test)
    
    return acc_test

def cart_test(X_train_raw, y_train_raw, X_test_raw, y_test_raw):
    # select features w/ CART
    y = column_or_1d(y_train_raw, warn=False)
    clf_dt = DecisionTreeClassifier()
    # find most fitted features
    sel_dt = SelectFromModel(clf_dt)
    sel_dt.fit(X_train_raw, y)
    
    # transform
    X_transform_dt = sel_dt.transform(X_train_raw)
    X_test_transform_dt = sel_dt.transform(X_test_raw)
    y_test_1d = column_or_1d(y_test_raw, warn=False)
    
    # model
    clf_dt_opt = DecisionTreeClassifier()
    clf_dt_opt.fit(X_transform_dt, y)
    
    # test
    y_pred_test = clf_dt_opt.predict(X_test_transform_dt)
    # calculate accuracy
    acc_test = np.trace(confusion_matrix(y_test_1d, y_pred_test)) / len(y_pred_test)
    
    return acc_test  

def svm_test(X_train_raw, y_train_raw, X_test_raw, y_test_raw):
    # select features w/ SVM
    y = column_or_1d(y_train_raw, warn=False)
    # use PCA
    pca = PCA(n_components=10)
    pca.fit(X_train_raw)

    # PCA compressed X_train and X_test
    X_train_raw_compr = pca.transform(X_train_raw)
    X_test_raw_compr = pca.transform(X_test_raw)
    y_test_1d = column_or_1d(y_test_raw, warn=False)

    # best parameters and svc
    best_params = {'gamma': 0.001,
                   'C': 1000}
    clf_svm_opt = SVC(**best_params)
    clf_svm_opt.fit(X_train_raw_compr, y)
    
    # test
    y_pred_test = clf_svm_opt.predict(X_test_raw_compr)
    acc_test = np.trace(confusion_matrix(y_test_1d, y_pred_test)) / len(y_pred_test)
    
    return acc_test

def nb_test(X_train_raw, y_train_raw, X_test_raw, y_test_raw):
    # select features w/ NB
    y = column_or_1d(y_train_raw, warn=False)
    # use PCA
    pca = PCA(n_components=10)
    pca.fit(X_train_raw)  
    
    # PCA compressed X_train and X_test
    X_train_raw_compr = pca.transform(X_train_raw)
    X_test_raw_compr = pca.transform(X_test_raw)
    y_test_1d = column_or_1d(y_test_raw, warn=False)
    
    clf_nb = GaussianNB()
    clf_nb.fit(X_train_raw_compr, y)
    
    # test
    y_pred_test = clf_nb.predict(X_test_raw_compr)
    acc_test = np.trace(confusion_matrix(y_test_1d, y_pred_test)) / len(y_pred_test)
    
    return acc_test

def ann_test(X_train_raw, y_train_raw, X_test_raw, y_test_raw):
    # No feature selection
    y = column_or_1d(y_train_raw, warn=False)
    clf_ANN = MLPClassifier(solver='adam',
                            alpha=1e-5,
                            activation='relu',
                            learning_rate='adaptive',
                            max_iter=400,
                            hidden_layer_sizes=(100, 3),
                            random_state=123)
    
    X_train_raw_ANN = X_train_raw.values
    X_test_raw_ANN = X_test_raw.values
    y_test_1d = column_or_1d(y_test_raw, warn=False)
    
    clf_ANN.fit(X_train_raw_ANN, y)
    
    # testing
    y_pred_test = clf_ANN.predict(X_test_raw_ANN)
    acc_test = np.trace(confusion_matrix(y_test_1d, y_pred_test)) / len(y_pred_test)
    
    return acc_test

def test_overall(control_seq):
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = import_data()
    
    if(control_seq):
        X_test_raw_0 = mask_source_channel(control_seq, X_test_raw)
    else:
        X_test_raw_0 = X_test_raw.copy(deep=True)
    # gbdt
    acc_gbdt = gbdt_test(X_train_raw, y_train_raw, X_test_raw_0, y_test_raw)
    # rf
    acc_rf = rf_test(X_train_raw, y_train_raw, X_test_raw_0, y_test_raw)
    # CART
    acc_cart = cart_test(X_train_raw, y_train_raw, X_test_raw_0, y_test_raw)
    # SVM
    acc_svm = svm_test(X_train_raw, y_train_raw, X_test_raw_0, y_test_raw)
    # NB
    acc_nb = nb_test(X_train_raw, y_train_raw, X_test_raw_0, y_test_raw)
    # ANN
    acc_ann = ann_test(X_train_raw, y_train_raw, X_test_raw_0, y_test_raw)
    
    # compressed to a list
    acc_list = [acc_gbdt, acc_rf, acc_cart, acc_svm, acc_nb, acc_ann]
    
    return acc_list
    
if __name__ == "__main__":
    acc_list = []
    control_seq_list = [
                        ['Acc'],
                        ['Acc','-X'],
                        ['Acc','-Y'],
                        ['Acc','-Z'],
                        ['Gyro'],
                        ['Gyro','-X'],
                        ['Gyro','-Y'],
                        ['Gyro','-Z'],
                        []]
    
    for control_seq in control_seq_list:
        acc_list.append(test_overall(control_seq))
        
    print(acc_list)
    acc_list_df = pd.DataFrame(acc_list)

    acc_list_df.to_csv('acc_list.csv')