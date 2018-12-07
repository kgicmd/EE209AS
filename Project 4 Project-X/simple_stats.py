# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:49:17 2018

@author: kgicmd
"""


import numpy as np
import os
os.chdir('E:\\har')

import pandas as pd
from util import plot_confusion_matrix
import matplotlib.pyplot as plt

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

class_names = ['WALKING',
           'WALKING_UPSTAIRS',
           'WALKING_DOWNSTAIRS',
           'SITTING',
           'STANDING',
           'LAYING']

plt.bar(np.arange(6),np.unique(y_train_raw, return_counts=True)[1])
plt.xlabel('categories')
plt.ylabel('count')
plt.xticks(np.arange(6), class_names, rotation=90)
plt.title('Distribution of training')
plt.show()

plt.bar(np.arange(6),np.unique(y_test_raw, return_counts=True)[1])
plt.xlabel('categories')
plt.ylabel('count')
plt.xticks(np.arange(6), class_names, rotation=90)
plt.title('Distribution of testing')
plt.show()

# calculate avg drop
acc_list_df = pd.read_csv('acc_list.csv')
del acc_list_df['Unnamed: 0']
acc_acc_drop = acc_list_df.iloc[8,:] - acc_list_df.iloc[0,:]
acc_gyro_drop = acc_list_df.iloc[8,:] - acc_list_df.iloc[4,:]
acc_acc_axis_drop = (acc_list_df.iloc[8,:]*3 - acc_list_df.iloc[1,:] - acc_list_df.iloc[2,:] - acc_list_df.iloc[3,:])/3
acc_gyro_axis_drop = (acc_list_df.iloc[8,:]*3 - acc_list_df.iloc[5,:]- acc_list_df.iloc[6,:]- acc_list_df.iloc[7,:])/3

# plot figures
ind = np.arange(6)
width = 0.2

fig, ax = plt.subplots(figsize=(9,6))
rects1 = ax.bar(ind - width, acc_acc_drop, width, color='#06858C')
rects2 = ax.bar(ind, acc_gyro_drop, width, color='#45C48B')
rects3 = ax.bar(ind + width, acc_acc_axis_drop, width, color='#FFD038')
rects4 = ax.bar(ind + 2*width, acc_gyro_axis_drop, width, color='#F47942')

# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy drop')
ax.set_title('Accuracy drop by classification methods')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('GBDT', 'RF', 'CART', 'SVM', 'NB','ANN','ANN2'))

ax.legend((rects1[0], rects2[0],rects3[0],rects4[0]), ('w/o Acc','w/o Gyro','avg w/o Acc axis','avg w/o Gyro axis'))
plt.show()

# stacked bar
plt.figure(figsize=(9,6))
width = 0.5
p1 = plt.bar(ind, acc_acc_drop, width, color="#06858C")
p2 = plt.bar(ind, acc_gyro_drop, width,bottom=acc_acc_drop, color="#45C48B")
p3 = plt.bar(ind, acc_acc_axis_drop, width,bottom=acc_acc_drop+acc_gyro_drop, color="#FFD038")
p4 = plt.bar(ind, acc_gyro_axis_drop, width,bottom=acc_acc_drop+acc_gyro_drop+acc_acc_axis_drop,color="#F47942")

plt.ylabel('Accuracy drop')
plt.title('Accuracy drop by classification methods')
plt.xticks(ind, ('GBDT', 'RF', 'CART', 'SVM', 'NB','ANN'))
plt.yticks(np.arange(0, 1.8, 0.3))
plt.legend((p1[0], p2[0],p3[0],p4[0]),
           ('w/o Acc','w/o Gyro','avg w/o Acc axis','avg w/o Gyro axis'),
           bbox_to_anchor=(1.04,1),
           loc="upper left")
plt.show()