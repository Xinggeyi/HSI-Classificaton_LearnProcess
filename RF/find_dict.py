# -*-coding=utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import os

path = os.path.join("F:\deeplearning\cla")  # change this path for your dataset
PaviaU = os.path.join(path,'F:\deeplearning\cla\paviaU.mat')
PaviaU_gt = os.path.join(path,'F:\deeplearning\cla\paviaU_gt.mat')
method_path = 'SVM'

# 加载数据
data = sio.loadmat(PaviaU)
data_gt = sio.loadmat(PaviaU_gt)
print(type(data))
print(data.keys())
print(data_gt.keys())