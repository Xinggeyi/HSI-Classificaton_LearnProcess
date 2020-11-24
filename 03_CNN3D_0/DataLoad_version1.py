# 两个改进方向
# 把处理后的数据储存一起来，方便下次调用，可以节约内存
# 添加padding， 使用更多地数据
# 首先要理解这个代码！！还没有完全理解！

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def loadData(path_image='E:\Eric_HSI\hyperspectral_datasets\Indian_pines_corrected.mat',
             path_label='Eric_HSI\hyperspectral_datasets\Indian_pines_gt.mat',
             key_image='indian_pines_corrected',
             key_label='indian_pines_gt',
             window_size=19, input_size=200):
    mat = loadmat(path_image)
    # print(mat.keys())
    features = mat[key_image]
    features_shape = features.shape

    mat_labels = loadmat(path_label)
    labels = mat_labels[key_label]
    labels = np.reshape(labels, (-1, 1))
    f2 = np.zeros((labels.shape[0], window_size, window_size, input_size), dtype='float32')
    parameter_b = np.array([i - window_size // 2 for i in range(window_size)])
    for i in range(features_shape[0]):
        for j in range(features_shape[1]):
            index = i * features_shape[1] + j
            for p in range(window_size):
                for q in range(window_size):
                    f2[index][p][q] = \
                        features[(i + parameter_b[p]) % features_shape[0]][(j + parameter_b[q]) % features_shape[1]]

    index = np.where(np.reshape(labels, (-1)) == 0)
    labels = np.delete(labels, index, axis=0)
    # print(np.unique(labels))
    tmp = np.unique(labels)
    for i in range(len(labels)):
        labels[i][0] = np.where(tmp == labels[i][0])[0][0]
    f2 = np.delete(f2, index, axis=0)
    labels = np.reshape(labels, (-1))
    dataset = [f2, labels]
    (X, Y) = (dataset[0], dataset[1])  # -1,19,19,200
    X = X.swapaxes(1, 3)
    print('X_shape: ', X.shape)
    train_set = X[:, :, :, :, np.newaxis]
    print('TrainSet_shape: ', train_set.shape)
    classes = 16
    Y = tf.keras.utils.to_categorical(Y, classes)
    train_set = train_set.astype('float32')
    train_set -= np.mean(train_set)
    train_set /= np.max(train_set)
    # Split the data
    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_set, Y, test_size=0.8, random_state=4)
    return X_train_new, X_val_new, y_train_new, y_val_new, labels.shape[0]