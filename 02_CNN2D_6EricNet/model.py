import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import os
import matplotlib.pyplot as plt
import numpy as np 
import glob
import os
import time
from tqdm import tqdm


class NetBlock(keras.Model):
    
    def __init__(self):
        super(NetBlock, self).__init__()

        self.conv1 = layers.Conv2D(204, (1, 1), activation='relu', use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(102, (1, 1), activation='relu', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.dconv1 = layers.Conv2DTranspose(102, 3, 2, 'valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()

        self.dconv2 = layers.Conv2DTranspose(64, 3, 2, 'valid', use_bias=False)
        self.bn4 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', use_bias=False)
        self.bn5 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(128, (3, 3), activation='relu', use_bias=False)
        self.bn6 = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(256, (3, 3), activation='relu', use_bias=False)
        self.bn7 = layers.BatchNormalization()

        self.conv6 = layers.Conv2D(256, (3, 3), activation='relu', use_bias=False)
        self.bn8 = layers.BatchNormalization()

        # self.averagepool = layers.GlobalMaxPooling2D()
        # self.dense1 = layers.Dense(256, activation='relu')
        # self.dropout1 = layers.Dropout(0.5)
        # self.dense2 = layers.Dense(512, activation='relu')
        # self.dropout2 = layers.Dropout(0.5)
        # self.dense3 = layers.Dense(16, activation='softmax')

    # forward
    def call(self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.dconv1(x)
        x = tf.nn.leaky_relu(x)
        x = self.bn3(x)

        x = self.dconv2(x)
        x = tf.nn.leaky_relu(x)
        x = self.bn4(x)

        x = self.conv3(x)
        x = self.bn5(x)

        x = self.conv4(x)
        x = self.bn6(x)

        x = self.conv5(x)
        x = self.bn7(x)

        x = self.conv6(x)
        x = self.bn8(x)

        # x = self.averagepool(x)
        # x = self.dense1(x)
        # x = self.dropout1(x)
        # x = self.dense2(x)
        # x = self.dropout2(x)
        # x = self.dense3(x)

        return x

def _net(block, im_width=9, im_height=9, im_channel=204, num_classes=16, include_top=True):
    # tensorflow中的tensor通道排序是NHWC
    # (None, 9, 9, 204)
    # change
    input_image = layers.Input(shape=(im_height, im_width, im_channel), dtype="float32")
    # x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="SAME", use_bias=False, name="conv1")(input_image)
    x = block()(input_image)
    # print("include_top", include_top)

    if include_top:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten

        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        predict = layers.Dense(16, activation='softmax')(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model

def net(im_width=9, im_height=9, im_channel=204 ,num_classes=16, include_top=True):
    # print("include_top", include_top)
    return _net(NetBlock, im_width, im_height, im_channel, num_classes, include_top)


# model = (im_width=9, im_height=9, im_channel=204 ,num_classes=16, include_top=True)
# model.build(input_shape=(None, 9, 9, 204))
# model.summary()

# model = (im_width=9, im_height=9, im_channel=204 ,num_classes=16, include_top=True)
# model.build(input_shape=(None, 9, 9, 204))
# model.summary()