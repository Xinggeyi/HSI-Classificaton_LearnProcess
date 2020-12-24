import tensorflow as tf 
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import models, Model
from tensorflow import keras
from tensorflow.keras import layers, Sequential, datasets, optimizers


def VGG(feature, im_height=224, im_width=224, class_num=1000, im_channel=204):
    # tensorflow中的tensor通道排序是NHWC
    # change
    # input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    input_image = layers.Input(shape=(im_height, im_width, im_channel), dtype="float32")
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(class_num)(x)
    output = layers.Softmax()(x)
    model = models.Model(inputs=input_image, outputs=output)
    return model

def features(cfg):
    feature_layers = []
    for v in cfg:
        if v == "M":
            # change
            # feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=1))
        else:
            conv2d = layers.Conv2D(v, kernel_size=3, padding="SAME", activation="relu")
            feature_layers.append(conv2d)
    return Sequential(feature_layers, name="feature")


    # 网络配置列表
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name="vgg16", im_height=224, im_width=224, class_num=1000, im_channel=204):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(features(cfg), im_height=im_height, im_width=im_width, class_num=class_num, im_channel=204)
    return model


# 实例化模型
# model = vgg(model_name="vgg16", im_height=9, im_width=9, class_num=16, im_channel=204)
# model.summary()


def vgg13(model_name="vgg13", im_height=9, im_width=9, class_num=16, im_channel=204):
    return vgg(model_name="vgg13", im_height=9, im_width=9, class_num=16, im_channel=204)

def vgg16(model_name="vgg16", im_height=9, im_width=9, class_num=16, im_channel=204):
    return vgg(model_name="vgg16", im_height=9, im_width=9, class_num=16, im_channel=204)