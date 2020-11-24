import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import numpy as np
# from tensorflow.keras.utils import to_categorical

# 只用一种数据集 salinas
def load_HSI_Data():
    data_path = os.path.join(r'E:\Eric_HSI\hyperspectral_datasets')
    data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))
    labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))
    return data, labels


# 将dict to ndarray
def dict_to_array(X, y):
    data = X[list(X)[-1]]
    labels = y[list(y)[-1]]
    return data, labels

X, y = load_HSI_Data()
data, data_gt = dict_to_array(X, y)

# 原始数据转化为X_data 为 (4205, 5, 5, 204) # 21,445,500; 4386*25=109,650
X_data = []
step = 5
x = range(0, 510, step)
y = range(0, 215, step)
for i in x:
  for j in y:
    X_data.append(data[i:i+step, j:j+step])
print(len(X_data))
# print(X_data[0])
# print(X_data[0][0])
# print(len(X_data[0][0][0]))
X_data = np.array(X_data)[:4205]

# 这一行的作用是把5*5小方块中出现最多的像素作为这一个小方块中的类别编号,但是根据训练结果表明，这种效果很不好
# 标签数据 Y_data
import scipy.stats
Y_data = []
for i in x:
  for j in y:
      # 返回传入数组/矩阵中最常出现的成员以及出现的次数
    label = scipy.stats.mode(data_gt[i:i+step, j:j+step].reshape(step*step, 1))[0][0][0]
    Y_data.append(label)
Y_data = np.array(Y_data)

# 只用前4205个像素
Y_data = Y_data[:4205]
# from tensorflow.keras.utils import to_categorical
# Y_data = to_categorical(Y_data)


from sklearn.model_selection import train_test_split
(Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X_data, Y_data, test_size=0.8)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(256, (3, 3), input_shape=(5,5,204), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
# print("orgin shape", model.output.shape)
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
# print("afte MaxPool2D", model.output_shape)
# model.add(tf.keras.layers.MaxPool2D(3, 3))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
# print("afte MaxPool2D", model.output_shape)


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(17, activation='softmax', name='Salinas_Output'))

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=3, factor=0.5, min_lr=0.00001)

# optimizer =tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer="adam",
             loss='sparse_categorical_crossentropy',
             metrics=['acc']
             )


history = model.fit(Xtrain, 
                    Ytrain, 
                    epochs=50, 
                    validation_data=(Xtest, Ytest),
                    callbacks=[lr_reduce]
                    )

plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'], loc='best')
plt.show()


plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='best')
plt.show()
