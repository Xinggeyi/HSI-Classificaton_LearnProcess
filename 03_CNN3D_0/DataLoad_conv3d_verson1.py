import os
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, Dense, Flatten, Activation, BatchNormalization
import tensorflow as tf

from DataLoad_version1 import loadData

path_image=r'E:\Eric_HSI\hyperspectral_datasets\Indian_pines_corrected.mat'
path_label=r'E:\Eric_HSI\hyperspectral_datasets\Indian_pines_gt.mat'
key_image='data'
key_label='groundT'
window_size=19
input_size=200

X_train_new, X_val_new, y_train_new, y_val_new, train_samples = loadData(path_image, path_label, key_image, key_label)
print(X_train_new.shape, X_val_new.shape, y_train_new.shape, y_val_new.shape)
batch_size = 16
model = tf.keras.Sequential([
    Conv3D(16, kernel_size=(3, 3, 3), input_shape=(200, 19, 19, 1), strides=(5, 1, 1)),  # 17
    BatchNormalization(),
    Activation(tf.nn.relu),
    Conv3D(16, kernel_size=(3, 3, 3), padding='same'),
    BatchNormalization(),
    Activation(tf.nn.relu),
    MaxPooling3D(pool_size=2),  # 8
    Conv3D(32, kernel_size=(3, 3, 3), padding='same'),
    BatchNormalization(),
    Activation(tf.nn.relu),
    Conv3D(32, kernel_size=(3, 3, 3), padding='same'),
    BatchNormalization(),
    Activation(tf.nn.relu),
    MaxPooling3D(pool_size=2),  # 4
    Conv3D(64, kernel_size=(3, 3, 3), padding='same'),
    BatchNormalization(),
    Activation(tf.nn.relu),
    Conv3D(64, kernel_size=(3, 3, 3), padding='same'),
    BatchNormalization(),
    Activation(tf.nn.relu),
    MaxPooling3D(pool_size=2),  # 2
    Flatten(),
    Dense(128),
    BatchNormalization(),
    Activation(tf.nn.relu),
    Dropout(0.5),
    Dense(16, activation=tf.nn.softmax)]
)
# epoch = 100
epoch = 10
tf.keras.backend.set_learning_phase(True)
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(0.0003),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy])

# 如果是第一次训练，必须点击 n，如果是第二次训练，这里可以点击 y

model_dir = r'E:\Eric_HSI\excise2\weights'
model_file = 'model_weights'
model_saved_path = model_dir + '/' + model_file
is_load_model = input('Would you like load the existed model weights if it exist ? y/n\n')
if is_load_model == 'y':
    model.load_weights(model_saved_path)
    print('An existed model_weight table has been gotten...')
else:
    print('A new model will be gotten...')
print('*****************************************')
epoch = int(input('Please input epoch : '))
if epoch < 0:
    epoch = 0
print('*****************************************')
if epoch != 0:
    hist = model.fit(
        X_train_new,
        y_train_new,
        batch_size=batch_size,
        epochs=epoch,
        shuffle=True
    )
loss, acc = model.evaluate(
    X_val_new,
    y_val_new,
    batch_size=batch_size
)


print('acc:', acc)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model.save_weights(r'E:\Eric_HSI\excise2\weights\model_weight.h5')
print('model_weights have been saved at ' + model_dir)