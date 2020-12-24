import tensorflow as tf 
from  tensorflow import keras
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 由一个 z = tf.random.normal([b, 100]) 的正态分布开始生成
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # z: [b, 100] => [b, 3*3*512] => [b, 3, 3, 512] => ... => [b, 64, 64, 3]
        # filters(从大到小)， kernels（一般小于7）， strides， padding， 整个流程下来，输出要正好等于[b, 64, 64, 3]
        self.fc = layers.Dense(3*3*512)
        
        # [b, 3, 3, 512] => [b, 9, 9, 256]
        # # 转置卷积层1,输出channel为256, 核大小3, 步长3, 不使用padding, 因为BN层所以不使用偏置
        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid', use_bias=False)
        self.bn1= layers.BatchNormalization()

        # [b, 9, 9, 256] => [b, 21, 21, 128], k=5, s=2
        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        # [b, 21, 21, 128] => [b, 64, 64, 3]
        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid', use_bias=False)
        self.bn3= layers.BatchNormalization()

    def call(self, inputs, training=None):
        # [z, 100] => [z, 3*3*512]
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        # 这里的 bn1， bn2 是有参数，不能两层同用一个，否则会出错
        # [b, 9, 9, 256]
        x = self.bn1(self.conv1(x), training=training)
        x = tf.nn.leaky_relu(x)

        # [b, 21, 21, 128]
        x = self.bn2(self.conv2(x), training=training)
        x = tf.nn.leaky_relu(x)

        # [b, 64, 64, 3]
        x = self.conv3(x)
        #  # 输出x范围-1~1,与预处理一致
        x = tf.tanh(x)

        return x


# 普通的分类器
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # filters， kernels， strides， padding
        self.conv1 = layers.Conv2D(64, 5, 3, 'valid')
        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2= layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3= layers.BatchNormalization()
        
        # 特征打平
        self.flatten = layers.Flatten()
        
        # 2 分类全连接层
        self.fc = layers.Dense(1)


    def call(self, inputs, training=None):
        
        # [b,64,64,3] =>  [b,17,17,64]
        x = tf.nn.leaky_relu(self.conv1(inputs))

        # [b,5,5,128]
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))

        # [b,1,1,256]
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        # print(x.shape)

        x = self.flatten(x)
        logits = self.fc(x)

        return logits


# # test
# d = Discriminator()
# g = Generator()

# # x 模拟真实的分布
# x = tf.random.normal([2,64,64,3])
# # z 模拟生成器的输入
# z = tf.random.normal([2,100])

# # 判别器产生 0 1 之间的真和假
# prob = d(x)
# print(prob)  # tf.Tensor([[ 0.0872746 ] [-0.02736489]], shape=(2, 1), dtype=float32)

# # g(z) 产生生成的图像
# x_hat = g(z)
# print(x_hat.shape)   # (2, 64, 64, 3)









