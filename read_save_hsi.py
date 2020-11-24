import scipy.io as sio
import os
import numpy as np
from spectral import *
import matplotlib.pyplot as plt
""" 
我拥有的数据集
Indian_pines_corrected.mat   ----   Indian_pines_gt.mat
dict_keys(['__header__', '__version__', '__globals__', 'data'])
dict_keys(['__header__', '__version__', '__globals__', 'groundT'])

paviaU.mat   ---   paviaU_gt.mat
dict_keys(['__header__', '__version__', '__globals__', 'data'])
dict_keys(['__header__', '__version__', '__globals__', 'groundT'])

Salinas_corrected.mat   ---   Salinas_gt.mat
dict_keys(['__header__', '__version__', '__globals__', 'salinas_corrected'])
dict_keys(['__header__', '__version__', '__globals__', 'salinas_gt'])

Houston.mat   ---   Houston_GT.mat
dict_keys(['__header__', '__version__', '__globals__', 'Houston'])
dict_keys(['__header__', '__version__', '__globals__', 'Houston_GT'])

 """
#  load the Indian pines dataset which is the .mat format
def load_HSI_Data():
    data_path = os.path.join('F:\deeplearning\hyperspectral_datasets')
    # data = sio.loadmat(os.path.join(data_path, 'Houston.mat'))
    # labels = sio.loadmat(os.path.join(data_path, 'Houston_GT.mat'))

    # data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))
    # labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))

    # data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))
    # labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))

    data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))
    labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))

    return data, labels

X, y = load_HSI_Data()

# 打印keys
print(X.keys())
print(y.keys())

# 将dict to ndarray
def dict_to_array(X, y):
    data = X[list(X)[-1]]
    labels = y[list(y)[-1]]
    return data, labels

X, y = dict_to_array(X, y)

# 归一化
X = (X- float(np.min(X)))
X = X/np.max(X)

# 利用spectral显示 groundTruth,不灵活，不实用
# ground_truth = imshow(classes=y, figsize=(6, 6))
# plt.pause(10)  #显示秒数

# 检查文件夹是否存在
method_path = 'Geyi_HSI'
if not os.path.exists(os.path.join(method_path,'result')):
    os.makedirs(os.path.join(method_path,'result'))
# 保存文件(不灵活，不能指定保存图片的大小，不推荐，直接用上面的保存也行)
# save_rgb('Geyi_HSI/result/data.png', X, colors=spy_colors, format='png')

# 使用plt 显示保存文件，好用！
fig = plt.figure(figsize = (12, 6))

# 除去边框，坐标轴
plt.gcf().set_size_inches(512 / 100, 512 / 100)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.axis('off')

# 辅助修饰
# plt.colorbar()
# plt.title(f'hyperspectral image')

# 打印某一维高光谱数据
# q = np.random.randint(X.shape[2])
# plt.imshow(X[:,:,q], cmap='nipy_spectral')

# 打印并保存标签数据
plt.imshow(y, cmap='nipy_spectral')
# plt.imshow(y)
plt.savefig(os.path.join(r'Geyi_HSI\result', 'Houston.png'),format='png', dpi=600, bbox_inches='tight',pad_inches=0)#)
plt.show()



