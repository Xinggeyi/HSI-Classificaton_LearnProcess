import numpy as np
import matplotlib.pyplot as plt


a = np.array(
 [[2115, 0, 14, 0, 2, 5, 43, 43, 1],
 [1, 5730, 0, 41, 0, 1222, 0, 3, 0],
 [28, 5, 1772, 0, 0, 1, 0, 156, 0],
 [0, 247, 0, 2795, 0, 1, 0, 0, 0],
 [9, 0, 0, 1, 7141, 0, 0, 0, 0],
 [56, 1264, 1,17, 0, 4590, 0, 9, 0],
 [173, 0, 0, 0, 2, 0, 1084, 0, 0],
 [349, 3, 112, 0, 0, 10, 3, 3271, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 746]])
a.shape
# a = a[:, np.newaxis]
# a.astype(float)
plt.matshow(a, cmap=plt.cm.GnBu) #画混淆矩阵图，配色风格使用
plt.axis("off")
# plt.show()
plt.savefig(r'./xx.png', dpi=600, format='png')