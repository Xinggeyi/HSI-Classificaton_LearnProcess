
	1. 读取数据：(145, 145, 200), (145, 145)
	2. PCA降维: (145, 145, 30)
	3. padWithZeros，边缘填充零0，为以后的创建patches做准备
	4. createPactches，将一整张图片划分成一个个小的patch，并将背景像素删除。TotalPatNum，width，height，channels。((10366, 5, 5, 30), (10366,))
	5. splitTrainTestSet，测试比0.8， (2049, 5, 5, 200) (8200, 5, 5, 200) (2049,) (8200,)
	6. oversampleWeakClasses，将比例比较少的例子重复几次，然后叠加到原始数据上，训练集ndarray: ((7919, 5, 5, 200), (7919,))
	7. 数据标准化
	8. 数据增强
	9. 保存数据

