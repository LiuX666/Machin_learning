# -*- coding: UTF-8 -*-
"""
@author:liux
@time:2020/08/17
@func: 使用梯度上升找到最佳参数
       示例：使用Logistic回归估计马疝病的死亡率
"""
import numpy as np
import random


class GradAscentSimple:
    """
    z = wTx ,梯度上升法的伪代码如下：
                  每个回归系数初始化为1
                  重复R次：
                          计算整个数据集的梯度
                          使用alpha × gradient更新回归系数的向量
                          返回回归系数
    """

    @staticmethod
    def loadDataSet():
        """
        加载数据
        :return:   dataMat - 数据列表   labelMat - 标签列表
        """
        dataMat = []
        labelMat = []
        fr = open('./Data/testSet.txt')
        for line in fr.readlines():
            # 去掉每行两边的空白字符，并以空格分隔每行数据元素
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))

        fr.close()
        return dataMat, labelMat

    @staticmethod
    def sigmoid(inX):
        return 1.0 / (1 + np.exp(-inX))

    def gradAscent(self, dataMath, classLabels):
        """
        梯度上升法
        :param dataMath:      数据集
        :param classLabels:  数据标签
        :return:  weights.getA() - 求得的权重数组（最优参数）
                   weights_array - 每次更新的回归系数
        """
        dataMatrix = np.mat(dataMath)
        labelMat = np.mat(classLabels).transpose()                      # 转换成numpy的mat(矩阵)并进行转置
        m, n = np.shape(dataMatrix)                                     # 返回dataMatrix的大小，m为行数，n为列数

        alpha = 0.01                             # 移动步长，也就是学习效率，控制更新的幅度
        maxCycles = 500                          # 最大迭代次数

        weights = np.ones((n, 1))
        weights_array = np.array([])
        for k in range(maxCycles):
            # 梯度上升矢量化公式,Sigmoid函数
            h = self.sigmoid(dataMatrix * weights)
            error = labelMat - h
            weights = weights + alpha * dataMatrix.transpose() * error
            weights_array = np.append(weights_array, weights)
        weights_array = weights_array.reshape(maxCycles, n)

        # mat.getA()将自身矩阵变量转化为ndarray类型变量
        return weights.getA(), weights_array


class RandomGradAsc:
    """
    所有回归系数初始化为1
    对数据集中每个样本
        计算该样本的梯度
        使用alpha × gradient更新回归系数值
    返回回归系数值
    """
    @staticmethod
    def stocGradAscent(dataMatrix, classLabels, numIter=150):
        """
        随机梯度上升算法,没有矩阵的转换过程，所有变量的数据类型都是NumPy数组
        :param numIter: 迭代次数
        :return:
        """
        dataMatrix = np.array(dataMatrix)
        m, n = np.shape(dataMatrix)
        weights = np.ones(n)
        weights_array = np.array([])
        try:
            for j in range(numIter):
                dataIndex = list(range(m))
                for i in range(m):
                    alpha = 4 / (1.0 + j + i) + 0.01  # 每次都降低alpha的大小
                    randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选择样本
                    h = GradAscentSimple().sigmoid(sum(dataMatrix[randIndex] * weights))  # 随机选择一个样本计算h
                    error = classLabels[randIndex] - h  # 计算误差
                    weights = weights + alpha * error * dataMatrix[randIndex]
                    weights_array = np.append(weights_array, weights, axis=0)
                    del (dataIndex[randIndex])  # 删除已使用的样本
        except Exception as e:
            print(e)

        weights_array = weights_array.reshape(numIter * m, n)
        return weights, weights_array


# dataMat, labelMat = GradAscentSimple().loadDataSet()
# weights, weights_array = RandomGradAsc().stocGradAscent(dataMat, labelMat)
# print(weights)
# print(weights_array)


class TestLogistic:
    """
       使用Logistic回归来预测患有疝病的马的存活问题,如果Sigmoid值大于0.5函数返回1，否则返回0
       使用Logistic 回归方法进行分类并不需要做很多工作，所需做的只是把测试集上每个特征向量乘以优化方法得来的回归系数，
       再将该乘积结果求和，后输入到Sigmoid函数中即可。
    """

    @staticmethod
    def classifyVector(inX, weights):
        prob = GradAscentSimple().sigmoid(sum(inX * weights))
        if prob > 0.5:
            return 1.0
        else:
            return 0.0

    def colicTest(self):
        frTrain = open('./Data/horseColicTraining.txt')
        frTest = open('./Data/horseColicTest.txt')
        trainingSet = []
        trainingLabels = []
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(len(currLine) - 1):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[-1]))

        # 使用改进的随机上升梯度训练
        # trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
        # 使用上升梯度训练
        trainWeights = GradAscentSimple().gradAscent(np.array(trainingSet), trainingLabels)[0]

        errorCount = 0
        numTestVect = 0.0
        for line in frTest.readlines():
            numTestVect += 1.0
            currLine = line.strip().split('\t')
            lineArr = []                             # 作为测试输入
            for i in range(len(currLine) - 1):
                lineArr.append(float(currLine[i]))

            # 预测
            predict_class = self.classifyVector(np.array(lineArr), trainWeights[:, 0])
            if int(predict_class) != int(currLine[-1]):
                errorCount += 1

        # 错误概率计算
        errorRate = (float(errorCount) / numTestVect) * 100
        print("测试集错误率为：%.2f%%" % errorRate)


TestLogistic().colicTest()


# ----------------知识点-----------------

# numpy.append(arr, values, axis=None):就是arr和values会重新组合成一个新的数组，做为返回值。
# 当axis无定义时，是横向加成，返回总是为一维数组
# 当axis为0时，数组是加在下面（列数要相同）
