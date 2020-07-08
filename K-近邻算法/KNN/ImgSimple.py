# -*- coding: utf-8 -*-
"""
作者：liux
日期：2020/5/28 11:44
功能：示例：构造使用k-近邻分类器的手写识别系统。为了简单起见，这里构造的系统 只能识别数字0到9
       假设需要识别的数字已经使用图形处理软件，处理成具有相同的色彩和大小：宽高是32像素×32像素的黑白图像
"""
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as Knn
from KnnDemo import KnnPractice
import datetime


class ImgKnnSimple:

    @staticmethod
    def img2vector(filename):
        """
        将32*32的二进制图像转换为1*1024向量
        :param filename:
        :return:
        """
        # 创建1*1024零向量
        returnVect = np.zeros((1, 1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            # 每一行的前32个数据依次存储到returnVect中
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
        return returnVect

    def handwritingClassTest(self):
        # 测试集的Labels
        hwLabels = []

        trainingFilesList = listdir('./data/trainingDigits')
        m = len(trainingFilesList)
        trainingMat = np.zeros((m, 1024))
        # 从文件名中解析出训练集的类别
        for i in range(m):
            fileNameStr = trainingFilesList[i]
            # 获得分类的数字
            classNumber = int(fileNameStr.split('_')[0])
            # 将获得的类别添加到hwLabels中
            hwLabels.append(classNumber)
            # 将每一个文件的1x1024数据存储到trainingMat矩阵中
            trainingMat[i, :] = self.img2vector('./data/trainingDigits/%s' % (fileNameStr))

        # 返回testDigits目录下的文件列表，测试数据
        testFileList = listdir('./data/testDigits')
        errorCount = 0.0
        mTest = len(testFileList)
        for i in range(mTest):
            fileNameStr = testFileList[i]
            classNumber = int(fileNameStr.split('_')[0])
            vectorUnderTest = self.img2vector('./data/testDigits/%s' % (fileNameStr))
            classifierResult = KnnPractice().classify0(vectorUnderTest, trainingMat, hwLabels, 3)

            # print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
            if (classifierResult != classNumber):
                errorCount += 1.0
        print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))
        # return errorCount / mTest * 100

        # # 调用sklearn中的KNN分类器
        # # 构造kNN分类器
        # neigh = Knn(n_neighbors=3, algorithm='auto')
        # # 拟合模型，trainingMat为测试矩阵，hwLabels为对应标签
        # neigh.fit(trainingMat, hwLabels)
        # # 获得预测结果
        # classifierResult = neigh.predict(vectorUnderTest)


ImgKnnSimple().handwritingClassTest()

# def find_k():
#     k_dict = {}
#     for i in range(1, 10):
#         k_dict[i] = ImgKnnSimple().handwritingClassTest(i)
#     k_dict = sorted(k_dict.items(), key=lambda d: d[1], reverse=False)
#     print(k_dict)
# find_k()