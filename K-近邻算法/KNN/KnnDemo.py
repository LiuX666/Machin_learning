# -*- coding: utf-8 -*-
"""
作者：liux
日期：2020/5/26 15:45
功能：k近邻算法demo，示例为喜欢人员分类
"""
import numpy as np
import operator

import ShowData


class KnnPractice:

    @staticmethod
    def createDataSet():
        """
        创建数据集
        :return:
        """
        # 四组二维特征
        group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
        # 四组特征的标签
        labels = ['爱情片', '爱情片', '动作片', '动作片']
        return group, labels

    @staticmethod
    def classify0(inX, dataSet, labels, k):
        """
        分类器，该函数的功能是使用k-近邻算法将 每组数据划分到某个类中。
           1）距离计算，计算已知类别数据集中的点与当前点之间的距离
           2）按照距离递增次序排序，argsort
           3）选取与当前点距离最小的k个点
           4) 确定前k个点所在类别的出现频率
           5) 返回前k个点出现频率最高的类别作为当前点的预测分类。
        :param inX: 用于分类的数据（测试集），输入向量
        :param dataSet: 用于训练的数据（训练集）（n*1维列向量）
        :param labels: 分类标准（n*1维列向量），标签向量
        :param k: kNN算法参数，选择距离最小的k个点
        :return:
        """
        # numpy函数shape[0]返回dataSet的行数
        dataSetSize = dataSet.shape[0]
        # 将inX重复dataSetSize次并排成一列，并相减
        diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
        # 二维特征相减后平方（用diffMat的转置乘diffMat）
        sqDiffMat = diffMat ** 2
        # 0 列相加，1 行相加
        sqDistances = sqDiffMat.sum(axis=1)
        # 开方，计算出距离
        distances = sqDistances ** 0.5
        # argsort函数返回的是distances值从小到大的--索引值
        sortedDistIndicies = distances.argsort()

        classCount = {}
        # 选择距离最小的k个点
        for i in range(k):
            # 取出前k个元素的类别
            voteIlabel = labels[sortedDistIndicies[i]]
            # get()方法，返回指定键的值，如果值不在字典中返回0，计算类别次数、频率
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

        # key = operator.itemgetter(0)根据字典的键进行排序，key = operator.itemgetter(1)根据字典的值进行排序
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        # 返回次数最多的类别，即所要分类的类别
        return sortedClassCount[0][0]

# group, labels = KnnPractice().createDataSet()
# test_class = KnnPractice().classify0([100, 100], group, labels, 3)
# print(test_class)


# 示例
class KnnSimple:
    """
       分类：交往过三种类型的人： 不喜欢的人，魅力一般的人，极具魅力的人
       特征： 人员样本主要包含以下3种特征：每年获得的飞行常客里程数，玩视频游戏所耗时间百分比，每周消费的冰淇淋公升数
    """

    @staticmethod
    def file2matrix(filename):
        """
        打开解析文件，对数据进行分类
        :param filename:文件路径
        :return:
        """
        fr = open(filename)
        arrayOlines = fr.readlines()
        numberOfLines = len(arrayOlines)

        # 创建以零填充的矩阵NumPy矩阵，numberOfLines行3列
        returnMat = np.zeros((numberOfLines, 3))

        # 创建分类标签向量
        classLabelVector = []
        # 行的索引值
        index = 0

        for line in arrayOlines:
            # 去掉每一行首尾的空白符，例如'\n','\r','\t',' '
            line = line.strip()
            # 将每一行内容根据'\t'符进行切片,本例中一共有4列
            listFromLine = line.split('\t')
            # 将数据的前3列进行提取保存在returnMat矩阵中，也就是特征矩阵
            returnMat[index, :] = listFromLine[0:3]
            # 根据文本内容进行分类1：不喜欢；2：一般；3：喜欢
            if listFromLine[-1] == 'didntLike':
                classLabelVector.append(1)
            elif listFromLine[-1] == 'smallDoses':
                classLabelVector.append(2)
            elif listFromLine[-1] == 'largeDoses':
                classLabelVector.append(3)
            index += 1
        # 返回标签列向量以及特征矩阵
        return returnMat, classLabelVector
    # returnMat, classLabelVector = KnnSimple().file2matrix('./data/datingTestSet.txt')

    @staticmethod
    def autoNorm(dataSet):
        """
        对数据进行归一化,(x-min)/(max-min)
        :param dataSet:
        :return: normDataSet - 归一化后的特征矩阵,ranges - 数据范围,minVals - 数据最小值
        """
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals

        m = dataSet.shape[0]
        # 原始值减去最小值（x-xmin）,tile()函数将变量内容复制成输入矩阵同样大小的矩阵，
        diffDataSet = dataSet - np.tile(minVals, (m, 1))
        # 差值处以最大值和最小值的差值（x-xmin）/（xmax-xmin）
        normDataSet = diffDataSet / np.tile(ranges, (m, 1))

        return normDataSet, ranges, minVals

    def datingClassTest(self, filename):
        """
        分类器测试函数
        :param filename:
        :return:
        """
        # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
        datingDataMat, datingLabels = self.file2matrix(filename)
        # 数据归一化，返回归一化数据结果，数据范围，最小值
        normMat, ranges, minVals = self.autoNorm(datingDataMat)

        # 取所有数据的10% hoRatio越小，错误率越低
        hoRatio = 0.1
        # 获取normMat的行数
        m = normMat.shape[0]
        # 10%的测试数据的个数
        numTestVecs = int(m * hoRatio)
        # 分类错误计数
        errorCount = 0.0
        for i in range(numTestVecs):
            # 分类器，前numTestVecs个数据作为测试集，后m-numTestVecs个数据作为训练集,
            # k选择label数+1（结果比较好,或者循环找k, 4/7）
            # def find_k():
            #     k_dict = {}
            #     for i in range(1, 101):
            #         k_dict[i] = KnnSimple().datingClassTest('./data/datingTestSet.txt', i)
            #     k_dict = sorted(k_dict.items(), key=lambda d: d[1], reverse=False)
            #     print(k_dict)
            classifierResult = KnnPractice().classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                                       datingLabels[numTestVecs:m], 4)

            # print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
            if classifierResult != datingLabels[i]:
                errorCount += 1.0
        print(errorCount / float(numTestVecs) * 100)

    def classifyPerson(self, filename, inArr):
        """

        :param filename:
        :param inArr: ["每年获得的飞行常客里程数","玩视频游戏所消耗时间百分比","每周消费的冰淇淋公升数"]
        :return:
        """
        # 输出结果
        resultList = ['讨厌', '有些喜欢', '非常喜欢']

        datingDataMat, datingLabels = self.file2matrix(filename)
        normMat, ranges, minVals = self.autoNorm(datingDataMat)
        # 生成NumPy数组，测试集
        inArr = np.array(inArr)
        # 测试集归一化
        norminArr = (inArr - minVals) / ranges

        # 返回分类结果
        classifierResult = KnnPractice().classify0(norminArr, normMat, datingLabels, 4)
        # 打印结果
        print("你可能%s这个人" % (resultList[classifierResult - 1]))


# returnMat, classLabelVector = KnnSimple().file2matrix('./data/datingTestSet.txt')
# ShowData.showdata(returnMat, classLabelVector)
# KnnSimple().classifyPerson('./data/datingTestSet.txt', [1, 1, 0.2])




