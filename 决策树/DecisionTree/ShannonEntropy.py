# -*- coding: utf-8 -*-
"""
作者：liux
日期：2020/6/9 18:42
功能：香农熵相关
知识点： 信息熵:信息熵便是信息的期望值。
        条件熵:X给定条件下Y的条件分布的熵对X的数学期望，在机器学习中为选定某个特征后的熵。
        信息增益:信息增益代表了在一个条件下，信息复杂度（不确定性）减少的程度，信息增益越大，则这个特征的选择性越好；
                 在概率中定义为：信息熵与条件熵相减。
        信息增益比：特征的信息增益熵与该特征的信息熵的比值。
        ID3算法使用的是信息熵增益；C4.5算法使用的是信息熵增益比率。
        A发生的概率很大，事件B发生的概率很小,则事件B的信息量比事件A的信息量要大。
        所以当越不可能的事件发生了，我们获取到的信息量就越大。
        越可能发生的事件发生了，我们获取到的信息量就越小。

"""
from math import log
from collections import defaultdict
import operator

from ShowData import DrawTree


class ShannonEnt:

    @staticmethod
    def createDataSet():
        """
        年龄：0代表青年，1代表中年，2代表老年
        有工作：0代表否，1代表是
        有自己的房子：0代表否，1代表是
        信贷情况：0代表一般，1代表好，2代表非常好
        类别（是否给贷款）：no代表否，yes代表是
        """

        dataSet = [[0, 0, 0, 0, 'no'],
                   [0, 0, 0, 1, 'no'],
                   [0, 1, 0, 1, 'yes'],
                   [0, 1, 1, 0, 'yes'],
                   [0, 0, 0, 0, 'no'],
                   [1, 0, 0, 0, 'no'],
                   [1, 0, 0, 1, 'no'],
                   [1, 1, 1, 1, 'yes'],
                   [1, 0, 1, 2, 'yes'],
                   [1, 0, 1, 2, 'yes'],
                   [2, 0, 1, 2, 'yes'],
                   [2, 0, 1, 1, 'yes'],
                   [2, 1, 0, 1, 'yes'],
                   [2, 1, 0, 2, 'yes'],
                   [2, 0, 0, 0, 'no']]
        # 分类属性
        labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
        return dataSet, labels

    @staticmethod
    def calcShannonEnt(dataSet):
        numEntires = len(dataSet)
        # 保存每个标签（Label）出现次数的“字典”
        # labelCounts = {}
        labelCounts = defaultdict(int)  # 字典所有的值会被初始化为0
        for featVec in dataSet:
            # 提取标签（Label）信息
            currentLabel = featVec[-1]
            # # 如果标签（Label）没有放入统计次数的字典，添加进去
            # if currentLabel not in labelCounts.keys():
            #     # 创建一个新的键值对，键为currentLabel值为0
            #     labelCounts[currentLabel] = 0
            # Label计数
            labelCounts[currentLabel] += 1

        # 计算香农熵
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntires           # 选择该标签（Label）的概率
            shannonEnt -= prob*log(prob, 2)                       # 利用公式计算,log指数

        return shannonEnt


# dataSet, labels = ShannonEnt().createDataSet()
# print(ShannonEnt().calcShannonEnt(dataSet))

class DecisionEnt:

    @staticmethod
    def splitDataSet(dataSet, axis, value):
        """
        按照给定特征划分数据集
        :param dataSet: 待划分的数据集
        :param axis: 划分数据集的特征  如：0
        :param value: 特征的返回值     如：0    结果是第一个字符为0的各个子列表
        :return:
        """
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                # 去掉axis特征
                reducedFeatVec = featVec[:axis]
                # 将符合条件的添加到返回的数据集,extend() 函数用于在列表末尾一次性追加另一个序列中的多个值。
                reducedFeatVec.extend(featVec[axis + 1:])
                # 列表中嵌套列表
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def chooseBestFeatureToSplit(self, dataSet):
        """
        选择最优特征， Gain(D,g) = Ent(D) - SUM(|Dv|/|D|)*Ent(Dv)
        :param dataSet:  数据集
        :return:  信息增益最大的（最优）特征的索引值
        """
        numFeatures = len(dataSet[0]) - 1                     # 特征数量
        baseEntropy = ShannonEnt().calcShannonEnt(dataSet)    # 计算数据集的香农熵
        bestInfoGain = 0.0                                    # 信息增益
        bestFeature = -1                                      # 最优特征的索引值
        # 遍历所有特征
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0                                                     # 经验条件熵
            # 计算信息增益
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)                # 划分子集 i-->axis
                prob = len(subDataSet) / float(len(dataSet))                     # 计算子集的概率
                newEntropy += prob * ShannonEnt().calcShannonEnt(subDataSet)     # 根据公式计算经验条件熵
            # 信息增益，信息熵与条件熵之差
            infoGain = baseEntropy - newEntropy
            # print("第%d个特征的增益为%.3f" % (i, infoGain))

            # 计算信息增益
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain                                # 更新信息增益，找到最大的信息增益
                bestFeature = i                                        # 记录信息增益最大的特征的索引值
        return bestFeature

    @staticmethod
    def majorityCnt(classList):
        """
        统计classList中出现次数最多的元素（类标签）,服务于递归第2个终止条件
        :param classList: 类标签列表
        :return: 出现次数最多的元素（类标签）
        """
        classCount = defaultdict(int)
        # 统计classList中每个元素出现的次数
        for vote in classList:
            # if vote not in classCount.keys():
            #     classCount[vote] = 0
            classCount[vote] += 1
        # operator.itemgetter(1)获取对象的第1列的值
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, dataSet, labels):
        """
        创建决策树（ID3算法）
        递归有两个终止条件：1、所有的类标签完全相同，直接返回类标签
                            2、用完所有标签但是得不到唯一类别的分组，即特征不够用，挑选出现数量最多的类别作为返回
        :param dataSet: 训练数据集
        :param labels:  分类属性标签
        :return:
        """
        classList = [example[-1] for example in dataSet]                # 取分类标签（是否放贷：yes or no）
        if classList.count(classList[0]) == len(classList):             # 如果类别完全相同则停止继续划分
            return classList[0]
        # 遍历完所有特征时返回出现次数最多的类标签
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)

        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]

        myTree = {bestFeatLabel: {}}                                   # 根据最优特征的标签生成树
        # del (labels[bestFeat])                                         # 删除已经使用的特征标签
        featValues = [example[bestFeat] for example in dataSet]        # 得到训练集中所有最优解特征的属性值
        uniqueVals = set(featValues)                                   # 去掉重复的属性值
        # 遍历特征，创建决策树
        for value in uniqueVals:
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), labels)
        return myTree


# dataSet, labels = ShannonEnt().createDataSet()
# print(DecisionEnt().createTree(dataSet, labels))


class UseDecisionTree:
    """
    依靠训练数据构造了决策树之后，我们可以将它用于实际数据的分类。
    在执行数据分类时， 需要决策树以及用于构造树的标签向量。
    """

    def classify(self, inputTree, featLabels, testVec):
        """
        使用决策树分类,
        递归遍历整棵树，比较testVec变量中的值与树节点的值，如果到达叶子节点，则返回当前节点的分类标签
        :param inputTree: 已经生成的决策树
         :param featLabels: 存储选择的最优特征标签
        :param testVec: 测试数据列表，顺序对应最优特征标签
        :return:
        """

        # 获取决策树结点
        firstStr = next(iter(inputTree))
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(str(firstStr))

        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
        return classLabel


# dataSet, labels = ShannonEnt().createDataSet()
# myTree = DecisionEnt().createTree(dataSet, labels)
# featLabels = ['年龄', '有工作', '有自己的房子', '信贷情况']
# print(UseDecisionTree().classify(myTree, featLabels， [1, 1, 0, 0]))

