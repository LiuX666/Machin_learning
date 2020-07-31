# -*- coding: UTF-8 -*-
"""
@author:liux
@time:2020/07/23
@func: 创建数据样本、简单示例,过滤网站的恶意留言
"""
import numpy as np


class SimpleBayes:

    @staticmethod
    def create_dataset():

        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                       ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                       ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                       ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                       ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        # 类别标签向量，1代表侮辱性词汇，0代表不是
        classVec = [0, 1, 0, 1, 0, 1]

        return postingList, classVec

    @staticmethod
    def setOfWords2Vec(vocabList, inputSet):
        returnVec = [0] * len(vocabList)

        for word in inputSet:
            if word in vocabList:
                # 如果词条存在于词汇表中，则置1, index返回word出现在vocabList中的索引
                # 若这里改为+=则就是基于词袋的模型，遇到一个单词会增加单词向量中的对应值
                returnVec[vocabList.index(word)] = 1
            else:
                print("词汇表无此词条：%s" % word)

        return returnVec

    @staticmethod
    def createVocabList(dataSet):
        vocabSet = set([])
        for document in dataSet:
            # 取并集
            vocabSet = vocabSet | set(document)
        return list(vocabSet)


class TrainDoc:

    @staticmethod
    def trainNB0(trainMatrix, trainCategory):
        """
        朴素贝叶斯分类器训练函数
        :param trainMatrix: 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
        :param trainCategory: 训练类标签向量，即create_dataset返回的classVec
        :return:
        """
        # 计算训练文档数目
        numTrainDocs = len(trainMatrix)
        # 计算每篇文档的词条数目
        numWords = len(trainMatrix[0])
        # 文档属于侮辱类的概率
        pAbusive = sum(trainCategory) / float(numTrainDocs)

        # 创建numpy.zeros数组，词条出现数初始化为0; 创建numpy.ones数组，词条出现数初始化为1,拉普拉斯平滑
        p0Num = np.ones(numWords)
        p1Num = np.ones(numWords)
        # 分母初始化为2，拉普拉斯平滑
        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(numTrainDocs):
            # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)...
            if trainCategory[i] == 1:
                p1Num += trainMatrix[i]                     # 统计所有侮辱类文档中每个单词出现的个数
                p1Denom += sum(trainMatrix[i])              # 统计一共出现的侮辱单词的个数
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])

        # 每类单词分别出现的概率，p1Vect = p1Num / p1Denom，取对数，
        # 通过求对数可以避免下溢出或者浮点数舍入导致的错误(相乘许多很小的数，后四舍五入后会得到0)
        p1Vect = np.log(p1Num / p1Denom)
        p0Vect = np.log(p0Num / p0Denom)

        # 返回每类的条件概率数组、文档属于侮辱类的概率
        return p0Vect, p1Vect, pAbusive

    @staticmethod
    def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
        """
        朴素贝叶斯分类器分类函数
        :param vec2Classify: 待分类的词条数组
        :param p0Vec: 侮辱类的条件概率数组
        :param p1Vec: 非侮辱类的条件概率数组
        :param pClass1: 文档属于侮辱类的概率
        :return:
        """
        # 对应元素相乘，logA*B = logA + logB所以这里是累加
        p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
        p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
        if p1 > p0:
            return 1
        else:
            return 0


# postingList, classVec = SimpleBayes().create_dataset()
# myVocabList = SimpleBayes().createVocabList(postingList)
#
# trainMat = []
# for itemPostingDoc in postingList:
#       trainMat.append(SimpleBayes().setOfWords2Vec(myVocabList, itemPostingDoc))
#
# p0Vect, p1Vect, pAbusive = TrainDoc().trainNB0(trainMat, classVec)
# print(p0Vect, '\n', p1Vect,  '\n', pAbusive)

# 测试
# testEntry = ['stupid', 'garbage']
# thisDoc = SimpleBayes().setOfWords2Vec(myVocabList, testEntry)
# print(TrainDoc().classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))
