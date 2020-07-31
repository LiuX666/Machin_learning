# -*- coding: UTF-8 -*-
"""
@author:liux
@time:2020/07/27
@func:  测试算法，过滤垃圾邮件
        (1) 收集数据：提供文本文件。
        (2) 准备数据：将文本文件解析成词条向量。
        (3) 分析数据：检查词条确保解析的正确性。
        (4) 训练算法：使用我们之前建立的trainNB0()函数。
        (5) 测试算法：使用classifyNB()，并且构建一个新的测试函数来计算文档集的错误率。
        (6) 使用算法：构建一个完整的程序对一组文档进行分类
"""

import re
import numpy as np
import random

from BayesSimple import SimpleBayes, TrainDoc


class FillterEmail:

    @staticmethod
    def textParse(bigString):
        """
        接受一个大字符串并将其解析为字符串列表
        :param bigString:
        :return:
        """
        # 用特殊符号作为切分标志进行字符串切分，即非字母、非数字,
        # \W* 0个或多个非字母数字或下划线字符（等价于[^a-zA-Z0-9_]）
        # 除了单个字母，例如大写I，其他单词变成小写，去掉少于两个字符的字符串
        listOfTockens = re.split(r'\W*', bigString)
        return [tok.lower() for tok in listOfTockens if len(tok) > 2]

    def spamTest(self):
        docList = []
        classList = []
        fullText = []
        # 遍历25个txt文件
        for i in range(1, 26):
            # --可提取
            # 读取每个垃圾邮件，并以字符串转换成字符串列表
            wordList = self.textParse(open('./Data/spam/%d.txt' % i, 'r').read())
            docList.append(wordList)
            fullText.append(wordList)
            classList.append(1)                                        # 标记垃圾邮件，1表示垃圾文件

            wordList = self.textParse(open('./Data/ham/%d.txt' % i, 'r').read())
            docList.append(wordList)
            fullText.append(wordList)
            classList.append(0)                                         # 标记非垃圾邮件，0表示非垃圾文件

        vocabList = SimpleBayes().createVocabList(docList)              # 创建词汇表，不重复
        trainingSet = list(range(50))                             # 创建存储训练集的索引值的列表和测试集的索引值的列表
        testSet = []
        # 从50个邮件中，随机挑选出40个作为训练集，10个作为测试集
        for i in range(10):
            randIndex = int(random.uniform(0, len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del (trainingSet[randIndex])                          # 在训练集列表中删除添加到测试集的索引值

        # 创建训练集矩阵和训练集类别标签向量
        trainMat = []
        trainClasses = []
        # 遍历训练集
        for docIndex in trainingSet:
            trainMat.append(SimpleBayes().setOfWords2Vec(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])

        # 训练朴素贝叶斯模型
        p0V, p1V, pSpam = TrainDoc().trainNB0(np.array(trainMat), np.array(trainClasses))

        errorCount = 0
        for docIndex in testSet:
            # 测试集的词集模型
            wordVector = SimpleBayes().setOfWords2Vec(vocabList, docList[docIndex])
            if TrainDoc().classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
                errorCount += 1
                print("分类错误的测试集：", docList[docIndex])
        print("错误率：%.2f%%" % (float(errorCount) / len(testSet) * 100))


FillterEmail().spamTest()