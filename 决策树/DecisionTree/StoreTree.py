# -*- coding: utf-8 -*-
"""
作者：liux
日期：2020/7/2 10:18
功能：构造决策树是很耗时的任务，即使处理很小的数据集，如果数据集很大，将会耗费很多计算时间。
     然而用创建好的决策树解决分类问题，则可以很快完成。
     因此，为了节省计算时间，好能够在每次执行分类时调用已经构造好的决策树。
     为 了解决这个问题，需要使用Python模块pickle序列化对象，
     序列化对象可以在磁 盘上保存对象，并在需要的时候读取出来。
"""

import pickle

import ShannonEntropy


class SaveTree:

    @staticmethod
    def store_tree(inputTree, filename):
        """
         存储决策树
        :param inputTree: 已经生成的决策树
        :param filename: 决策树的存储文件
        :return:
        """
        with open(filename, 'wb') as fw:
            pickle.dump(inputTree, fw)

    @staticmethod
    def grabTree(filename):
        """
         读取决策树
        """
        fr = open(filename, 'rb')
        return pickle.load(fr)


dataSet, labels = ShannonEntropy.ShannonEnt().createDataSet()
myTree = ShannonEntropy.DecisionEnt().createTree(dataSet, labels)

# SaveTree().store_tree(myTree, "./data/classifier_tree.txt")
print(SaveTree().grabTree("./data/classifier_tree.txt"))

