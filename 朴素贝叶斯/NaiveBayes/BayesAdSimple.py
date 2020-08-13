# -*- coding: UTF-8 -*-
"""
@author:liux
@time:2020/07/31
@func:示例：新闻分类器
      训练算法：使用我们之前建立的trainNB0()函数。
"""

from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os
import random
import jieba
from collections import defaultdict


class AdSimple:

    @staticmethod
    def TextProcessing(folder_path):
        """
        中文文本处理,将文件夹内部的所有txt文档分词并存储在data_list中，将txt上一级文件夹名称存储在class_list中
        :param folder_path: 文本存放的路径
        :return:   all_words_list - 按词频降序排序的训练集列表
                    train_data_list - 训练集列表
                    train_class_list - 训练集标签列表
        """
        folder_list = os.listdir(folder_path)

        data_list = []
        class_list = []

        # 遍历每个子文件夹
        for folder in folder_list:
            new_folder_path = os.path.join(folder_path, folder)
            files = os.listdir(new_folder_path)
            j = 1
            for file in files:
                # 每类txt样本数最多100
                if j > 100:
                    break

                with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:
                    raw = f.read()

                # jieba.cut方法接受两个输入参数：1）第一个参数为需要分词的字符串 2）cut_all参数用来控制是否采用
                word_cut = jieba.cut(raw, cut_all=False)
                word_list = list(word_cut)
                data_list.append(word_list)
                class_list.append(folder)
                j += 1

        # zip压缩合并，将数据与标签对应压缩
        data_class_list = list(zip(data_list, class_list))
        random.shuffle(data_class_list)          # 将data_class_list乱序，shuffle()方法将序列或元组所有元素随机排序

        # 训练集和测试集切分的索引值
        train_data_list, train_class_list = zip(*data_class_list)

        # 统计训练集词频
        all_words_dict = {}
        for word_list in train_data_list:
            all_words_dict = defaultdict(int)  # 字典所有的值会被初始化为0
            for x in word_list:
                all_words_dict[x] += 1

        all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
        all_words_list, all_words_nums = zip(*all_words_tuple_list)
        all_words_list = list(all_words_list)
        return all_words_list, train_data_list, train_class_list

    @staticmethod
    def MakeWordsSet(words_file):
        """
        读取文件里的内容，并去重
        :param words_file: 文件路径
        :return:
        """
        words_set = set()
        with open(words_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                word = line.strip()                                       # 去掉每行两边的空字符
                if len(word) > 0:
                    words_set.add(word)
        return words_set

    @staticmethod
    def words_dict(all_words_list, deleteN, stopwords_set=set()):
        """
        文本特征选取
        :param all_words_list: 训练集所有文本列表
        :param deleteN: 删除词频最高的deleteN个词
        :param stopwords_set: 指定的结束语
        :return:
        """
        feature_words = []
        n = 1
        for t in range(deleteN, len(all_words_list), 1):
            # feature_words的维度为1000
            if n > 1000:
                break
            # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
            if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                    all_words_list[t]) < 5:
                feature_words.append(all_words_list[t])
            n += 1
        return feature_words

    @staticmethod
    def TextFeatures(data_list, feature_words):
        """
        根据feature_words将文本向量化
        :param train_data_list: 训练集
        :param test_data_list: 测试集
        :param feature_words: 特征集
        :return:
        """
        # 出现在特征集中，则置1
        def text_features(text, feature_words):
            text_words = set(text)
            features = [1 if word in text_words else 0 for word in feature_words]
            return features

        feature_list = [text_features(text, feature_words) for text in data_list]
        return feature_list

    @staticmethod
    def TextClassifier(train_feature_list, train_class_list):
        """
        新闻分类器
        :param train_feature_list: 训练集向量化的特征文本
        :param test_feature_list: 测试集向量化的特征文本
        :param train_class_list: 训练集分类标签
        :param test_class_list: 测试集分类标签
        :return: 分类器精度
        """
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        # test_accuracy = classifier.score(test_feature_list, test_class_list)
        return classifier

    def txt_pre_process(self, file_path):
        data_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read()

        # jieba.cut方法接受两个输入参数：1）第一个参数为需要分词的字符串 2）cut_all参数用来控制是否采用
        word_cut = jieba.cut(raw, cut_all=False)
        word_list = list(word_cut)
        data_list.append(word_list)
        return data_list

    def main(self):
        # 文本预处理

        # 训练集存放地址
        folder_path = './Data/SogouC/Sample'
        all_words_list, train_data_list, train_class_list = self.TextProcessing(folder_path)

        # 生成stopwords_set
        stopwords_file = './Data/SogouC/stopwords_cn.txt'
        stopwords_set = self.MakeWordsSet(stopwords_file)

        feature_words = self.words_dict(all_words_list, 10, stopwords_set)
        train_feature_list = self.TextFeatures(train_data_list, feature_words)
        classifier = self.TextClassifier(train_feature_list, train_class_list)

        test_data_list = self.txt_pre_process('./Data/SogouC/Test/20.txt')
        test_feature_list = self.TextFeatures(test_data_list, feature_words)
        print(classifier.predict([test_feature_list[0]]))


AdSimple().main()


# ---------------------知识点--------------------
# os.listdir(path)方法用于返回指定的文件夹包含的文件或文件夹的名字列表。
# 这个列表以字母顺序。它不包括'.'和'..'即使它在文件夹中。

# jieba分词有三种模式：全模式、精确模式、搜索引擎模式。全模式（true）和精确模式（false）通过jieba.cut实现，
# 搜索引擎模式对应cut_for_search

# zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
# 如果各个迭代器的元素个数不一致，则返回列表长度与最短对象相同，利用*号操作符，可以将元组解压为列表
# python3中zip()返回一个对象，如需展示列表，需手动list()转换

# 集合add方法：把要传入的元素作为一个整体添加到集合中#,如add('python')即为‘python’
# 集合update方法：要把传入元素拆分，作为个体传入到集合中,如update('python')即为'p''y''t''h''o''n'
