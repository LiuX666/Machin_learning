# -*- coding: utf-8 -*-
"""
作者：liux
日期：2020/7/2 18:35
功能： 示例：使用决策树预测隐形眼镜类型, 类别：no lenses，soft,hard
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pydotplus
from six import StringIO
from sklearn import tree


import ShannonEntropy


class SimpleDecisionTree:

    @staticmethod
    def lenses_tree():
        try:
            with open("./data/lenses.txt") as fr:
                # 处理文件，去掉每行两头的空白符，以'|'分隔每个数据
                lenses = [inst.strip().split('|') for inst in fr.readlines()]

                # 特征标签,年龄、症状、是否散光、眼泪数量
                lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
                lensesTree = ShannonEntropy.DecisionEnt().createTree(lenses, lensesLabels)
                return lensesTree
        except Exception as e:
            print(e)
            raise

    @staticmethod
    def sklearn_tree():
        with open("./data/lenses.txt") as fr:
            lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lenses_targt = []                                                        # 提取每组数据的类别，保存在列表里
        for each in lenses:
            lenses_targt.append([each[-1]])                                      # 存储Label到lenses_targt中
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

        lenses_list = []
        lenses_dict = {}
        for each_label in lensesLabels:
            for each in lenses:
                lenses_list.append(each[lensesLabels.index(each_label)])
            lenses_dict[each_label] = lenses_list
            lenses_list = []

        lenses_pd = pd.DataFrame(lenses_dict)
        # 打印数据
        # print(lenses_pd)
        # 创建LabelEncoder对象
        le = LabelEncoder()
        # 为每一列序列化
        for col in lenses_pd.columns:
            # fit_transform()干了两件事：fit找到数据转换规则，并将数据标准化
            # transform()直接把转换规则拿来用,需要先进行fit
            # transform函数是一定可以替换为fit_transform函数的，fit_transform函数不能替换为transform函数
            lenses_pd[col] = le.fit_transform(lenses_pd[col])
        # 打印归一化的结果
        # print(lenses_pd)
        # 创建DecisionTreeClassifier()类
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
        # 使用数据构造决策树
        # fit(X,y):Build a decision tree classifier from the training set(X,y)
        # 所有的sklearn的API必须先fit
        clf = clf.fit(lenses_pd.values.tolist(), lenses_targt)
        dot_data = StringIO()
        # 绘制决策树
        tree.export_graphviz(clf, out_file=dot_data, feature_names=lenses_pd.keys(),
                             class_names=clf.classes_, filled=True, rounded=True,
                             special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # 保存绘制好的决策树，以PDF的形式存储。
        graph.write_pdf("tree.pdf")


# lensesTree = SimpleDecisionTree().lenses_tree()
# Labels = ['age', 'prescript', 'astigmatic', 'tearRate']
# print(ShannonEntropy.UseDecisionTree().classify(lensesTree, Labels, ['presbyopic', 'myope', 'yes', 'normal']))

# print(clf.predict([[1,1,1,0]]))


