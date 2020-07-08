# -*- coding: utf-8 -*-
"""
作者：liux
日期：2020/6/16 19:48
功能：基于Matplotlib绘制决策树图形
"""

class DrawTree:

    def getNumLeafs(self, myTree):
        """
        获取决策树叶子结点的数目，确定x轴的长度
        :param myTree: 决策树
        :return:
        """
        numLeafs = 0                                   # 初始化叶子
        firstStr = next(iter(myTree))                  # next() 返回迭代器的下一个项目 next(iterator[, default])
        secondDict = myTree[firstStr]                  # 获取下一组字典
        for key in secondDict.keys():
            # 判断该结点是否为字典，如果不是字典，代表此节点为叶子结点
            if type(secondDict[key]).__name__ == 'dict':
                numLeafs += self.getNumLeafs(secondDict[key])
            else:
                numLeafs += 1
        return numLeafs

    def getTreeDepth(self, myTree):
        """
        获取决策树的层数，确定y轴的高度
        """
        maxDepth = 0                                  # 初始化决策树深度
        firstStr = next(iter(myTree))
        secondDict = myTree[firstStr]

        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict':
                thisDepth = 1 + self.getTreeDepth(secondDict[key])
            else:
                thisDepth = 1

            # 更新最深层数
            if thisDepth > maxDepth:
                maxDepth = thisDepth

        return maxDepth



