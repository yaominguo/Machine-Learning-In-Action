#encoding:utf-8
from numpy import *
import operator


def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

# k-近邻算法
# inX为用于分类的输入向量，dataSet为输入的训练样本集，labels为标签向量，k表示用于选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
# 距离计算
    dataSetSize=dataSet.shape[0]    ##return 4
    ##numpy.tile([0,0],(1,2))即将[0,0]二维上复制2次，一维上复制1次，为[[0,0,0,0]]。numpy.tile([0,0],(2))一个参数默认返回一维复制2次，为[0,0,0,0]。numpy.title([3,5],(2,1))为[[3,5],[3,5]]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    ##axis=1为数组内自己相加，axis=0为跨数组相加。例如[[0, 2, 1], [3, 5, 6], [0, 1, 1]],sum()返回19，sum(axis=0)返回[3 8 8]，sum(axis=1)返回[3,14,2]。
    ##另外，一维数组只有0轴，没有1轴
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5  ##开根号
    sortedDistIndicies=distances.argsort() ##argsort函数返回的是数组值从小到大的索引值
# 选择距离最小的k个点
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        ##dict.get(key, default=None)。key -- 字典中要查找的键。default -- 如果指定键的值不存在时，返回该默认值值
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
 # 排序
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)##reverse=False为升序排序；reverse=True为降序排序
    return sortedClassCount


###运行>>>kNN.classify0([0,0],group,labels,3)   结果为 'B'。  ###