# coding=utf-8
from math import log

def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

##计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)  #数据集中实例的总数
    labelCounts={} #创建数据字典，它的键值是最后一列的数据。每个键值都记录了当前类别出现的次数
    for featVec in dataSet:
        currentLabel=featVec[-1]
        # if currentLabel not in labelCounts.keys(): #如果当前键值不存在，则扩展字典并将当前键值加入字典
        #     labelCounts[currentLabel]=0
        #     labelCounts[currentLabel]+=1
        labelCounts[currentLabel]=labelCounts.get(currentLabel,0)+1 #用字典get()取代上面三行代码，效果一样
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries #使用所有类标签的发生频率计算类别出现的概率
        shannonEnt-=prob*log(prob,2) #计算信息熵的期望值
    return shannonEnt

##按照给定特征划分数据集
def splitDataSet(dataSet,axis,value): #参数为（数据集，划分数据集的特征，特征的返回值）
    retDataSet=[] #创建新的list对象
    for featVec in dataSet:
        if featVec[axis]==value: #抽取符合特征的数据
            reducedFeatVec=featVec[:axis] #返回从头开始到axis结束的数组（不含axis位置）
            reducedFeatVec.extend(featVec[axis+1:]) #[1].extend([2,3])=>[1,2,3]。返回从axis+1位置开始到结尾的数组（含axis+1位置）
            retDataSet.append(reducedFeatVec) #[1].append([2,3])=>[1,[2,3]]
    return retDataSet

##选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet) #整个数据集的原始香农熵
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet] #创建唯一的分类标签列表
        uniqueVals=set(featList) #创建python的集合数据类型，与列表区别在于集合类型中的每个值互不相同
        newEntropy=0.0
        for value in uniqueVals: #遍历当前特征中的所有唯一属性值
            subDataSet=splitDataSet(dataSet,i,value) #对每个特征划分一次数据集
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet) #计算数据集的新熵值并对所有唯一特征值取得到的熵求和
        infoGain=baseEntropy-newEntropy #计算基尼指数，得出熵最低的特征
        if(infoGain>bestInfoGain): #比较所有特征中的信息增益
            bestInfoGain=infoGain
            bestFeature=i
    return  bestFeature
