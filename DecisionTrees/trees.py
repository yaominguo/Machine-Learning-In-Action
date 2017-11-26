# coding=utf-8
from math import log
import operator

###这里的决策树算法称为ID3，它是一个号的算法但是并不完美，因为它无法直接处理数值型数据。
###尽管可以通过量化的方法将数值型数据转化为标称型数值，但是如果存在太多的特征划分，就会面临其他问题。



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

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.key():
            classCount[vote]=0
            classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

##创建树的函数代码
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet] #创建列表变量，包含了数据集的所有类标签
    if classList.count(classList[0])==len(classList): #所有类标签完全相同则直接返回该类标签
        return classList[0]
    if len(dataSet[0])==1: #遍历完所有特征仍不能将数据集划分成仅包含唯一类别的分组则返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet] #得到列表包含的所有属性值
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

##使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStr=inputTree.keys()[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr) #查找当前列表中第一个匹配firstStr变量的元素
    for key in secondDict.keys():
        if testVec[featIndex]==key: #比较testVec变量中的值与树节点的值
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else: classLabel=secondDict[key]
    return classLabel

##使用pickle模块存储决策树
def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)
