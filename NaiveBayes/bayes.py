# coding=utf-8

from numpy import *

##创建实验样本
def loadDataSet():
    postingList=[['my','dog','has','flea','problem','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]  # 1代表侮辱性，0代表正常言论
    return postingList,classVec

##创建不重复词列表
def createVocabList(dataSet):
    vocabSet=set([]) #创建空集合，set数据类型会返回一个不重复的词表
    for document in dataSet:
        vocabSet=vocabSet|set(document) #将每篇文档返回的新词集合添加到该集合中，‘|’为求两个集合的并集
    return list(vocabSet)

##将词组列表转换为文档向量列表
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList) #创建一个和词汇表等长的向量，元素都设为0
    for word in inputSet: #遍历文档所有单词
        if word in vocabList: #如果出现了词汇表中的单词
            returnVec[vocabList.index(word)]=1 #则将输出的文档向量中的对应值设为1
        else: print "The word: %s is not in my Vocabulary!" % word
    return returnVec

##朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=zeros(numWords);p1Num=zeros(numWords)
    p0Denom=0.0;p1Denom=0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=p1Num/p1Denom
    p0Vect=p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive