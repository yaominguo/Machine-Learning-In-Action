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
    p0Num=ones(numWords);p1Num=ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    for i in range(numTrainDocs):  #遍历trainMatrix,一旦（侮辱或正常）词语在某一文档中出现
        if trainCategory[i]==1:    #则改词对应的个数（p1Num or p0Num）就加1，并且该文档的总次数也加1
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)   #对每个元素除以该类别中的总词数
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

##朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1) #用Numpy的数组来计算两个向量相乘的结果，就是将两个向量中的第1个元素相乘，然后将第2个元素相乘，以此类推。
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1) # 接下来将词汇表中所有词的对应值相加，再加到类别的对数概率上
    if p1>p0:   #最后返回大概率对应的类别标签
        return 1
    else:
        return 0


def testingNB():
    listOposts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOposts)
    trainMat=[]
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb)