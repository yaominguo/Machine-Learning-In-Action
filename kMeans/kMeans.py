# coding=utf-8
from numpy import *

##k均值聚类支持函数
def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=map(float,curLine)
        dataMat.append(fltLine)
    return dataMat
def distEclud(vecA,vecB): #计算两个向量的欧氏距离
    return sqrt(sum(power(vecA-vecB,2)))
def randCent(dataSet,k): #为给定数据集创建一个包含k个随机质心的集合
    n=shape(dataSet)[1]
    centroids=mat(zeros((k,n)))
    for j in range(n): #确保随机点在数据的边界之内
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)
    return centroids

## K均值聚类算法
def kMeans(dataSet,k,distMeas=distEclud, createCent=randCent):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf;minIndex=-1
            for j in range(k): #遍历所有质心并计算点到每个质心的距离来找到距离每个点最近的质心
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI;minIndex=j
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        print centroids
        for cent in range(k): #遍历所有的质心并更新他们的取值
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment

##二分K-均值聚类算法
def biKmeans(dataSet,k,distMeas=distEclud):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroid0=mean(dataSet,axis=0).tolist()[0] #创建初始簇
    centList=[centroid0]
    for j in range(m):
        clusterAssment[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2
    while (len(centList)<k):
        lowestSSE=inf
        for i in range(len(centList)): #尝试划分每一簇
            ptsInCurrCluster=dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat,splitClustAss=kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit=sum(splitClustAss[:,1])
            sseNotSplit=sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit, sseNotSplit
            if(sseSplit+sseNotSplit)<lowestSSE:
                bestCentTosplit=i
                bestNewCents=centroidMat
                bestClustAss=splitClustAss.copy()
                lowestSSE=sseSplit+sseNotSplit
        #更新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentTosplit
        print "the bestCentToSplit is: ", bestCentTosplit
        print "the len of bestClustAss is: ",len(bestClustAss)
        centList[bestCentTosplit]=bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentTosplit)[0],:]=bestClustAss
    return centList,clusterAssment
