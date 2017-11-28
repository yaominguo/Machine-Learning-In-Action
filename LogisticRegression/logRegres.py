# coding=utf-8
from numpy import *
##Logistic 回归梯度上升优化算法
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('src/testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()  #每行前两个值分别是轴X1`X2
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #同时将X0设为1.0
        labelMat.append(int(lineArr[2]))  #第三个值是数据对应的类别标签
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels): #dataMatIn是二维numpy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。
    dataMatrix=mat(dataMatIn)  #转换为numpy矩阵数据类型
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001 #向目标移动的步长
    maxCycles=500 #最大迭代次数
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights) #h是一个列向量，列向量的元素个数等于样本个数，这里是100.
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights


##画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    # weights=wei.getA()
    weights=wei
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0, 3.0, 0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

##随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels, numIter=150):
    m,n=shape(dataMatrix) #跟梯度上升算法对比，这里没有矩阵的转换过程，所有变量的数据类型都是numpy数组
    weights=ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01 #alpha每次迭代的时候都会调整，随着迭代不断减少，但永远不会少于0
            randIndex=int(random.uniform(0,len(dataIndex))) #随机选取样本来更新回归系数，将可以减少周期性的波动。
            h=sigmoid(sum(dataMatrix[randIndex]*weights)) #跟梯度上升算法对比，这里h和error全是数值
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights