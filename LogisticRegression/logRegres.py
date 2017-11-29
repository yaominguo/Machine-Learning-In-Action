# coding=utf-8
from numpy import *

###Logistic回归的目的是寻找一个非线性函数sigmoid的最佳拟合参数，求解过程可以由最优化算法来来完成。
###最优化算法中，最常用的就是梯度上升算法，而梯度上升算法又可以简化为随机梯度上升算法。
###随机梯度上升算法与梯度上升算法的效果相当，但占用更少的计算资源。
###此外，随机梯度上升算法是在线算法，可以再新数据到来时就完成参数更新，而不需要重新读取整个数据集来进行批处理运算。



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
def stocGradAscent1(dataMatrix,classLabels, numIter=150):
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

##Logistic回归分类函数
def classifyVector(inX,weights): #以回归系数和特征向量作为输入来计算对应的sigmoid值
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0
def colicTest(): #打开测试集和训练集，并对数据进行格式化处理
    frTrain=open('src/HorseColicTraining.txt')
    frTest=open('src/HorseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate
def multiTest(): #调用colicTest 10次并求结果的平均值
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print "after %d iterations the  average error rate is: %f" % (numTests,errorSum/float(numTests))


