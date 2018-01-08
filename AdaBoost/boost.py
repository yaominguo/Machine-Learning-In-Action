# coding=utf-8
from numpy import *

def loadSimpData():
    datMat=matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

##单层决策树生成函数

#通过阈值比较对数据进行分类，所有在阈值以便的数据会分到类别-1，另一边的数据分到类别+1
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1)) #首先将返回数组的全部元素设置为1
    if threshIneq == 'lt': #然后将所有不满足不等式要求的元素设置为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

#遍历stumpClassify函数所有的可能输入值，并找到数据集上最佳的单层决策树
def buildStump(dataArr, classLabels, D): # ‘最佳’是基于权重向量D来定义的
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0 #用于在特征的所有可能值上进行遍历
    bestStump = {} #用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m, 1)))
    minError = inf #设置成正无穷大，之后用于寻找可能的最小错误率
    for i in range(n): #遍历数据集的所有特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones(m, 1))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr #计算加权错误率
                print "split: dim %d, thresh %.2f, thresh inequal: %s, the weight error is %.3f" % (
                i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
        return bestStump, minError, bestClasEst
