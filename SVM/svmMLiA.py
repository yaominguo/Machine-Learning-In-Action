# coding=utf-8
from numpy import *
###SMO表示序列最小优化（Sequential Minimal Optimization）

#SMO算法中的辅助函数
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])]) #得到整个数据矩阵
        labelMat.append(float(lineArr[2])) #得到每行的列标签
    return dataMat, labelMat
def selectJrand(i,m): #i是第一个alpha的下标，m是所有alpha 的数目
    import random
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

##简化版SMO算法
def smoSimple(dataMatIn,classLabels, C, toler, maxIter): #(数据集，类别标签，常数C，容错率，最大循环次数)
    dataMatrix=mat(dataMatIn);labelMat=mat(classLabels).transpose()
    b=0; m,n=shape(dataMatrix);
    alphas=mat(zeros((m,1)))
    iter=0
    while (iter<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei=fXi-float(labelMat[i])
            if((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)): #如果alpha可以更改进入优化过程
                j=selectJrand(i,m) #随机选择第二个alpha
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy();
                alphaJold=alphas[j].copy();
                if (labelMat[i]!=labelMat[j]): #保证alpha在0和C质检
                    L=max(0,alphas[j] - alphas[i])
                    H=min(C, C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C, alphas[j]+alphas[i])
                if L==H:
                    print "L==H";
                    continue
                eta =2.0 * dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >=0:
                    print "eta>=0";
                    continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001):
                    print "j not moving enough";
                    continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j]) #对i进行修改，修改量与j相同但放心相反
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]): #设置常数项
                    b=b1
                elif (0<alphas[j]) and (C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print "iter: %d i:%d, pairs changed %d" %(iter,i,alphaPairsChanged)
        if(alphaPairsChanged==0):
            iter+=1
        else:
            iter=0
        print "iteration number: %d" % iter
    return b,alphas
