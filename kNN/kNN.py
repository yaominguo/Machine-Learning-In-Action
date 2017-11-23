#encoding:utf-8
from numpy import *
from os import listdir
import operator


###k-近邻算法是分类数据最简单最有效的算法，使用该算法时必须有接近实际数据的训练样本数据。
###k-近邻算法必须保存全部数据集，如果训练数据集很大，必须使用大量的存储空间。
###此外，由于必须对数据集中的每个数据计算距离值，实际使用时可能非常耗时。
###k-近邻算法的另一个缺陷是它无法给出任何数据的基础结构信息，因此我们也无法得知平均实例样本和典型实例样本具有什么特征。




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
    return sortedClassCount[0][0]


###运行>>>kNN.classify0([0,0],group,labels,3)   结果为 'B'。  ###


def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines) ##得到文件行数
    returnMat=zeros((numberOfLines,3)) ##创建以零填充的矩阵(二维数组),为了简化将另一维固定值设为3
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()  ##截取掉所有的回撤字符
        listFromLine=line.split('\t') ##将整行数据分割成一个元素列表
        returnMat[index,:]=listFromLine[0:3]  ##选取前3个元素
        classLabelVector.append(int(listFromLine[-1])) ##将列表的最后一列添加进去
        index+=1
    return returnMat,classLabelVector

##归一化特征值
def autoNorm(dataSet):
    minVals=dataSet.min(0)  ##选取最小值，参数0可以使函数从列中选取值而不是当前行中选取
    maxVals=dataSet.max(0)  ##选取最大值，参数0可以使函数从列中选取值而不是当前行中选取
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
##测试错误率
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('src/datingTestSet.txt')##读取文件数据
    normMat,ranges,minVals=autoNorm(datingDataMat) ##转换为归一化特征值
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount+=1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs)) ##返回错误率为0.050000,约为5%


##预测喜欢与否的函数
def classifyPerson():
    resultList=['not at all', 'in small doses', 'in large doses']
    percentTats=float(raw_input("Percentage of time spent playing video games?"))
    ffMiles=float(raw_input("Frequent flier miles earned per year?"))
    iceCream=float(raw_input("Liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('src/datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probable like this person: ",resultList[classifierResult-1]

##将图像转换为向量
def img2vector(filename):
    returnVect=zeros((1,1024)) #创建1*1024的NumPy数组
    fr=open(filename)
    for i in range(32): #循环读取文件的前32行
        lineStr=fr.readline()
        for j in range(32): #每行的头32个字符值存储在数组中
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect   #返回该数组

##手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('src/trainingDigits') #获取目录内容
    m=len(trainingFileList) #获取文件数量存储在m中
    trainingMat=zeros((m,1024))  #穿件一个m行1024列的训练矩阵，该矩阵每行数据存储一个图像
    for i in range(m):  #从文件名解析分类数字
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0] #截取.txt之前的文件名
        classNumStr=int(fileStr.split('_')[0]) #截取 _ 之前的数字
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('src/trainingDigits/%s' %fileNameStr)

    testFileList=listdir('src/testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('src/testDigits/%s' %fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3) #使用classify0函数测试该目录下的每个文件
        print "The classifier came back with: %d, the real answer is: %d" %(classifierResult,classNumStr)
        if(classifierResult!=classNumStr):
            errorCount+=1.0
    print "\nThe total number of errors is: %d" %errorCount
    print "\nThe total error rate is: %f" %(errorCount/float(mTest))















