#!/bin/python
#coding=utf-8

#手写数字识别
#当k=3时分类结果最好，错误率为12%

from numpy import *
from os import listdir
import operator

def img2vector(filename):
	returnVector = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVector[0, 32 * i + j] = int(lineStr[j])

	return returnVector


#img2vector("digits/trainingDigits/0_1.txt")

def classify(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	#计算距离
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis = 1)


	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}

	#选择距离最小的k个点
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
		

	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]


def handwritingClassTest():
	hwLabels = []

	#训练集数据转换成向量
	trainingFileList = listdir('digits/trainingDigits')
	m = len(trainingFileList)
	trianingMat = zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trianingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)

	#测试集数据
	testFileList = listdir('digits/testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)

		#调用knn进行分类
		classifierResults = classify(vectorUnderTest, trianingMat, hwLabels, 3)
		print "分类结果为：%d, 实际类别为：%d" % (classifierResults, classNumStr)
		if(classifierResults != classNumStr):
			errorCount += 1.0

	print "总误差为：%d" % errorCount
	print "误差率为：%f" % (errorCount/float(mTest))

if __name__ == '__main__':

	handwritingClassTest()
