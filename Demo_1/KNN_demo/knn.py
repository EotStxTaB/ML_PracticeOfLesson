# coding:utf-8
'''KNN分类器'''

import numpy as np
from numpy import *
import operator

## 载入训练集
def createDataSet():
    filepath = './train_data.txt' # 数据文件路径
    group = np.loadtxt(filepath,dtype = float,usecols = (1,2,3,4),  delimiter = '\t')
    label2 = np.loadtxt('./train_data.txt',dtype = str, usecols = (5,), delimiter = '\t')       
    labels = label2 
    return group,labels

## KNN分类器
def classify(input,dataSet,label,k):
    dataSize = dataSet.shape[0]
    #print(dataSize) =135
    
    # 欧式距离
    diff = tile(input,(dataSize,1))-dataSet
    sqdiff = diff**2
    squareDist = sum(sqdiff,axis = 1)#行向量分别相加
    dist = squareDist**0.5
    
    ##对距离从小到大进行排序，返回下标
    sortedDistIndex = argsort(dist)
    #print(sortedDistIndex)

    classCount={}
    maxCount = 0
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1  #对选取的K个样本所属的类别个数进行统计
    # 找出现次数最多的类别
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key
    return classes
