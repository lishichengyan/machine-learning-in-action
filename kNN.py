#%%
from numpy import *
import operator

def creatDataSet():
    group = array(
        [[1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]]
    )
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#%%
import matplotlib.pyplot as plt
group, labels = creatDataSet()
plt.scatter(group[:,0], group[:,1])
plt.show()

# %%
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # print(tile(inX, (dataSetSize, 1)))
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # print(diffMat)
    sqDiffMat = diffMat**2
    # print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    # print(sqDistances)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0)  + 1
    sortedClassCount = sorted(
        classCount.items(),  # the book is wrong
        key=operator.itemgetter(1),
        reverse=True 
    )
    return sortedClassCount[0][0]

classify0([0,0], group, labels, 3)

# %%
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    labels = {'didntLike':1,'smallDoses':2,'largeDoses':3}  # need to add this line
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(labels[listFromLine[-1]])  # the book is wrong
        index += 1
    return returnMat, classLabelVector

#%%
datingDataMat, datingLabels = file2matrix('./datingTestSet.txt')
print(datingDataMat)
print(datingLabels)

# %%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

#%%
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m ,1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

#%%
normMat, ranges, minVals = autoNorm(datingDataMat)
print(normMat)

# %%
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('./datingTestSet.txt')
    normMat, _, _ = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(hoRatio*m)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print(f"prediciton: {classifierResult}, real answer: {datingLabels[i]}")
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("error rate: ", errorCount / numTestVecs)
datingClassTest()
# %%
