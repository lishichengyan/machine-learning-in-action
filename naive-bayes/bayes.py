#%% 
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

#%%
def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#%%
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: " + word + " not in my Vocabulary!")
    return returnVec

#%%
listOfPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOfPosts)
print(myVocabList)

# %%
print(setOfWords2Vec(myVocabList, listOfPosts[0]))

# %%
from numpy import *
def trainNB0(trainMatrix, trainCategory):
    """
    [
        [1, 0, 1, 0],  # 0
        [0, 1, 1, 1],  # 1
        [1, 0, 0, 1],  # 1
        [1, 0, 0, 0],  # 0
    ]
    [0, 1, 1, 0]

    p(c|W) = p(W|c)p(c) / p(W)
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])  # each row is vector
    pAbusive = sum(trainCategory) / numTrainDocs
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom  # change to log to avoid underflow
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive

#%%
trainMat = []
for postinDoc in listOfPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = trainNB0(trainMat, listClasses)
print("p0V: ", p0V)
print("p1V: ", p1V)
print("pAb: ", pAb)

# %%
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    return 1 if p1 > p0 else 0

def testingNB():
    listOfPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, " classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, " classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()
# %%
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# %%
