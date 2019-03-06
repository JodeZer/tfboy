import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

def ImageExpand(data, cur, target):
    scale = target//cur
    newData = np.zeros((target, target), dtype= np.float32)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            newData[scale*i:scale*(i+1),scale*j:scale*(j+1)] = value
    return newData

def BatchImageExpand(datas, target):
    newDatas = np.zeros((datas.shape[0], target, target))
    for i in range(len(newDatas)):
        newDatas[i] = ImageExpand(datas[i], datas[i].shape[0], target)
    return newDatas

def plotDigit(some_digit_image):
    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

class DataSet():
    def __init__(self, dataX, dataY, testRate = 0.3, shuffle=True, batchSize = 16):
        self.batchSize = batchSize
        self.iter, self.epoch = 0, 1
        self.testRate = testRate
        self.dataSzie = len(dataX)
        if shuffle:
            self._shuffle()
        self._split(dataX, dataY)
        self.trainSize = len(self.trainX)
        self.testSize = len(self.testX)
    
    def _shuffle(self):
        pass
    
    def _split(self, dataX, dataY):
        splitP = int(len(dataX)*(1-self.testRate))
        self.trainX, self.trainY, self.testX, self.testY = dataX[:splitP], dataY[:splitP], dataX[splitP:], dataY[splitP:]

    def getNextBatch(self):
        if self.iter*self.batchSize >= self.trainSize:
            return None
        l, r = self.iter*self.batchSize, (self.iter+1)*self.batchSize
        self.iter += 1
        return self.trainX[l:r], self.trainY[l:r]
    
    def getTestSet(self):
        return self.testX, self.testY

    def nextEpoch(self):
        self.iter = 0


# test bach
# ds = DataSet([1,2,3,4,5,6,7,8,9,10], [0]*10, testRate=0.3, batchSize=3)
# for _ in range(10):
#     print(ds.getNextBatch())
    

class learnCurve():
    def __init__(self):
        self.loss = []
        self.rate = [[],[]]
    
    def append(self, loss, trainRate, validRate):
        self.loss.append(loss)
        self.rate[0].append(trainRate)
        self.rate[1].append(validRate)
        
    def plotRate(self):
        
        plt.plot(range(1,len(self.rate[0])+1), self.rate[0])
        plt.plot(range(1,len(self.rate[1])+1), self.rate[1])
        plt.show() 
        
# lc = learnCurve()
# for i in range(10):
#     lc.append(10-i, (10-i)*0.1, (9-i)*0.1)

# lc.plotRate()
        

