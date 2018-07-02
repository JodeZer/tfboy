import matplotlib.pyplot as plt
import csv
import numpy as np
import math
import collections

SHOPID = 0
ORDER_COUNT = 2
BUYER_COUNT = 3
AMOUNT = 4
OVERTIME_RATE = 5
CANCEL_RATE = 6
DISPATCH_AGE = 7
# MARCO
def log2(val):
    return math.log2(val)

def normal(val):
    return val

def log10(val):
    return math.log10(val)

COLUMNS = collections.OrderedDict(
    [
        (ORDER_COUNT, [log2, lambda x:float(x), lambda x: False]),
        (BUYER_COUNT, [log2, lambda x:float(x), lambda x: False]),
        (AMOUNT, [log2, lambda x:float(x), lambda x: False]),
        (OVERTIME_RATE, [log2, lambda x:float(x)*100+0.01, lambda x: x>0.5*100]), 
        (CANCEL_RATE, [log2, lambda x:float(x)*100+0.01,lambda x: x > 0.10*100]), 
        (DISPATCH_AGE, [log2, lambda x:float(x),lambda x: False]), 
    ]
)

def loadDataInColumn(fileName, columns):
    ret = []
    for i in range(len(columns.keys())):
        ret.append([])
    with open(fileName, 'rt') as file:
        reader = csv.reader(file)
        for row in reader:
            for i, ind in enumerate(columns.keys()):
                defList = columns.get(ind)
                val = defList[1](row[ind])
                if defList[2](val):
                    #print("ignore")
                    continue
                val = defList[0](val)
                #print("origin {}, log2 {}".format(defList[1](row[ind]),val))
                ret[i].append(val)
    return ret

def shit_filter(val):
    return False

def stat_avg_dev(data):
    return (np.std(a = data, ddof= 1),  np.average(a= data))
   
def stat_min_max(data):
    return (np.min(a = data), np.max(a = data))

def calXYAxis(data, intervalCount):
    (mini, maxi) = stat_min_max(data)
    gap = float(maxi - mini) / float(intervalCount)
    yaxis = [0]*(intervalCount+1)
    for val in data:
        yaxis[int((val - mini)/gap)] += 1
    return (intervalAsix(mini, maxi, intervalCount+1), yaxis)

def intervalAsix(mini, maxi, count):
    ret = [mini]*count
    gap = float(maxi -mini)/ count
    for i in range(count):
        ret[i] = mini+i*gap+0.5*gap
    return ret

def incArray(length):
    ret = [0]*length
    for i in range(length):
        ret[i] = i
    return ret

# [(x,y),(x,y)]
def plotXY(rawData):
    for i, t in enumerate(rawData):
        (x, y) = t
        print(x)
        print(y)
        #print(stat_avg_dev(x))
        print()
        plt.subplot(len(rawData), 1, i+1)
        plt.plot(x, y)
    plt.show()

def drawAxisData(data, gap):
    drawData = []
    for row in data:
        (x, y) = calXYAxis(row, gap)
        drawData.append((x,y))
    return drawData

def main():
    data = loadDataInColumn("hotsale_score_cat.csv", COLUMNS)
    #print(data)
    drawData = drawAxisData(data, 50)
       
    plotXY(drawData)

if __name__ == "__main__":
    main()

