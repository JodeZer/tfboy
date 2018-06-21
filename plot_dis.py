import matplotlib.pyplot as plt
import csv
import numpy as np
import math
import collections

ORDER_COUNT = 2
BUYER_COUNT = 3
AMOUNT = 4
IGNORE = 7

# MARCO
def log2(val):
    return int(math.log2(val))

def normal(val):
    return int(val)

COLUMNS = collections.OrderedDict(
    [
        (ORDER_COUNT, log2),
        (BUYER_COUNT, log2),
        (AMOUNT, log2),
    ]
)

def loadDataInColumn(fileName):
    ret = []
    for i in range(len(COLUMNS.keys())):
        ret.append([])
    with open(fileName, 'rt') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if row[IGNORE] == "I":
                continue
            for i, ind in enumerate(COLUMNS.keys()):
                val = int(row[ind])
                if shit_filter(val):
                    continue
                ret[i].append(COLUMNS.get(ind)(val))
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
        ret[i] = i*gap
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
        plt.subplot(len(rawData), 1, i+1)
        plt.plot(x, y)
    plt.show()


def main():
    data = loadDataInColumn("hotsale_score_right.csv")
    
    drawData = []
    for row in data:
        (x, y) = calXYAxis(row, 6)
        drawData.append((x,y))
       
    plotXY(drawData)

if __name__ == "__main__":
    main()

