import plot_dis
import collections
from sklearn import preprocessing

COLUMNS = collections.OrderedDict(
    [
        (plot_dis.ORDER_COUNT, [plot_dis.log2, lambda x:float(x), lambda x: False]),
        (plot_dis.BUYER_COUNT, [plot_dis.log2, lambda x:float(x), lambda x: False]),
        (plot_dis.AMOUNT, [plot_dis.log2, lambda x:float(x), lambda x: False]),
    ]
)

COLUMNS2 = collections.OrderedDict(
    [
        (plot_dis.ORDER_COUNT, [plot_dis.normal, lambda x:float(x), lambda x: False]),
        (plot_dis.BUYER_COUNT, [plot_dis.normal, lambda x:float(x), lambda x: False]),
        (plot_dis.AMOUNT, [plot_dis.normal, lambda x:float(x), lambda x: False]),
    ]
)

def main():
    data = plot_dis.loadDataInColumn("hotsale_score_cat.csv", COLUMNS)
    #plot_dis.plotXY(plot_dis.drawAxisData(data))
    normalData = preprocessing.scale(data,copy=False,axis = 1,with_std=True,with_mean=True)

    print("mean:{}, std:{}".format(normalData.mean(axis=1), normalData.std(axis=1)))
    print(normalData)
    drawData = plot_dis.drawAxisData(normalData, 10)
    plot_dis.plotXY(drawData)
    

if __name__ == "__main__":
    main()