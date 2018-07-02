
import pandas as pd
import collections
import numpy as np
import tensorflow as tf

COLUMN_TYPES = collections.OrderedDict([
    ("order_count", float),
    ("customer_count", float),
    ("amout", float),
    ("prod_count", int),
    ("score", float)
])

def readData():
    return pd.read_csv("data.csv", names=COLUMN_TYPES.keys(),dtype=COLUMN_TYPES)

def loaddata(yname="y",train_frac=0.7,seed=None):
    # read data
    pdframe = readData()

    # shuffle seeds
    np.random.seed(seed)

    # split
    x_train = pdframe.sample(frac=train_frac, random_state=seed)
    x_test = pdframe.drop(x_train.index)

    y_train = x_train.pop(yname)
    y_test = x_test.pop(yname)

    return (x_train, y_train),(x_test, y_test)

def make_dataset(batch_sz, x, y=None, shuffle=False, shuffle_buffer_size=1000):

    def input_fn():
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(dict(x))
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_sz).repeat()
        else:
            dataset = dataset.batch(batch_sz)
        return dataset.make_one_shot_iterator().get_next()

    return input_fn

def main(argv):

    (train_x,train_y), (test_x, test_y) = loaddata(yname="score")

    train_input_fn = make_dataset(100, train_x, train_y, True, 1000)

    test_input_fn = make_dataset(100, test_x, test_y, True, 1000)

    feature_columns = [ 
        tf.feature_column.numeric_column(key="order_count"),
        tf.feature_column.numeric_column(key="customer_count"),
        tf.feature_column.numeric_column(key="amout"),
        tf.feature_column.numeric_column(key="prod_count")
    ]

    model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    model.train(input_fn=train_input_fn, steps=1000)

    evaluate(model,test_input_fn)

    #perdict(model)

    wt_names = model.get_variable_names()
    for name in wt_names:
        print("{}:{}".format(name, model.get_variable_value(name)))

def evaluate(model, testFn):
    eval_result = model.evaluate(input_fn=testFn)
    print(eval_result)

def perdict(model):
    input_dict = {
      "f1": np.array([2000, 3000]),
      "f2": np.array([30, 40])
    }

    predict_input_fn = make_dataset(1, input_dict)
    predict_results = model.predict(input_fn=predict_input_fn)
    print("\nPrediction results:")
    for i, prediction in enumerate(predict_results):
        msg = ("f1: {: 10f}, "
            "f2: {: 10f}, "
            "y: {: 10f}")
        msg = msg.format(input_dict["f1"][i], input_dict["f2"][i],prediction["predictions"][0])

    print("    " + msg)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)