import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utils import *


class BasicNN(object):

    def __init(self, hidden_layers=(2,)):
        pass

    def fit(self, X, y, alpha=0.001, max_iteration=10000):
        pass

    def predict(self, X):
        pass

    def show(self):
        pass



def _main(argv):
    args = parser.parse_args(argv[1:])


    data_raw = pd.read_csv('../data/iris.csv', index_col=0)
    
    # shuffle
    data_raw = data_raw.sample(frac=1).reset_index(drop=True)

    # data cleaning

    # data convert 
    
    # split dataset
    cv_size = data_raw.shape[0] * 0.3
    data, data_cv = data_raw[:cv_size], data_raw[cv_size:]

    # train
    clf = BasicNN(10)
    clf.fit(data.drop(columns=['type'], axis=1), data['type'])

    # evaluate
    y_pred = clf.predict(data_cv.drop(columns=['type'], axis=1))
    y_true = data_cv['type'].tolist()
    classification_report(y_true, y_pred)


if __name__ == '__main__':
    import sys, argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--file', type=str, help='data file, only for csv')
    
    _main(sys.argv)
