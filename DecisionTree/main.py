
import numpy as np
import pandas as pd
from DecisionTree import DecisionTree

def main(argv):
    args = parser.parse_args(argv[1:])


    data = pd.read_csv(args.file)
    data = np.array(data)
    X, y = data[:,1:-1].tolist(), data[:,-1].tolist()
    

    dt = DecisionTree()
    dt.train(X, y, ['outlook','temperature','humidity','wind'])

    dt.plot()


if __name__ == '__main__':
    import sys, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='data file, only for csv')
    
    main(sys.argv)
