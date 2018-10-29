import numpy as np
import pandas as pd
import ann

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def get_X_labels(data):
    X, labels = data.drop(columns=['type'], axis=1), pd.get_dummies(data['type'])
    X, labels = np.array(X), np.array(labels,dtype=np.int64)
    return X, labels


data_raw = pd.read_csv('./iris.csv', index_col=0)

# shuffle
data_raw = data_raw.sample(frac=1).reset_index(drop=True)
data = data_raw.copy()

# data cleaning


# data convert

# data scaling
scaler = lambda x: (x-np.mean(x))/(np.max(x)-np.min(x))
data[['f1','f2','f3','f4']] = data[['f1','f2','f3','f4']].apply(scaler)

# split dataset
cv_size = int(data_raw.shape[0] * 0.2)
data, data_cv = data[cv_size:], data[:cv_size]

X, y = get_X_labels(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_cv, y_cv = get_X_labels(data_cv)

def my_solution(X_train, X_test, y_train, y_test):

    # train with self
    dimensions = [X_train.shape[1], 3, y_train.shape[1]]
    params = ann.train(X_train, y_train, dimensions,alpha=1, batch_size=50,epoch=5000,momentum=0)
    y_pred = ann.predict(X_test, params, dimensions)

    y_pred = y_pred.argmax(axis=1)
    y_test = y_test.argmax(axis=1)

    df = pd.DataFrame(X_test)
    df['yp'] = y_pred
    df['y'] = y_test
    print(df)

    # evaluate
    print(classification_report(y_test, y_pred))

    # cv evaluate
    # X_cv, y_cv = get_X_labels(data_cv)
    # y_pred = ann.predict(X_cv, params, dimensions)

    # y_cv = y_cv.argmax(axis=1)
    # y_pred = y_pred.argmax(axis=1)
    # print(classification_report(y_cv, y_pred))


def sklearn_solution(X_train, X_test, y_train, y_test):
    y_train = y_train.argmax(axis=1)
    y_test = y_test.argmax(axis=1)

    ## training using sklearn
    
    clf = MLPClassifier(solver='sgd', activation='logistic', 
        hidden_layer_sizes=(3, ),
        learning_rate_init=0.1,
        batch_size=50,
        max_iter=3472,
        shuffle=False,
        tol=1e-20,
        momentum=0,
        # verbose=True,
        # random_state=1
        )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    df = pd.DataFrame(X_test)
    df['yp'] = y_pred
    df['y'] = y_test
    print(df)

    print(classification_report(y_test, y_pred))

my_solution(X_train, X_test, y_train, y_test)
sklearn_solution(X_train, X_test, y_train, y_test)

