import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import ann, csv, time, utils
from NN import NeuralNetwork

from sklearn.metrics import classification_report
from sklearn import metrics
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
cv_size = int(data_raw.shape[0] * 0)
data, data_cv = data[cv_size:], data[:cv_size]


X, y = get_X_labels(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_cv, y_cv = get_X_labels(data_cv)

def my_solution(X_train, X_test, y_train, y_test):
    # train with self
    dimensions = [X_train.shape[1], 3, y_train.shape[ 1]]
    params = ann.train(X_train, y_train, dimensions,
    alpha=0.1, batch_size=50,epoch=3472,momentum=0.04,tol=1e-10)
    y_pred = ann.predict(X_test, params, dimensions)

    y_pred = y_pred.argmax(axis=1)
    y_test = y_test.argmax(axis=1)

    # evaluate
    print('My implements1:')
    print(classification_report(y_test, y_pred))

    return params, y_pred, y_test

def my_solution2(X_train, X_test, y_train, y_test, hyperparams=None):
    if hyperparams == None:
        hyperparams = {
            'hidden_layer_sizes':(3, ),
            'learning_rate':0.1, 
            'epoch':3852,
            'momentum':0.04,
            'tol': 1e-10,
            'reg_coef': 0
        }
    
    if not hasattr(hyperparams, 'batch_size'):
        hyperparams['batch_size'] = X_train.shape[0]

    # train with self
    nn = NeuralNetwork(**hyperparams)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)

    y_pred = y_pred.argmax(axis=1)
    y_test = y_test.argmax(axis=1)

    # evaluate
    print('My implements2:')
    print(classification_report(y_test, y_pred))

    return nn, y_pred, y_test

def sklearn_solution(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(solver='sgd', activation='logistic', 
        hidden_layer_sizes=(3, ),
        learning_rate_init=0.1,
        batch_size=X_train.shape[0],
        max_iter=3852,
        shuffle=False,
        tol=1e-10,
        momentum=0.04,
        # verbose=True,
        random_state=1
        )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    y_pred = y_pred.argmax(axis=1)
    y_test = y_test.argmax(axis=1) 

    print('sklearn.MLPClassifier implements:')
    print(classification_report(y_test, y_pred))

    return clf, y_pred, y_test


def found_best_params(func, history_file='./hyperparams_history.csv', try_times=5):
    hyperparam_try_list = {
        'learning_rate': [0.01, 0.1, 1, 2],
        'epoch' : [555, 1382, 2582, 3852],
        'reg_coef': [0],
        'momentum': [0.04, 0, 1],
        'hidden_layer_sizes': [(3,), (5, )]
    }

    max_iter, params_index = 1, {}
    for (k, v) in hyperparam_try_list.items():
        max_iter *= len(v)
        params_index[k] = 0

    soft_mean = lambda x: (sum(x) - max(x) - min(x)) / (len(x) - 2) if len(x) > 2 else 0

    with open(history_file,"w") as csvfile: 
        writer = csv.writer(csvfile)

        # write csv header
        header = ['learning_rate', 'epoch', 'momentum', 'hidden_layer_sizes', 'reg_coef']
        header += ['fbeta_score', 'precision', 'recall', 'accuracy', 'time']
        writer.writerow(header)

        for i in range(max_iter):
            hyperparams = {}
            # set hyper params
            for (k, index) in params_index.items():
                hyperparams[k] = hyperparam_try_list[k][index]
            
            s_time_cost, s_precision,s_recall,s_fbeta_score,s_accuracy = [],[],[],[],[]

            for j in range(try_times):
                begin = time.time()
                nn, y_pred, y_test2 = func(X_train, X_test, y_train, y_test, hyperparams)
                end = time.time()

                # s -> ms
                s_time_cost.append((end-begin)*1000)

                accuracy = metrics.accuracy_score(y_test2, y_pred)
                precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test2, y_pred)
                s_precision.append(np.mean(precision))
                s_recall.append(np.mean(recall))
                s_fbeta_score.append(np.mean(fbeta_score))
                s_accuracy.append(np.mean(accuracy))

            #写入多行用writerows
            row = [
                hyperparams['learning_rate'],
                hyperparams['epoch'],
                hyperparams['momentum'],
                hyperparams['hidden_layer_sizes'],
                hyperparams['reg_coef'],
                soft_mean(s_fbeta_score),
                soft_mean(s_precision),
                soft_mean(s_recall),
                soft_mean(s_accuracy),
                soft_mean(s_time_cost)
            ]
            writer.writerow(row)

            print('{}/{} ... spend {:.0f} ms, fbeta_score={:.4f}'.format(i+1, max_iter, soft_mean(s_time_cost), soft_mean(s_fbeta_score)))

            # next pairs of hyperparams
            passed_k = []
            for (k, v) in hyperparam_try_list.items():
                if params_index[k] != len(v) - 1:
                    params_index[k] += 1
                    for pass_k in passed_k:
                        params_index[pass_k] = 0
                    break
                passed_k.append(k)


def hyperparams_analyze(history_file='./hyperparams_history.csv'):
    data = pd.read_csv(history_file)
    data = data.drop(columns=['reg_coef'])

    learning_rate = list(set(data['learning_rate']))
    epoch_set = list(set(data['epoch']))
    momentum_set = list(set(data['momentum']))
    hidden_layer_sizes_set = list(set(data['hidden_layer_sizes']))

    # learning rate
    rows = data[
        (data['learning_rate']==learning_rate[1]) & 
        (data['epoch']==epoch_set[0]) & 
        (data['momentum']==momentum_set[1])
        # & (data['hidden_layer_sizes']==hidden_layer_sizes_set[0])
    ]
    plt.plot(rows['hidden_layer_sizes'], rows['fbeta_score'])
    plt.show()





# found_best_params(my_solution2,history_file='NN_hyperparams.csv',try_times=10)

# found_best_params(sklearn_solution,history_file='sklearn_hyperparams.csv',try_times=7)


# my_solution(X_train, X_test, y_train, y_test)
# clf1,foo,bar = my_solution2(X_train, X_test, y_train, y_test)
# clf2,foo,bar = sklearn_solution(X_train, X_test, y_train, y_test)


# utils.plot_loss(clf1.loss_)
# utils.plot_loss([clf1.loss_, clf2.loss_curve_], 2)
hyperparams_analyze(history_file='NN_hyperparams.csv')
