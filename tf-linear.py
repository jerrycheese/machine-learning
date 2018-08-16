import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_size', default=50, type=int, help='data size')
parser.add_argument('--num_steps', default=1000, type=int, help='number of trannig steps')
parser.add_argument('--devi_degree', default=10, type=int, help='Degree of deviation in stand linear data')
parser.add_argument('--alpha', default=0.00001, type=float, help='learning rate of gradient decent')
parser.add_argument('--method', default='fit_linear_model', type=str, help='the method to train')

def linear_data(data_size, devi_degree):
    """
    Make random linear data
    :param data_size: data size
    :param devi_degree: degree of deviation
    :return: linear data with x and y
    """
    # standard linear function
    x = np.array(range(data_size), dtype=np.float64)
    y = 3 * x + 0.6

    # make deviation
    y += np.random.randn(data_size) * devi_degree

    data = pd.DataFrame({'x': x, 'y': y})

    return data

def evaluate(train_set, test_set, W, b):
    """
    Evaluate the model's loss
    :param train_set:
    :param test_set:
    :param W:
    :param b:
    :return: train_loss, evaluate_loss
    """
    x = tf.placeholder(tf.float64)
    y = tf.placeholder(tf.float64)

    # predict
    pred = W * x + b

    # loss
    loss = tf.reduce_sum(tf.square(pred - y))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_loss = sess.run(loss, {x: train_set['x'], y: train_set['y']})
    evaluate_loss = sess.run(loss, {x: test_set['x'], y: test_set['y']})

    sess.close()

    return train_loss, evaluate_loss

def print_linear_model(data, W, b):
    """
    print the data and the predictions of linear model
    :param data:
    :param W: W of linear model
    :param b: b of linear model
    """
    x = np.array(data['x'])
    y = np.array(data['y'])

    pred = np.array(W * x + b)

    plt.scatter(x, y, linewidths=1)
    plt.plot(x, pred, color='red')

    plt.show()

def fit_linear_model(data, num_steps, alpha):
    """
    train with the machine learning

    :param data: training data
    :param num_steps: training steps
    :param alpha: learning rate
    :return: W and b of trained linear model
    """
    # variables
    W = tf.Variable(1, dtype=tf.float64)
    b = tf.Variable(1, dtype=tf.float64)
    x = tf.placeholder(tf.float64)
    y = tf.placeholder(tf.float64)

    # predict
    pred = W * x + b

    # loss
    loss = tf.reduce_sum(tf.square(pred - y))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(alpha)

    # train
    train = optimizer.minimize(loss)

    train_set, test_set = split_test_set(data, frac=0.3, random=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        sess.run(train, {x: train_set['x'], y: train_set['y']})

    final_W, final_b = sess.run([W, b])

    # evaluate
    final_loss, evaluate_loss = evaluate(train_set, test_set, final_W, final_b)

    print('W: {}, b: {}, final loss: {}, evaluate loss: {}'.format(final_W, final_b, final_loss, evaluate_loss))

    return final_W, final_b

def fit_estimator(data, num_steps):
    """
    train with estimator

    :param data:
    :param num_steps:
    :return:
    """
    feature_columns = [
        tf.feature_column.numeric_column('x')
    ]

    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    train_set, test_set = split_test_set(data, frac=0.3, random=True)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': train_set['x']}, train_set['y'], batch_size=4, num_epochs=None, shuffle=True
    )

    estimator.train(input_fn=input_fn, steps=num_steps)

    W = estimator.get_variable_value('linear/linear_model/x/weights')
    b = estimator.get_variable_value('linear/linear_model/bias_weights')
    final_W, final_b = float(W), float(b)

    final_loss, evaluate_loss = evaluate(train_set, test_set, final_W, final_b)

    print('W: {}, b: {}, final loss: {}, evaluate loss: {}'.format(final_W, final_b, final_loss, evaluate_loss))

    return final_W, final_b

def fit_custom_estimator(data, num_steps, alpha):
    """
    train with custom estimator

    :param data:
    :param num_steps:
    :param alpha:
    :return:
    """

    def model_fn(features, labels, mode):
        W = tf.get_variable('W', 1., dtype=tf.float64)
        b = tf.get_variable('b', 1., dtype=tf.float64)

        # predict
        pred = W * tf.cast(features['x'], dtype=tf.float64) + b

        # loss
        loss = tf.reduce_sum(tf.square(pred - labels))

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(alpha)

        # global step
        global_step = tf.train.get_global_step()

        # train
        train = tf.group(
            optimizer.minimize(loss),
            tf.assign_add(global_step, 1)
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred,
            loss=loss,
            train_op=train
        )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn
    )

    train_set, test_set = split_test_set(data, frac=0.3, random=True)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': train_set['x']}, train_set['y'], batch_size=4, num_epochs=None, shuffle=True
    )

    estimator.train(input_fn=input_fn, steps=num_steps)

    W = estimator.get_variable_value('W')
    b = estimator.get_variable_value('b')
    final_W, final_b = float(W), float(b)

    final_loss, evaluate_loss = evaluate(train_set, test_set, final_W, final_b)

    print('W: {}, b: {}, final loss: {}, evaluate loss: {}'.format(final_W, final_b, final_loss, evaluate_loss))

    return final_W, final_b

def main(argv):
    args = parser.parse_args(argv[1:])

    data = linear_data(args.data_size, args.devi_degree)

    if args.method == 'fit_linear_model':
        W, b = fit_linear_model(data, args.num_steps, args.alpha)
    elif args.method == 'fit_estimator':
        W, b = fit_estimator(data, args.num_steps)
    elif args.method == 'fit_custom_estimator':
        W, b = fit_custom_estimator(data, args.num_steps, args.alpha)
    else:
        print('Invalid method "{}"'.format(args.method))
        exit(-1)

    print_linear_model(data, W, b)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
