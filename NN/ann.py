#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck


def forward_backward_prop(X, labels, params, dimensions):
    """One hidden layer only

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    assert len(dimensions) == 3
    
    W1, b1, W2, b2 = unpack_parmas(params, dimensions)
    
    # forward

    # I put bias weight at the first column of W1, W2
    # meanwhile, put the `1` cell at the first position in every layers.

    M = X.shape[0]

    # forward
    # layer1 -> layer2
    z1 = np.dot(X, W1) + b1
    h = sigmoid(z1)

    # layer2 -> layer3
    z2 = np.dot(h, W2) + b2

    # ** STYLE1
    # y = sigmoid(z2)
    # cost = (-labels*np.log(y) - (1-labels)*np.log(1-y)).sum()/M

    # ** STYLE2
    y = softmax(z2)
    cost = (-labels*np.log(y)).sum()/M

    penalty = 0.00001
    cost += penalty*(np.power(W1, 2).sum() + np.power(W2, 2).sum())/(2*M)

    # backward

    # layer3 -> layer2
    # ∂cost/∂z2  shape=(M,Dy)
    d3 = (y - labels)/M
    gradW2 = np.dot(np.transpose(h), d3)
    gradW2 += W2*penalty/M
    gradb2 = d3.sum(axis=0)

    # layer2 -> layer3
    # ∂cost/∂h shape=(M,H)
    d2 = np.dot(d3, np.transpose(W2)) * sigmoid_grad(h)
    gradW1 = np.dot(np.transpose(X), d2)
    gradW1 += W1*penalty/M
    gradb1 = d2.sum(axis=0)

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def train(X, labels, dimensions, alpha=0.1, batch_size=10, epoch=1000, momentum=1):
    # d = 1e-8
    notice_len = 1000
    last_cost = 99999999

    batch_size = min(batch_size, X.shape[0])

    max_iteration = int(X.shape[0]*epoch/batch_size)

    ofs = 0

    params = []
    random_state = np.random.RandomState(1)
    for i in range(1, len(dimensions)):
        params_len = (dimensions[i-1] + 1) * dimensions[i]
        bound = 2./(dimensions[i] + dimensions[i-1])
        params = params +  random_state.uniform(-bound, bound, params_len).tolist()

    # params = np.random.randn(params_count(dimensions))

    velocities = np.zeros_like(params)

    # print(params)

    for i in range(max_iteration):
        
        begin, end = ofs, (ofs+batch_size)%X.shape[0]

        if begin < end:
            _X, _y = X[begin:end, :], labels[begin:end, :]
        else:
            _X = np.vstack((X[begin:, :],X[:end,:]))
            _y = np.vstack((labels[begin:, :],labels[:end,:]))

        ofs = end

        cost, grad = forward_backward_prop(_X, _y, params, dimensions)
        updates = momentum*velocities - alpha*grad
        velocities = updates
        params = params + updates

        # r = abs((last_cost-cost)/last_cost)
        # if r < d:
        #     break
        if (i + 1)%notice_len == 0:
            print('Iteration {:d}, cost = {:f}'.format(i+1, cost))
        # last_cost = cost

    return params


def predict(X, params, dimensions):
    assert len(dimensions) == 3

    W1, b1, W2, b2 = unpack_parmas(params, dimensions)

    z1 = np.dot(X, W1) + b1
    h = sigmoid(z1)

    z2 = np.dot(h, W2) + b2
    y = softmax(z2)
    y = (y==y.max(axis=0)).astype(np.int64)
    return y

def unpack_parmas(params, dimensions):
    assert len(dimensions) == 3
    
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    return W1, b1, W2, b2


def params_count(dimensions):
    """Init the weights and bias randomly

    Aruguments:

    dimensions -- (input_feature_size, hidden1_size, hidden2_size, .., output_size)
    
    Return:
    random params
    """
    params_len = 0
    for i in range(1, len(dimensions)):
        params_len += (dimensions[i - 1] + 1) * dimensions[i]
    
    return params_len

def ann_test():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running ann tests...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params_len = params_count(dimensions)
    params = np.random.randn(params_len)

    assert gradcheck(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)
    
    print("Success!!\n")


if __name__ == "__main__":
    ann_test()
