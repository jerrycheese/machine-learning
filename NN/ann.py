#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck


def _forward_backward_prop(X, labels, params, dimensions, penalty=0.001):
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

    M = X.shape[0]
    
    # ***** FORWORD PASS

    # forward
    # layer1 -> layer2
    z1 = np.dot(X, W1) + b1
    h = sigmoid(z1)

    # layer2 -> layer3
    z2 = np.dot(h, W2) + b2

    # ** OUTPUT WITH SIGMOID
    # y = sigmoid(z2)
    # cost = (-labels*np.log(y) - (1-labels)*np.log(1-y)).sum()/M

    # ** OUTPUT WITH SOFTMAX
    y = softmax(z2)
    cost = (-labels*np.log(y)).sum()/M

    # regularation
    cost += penalty*(np.power(W1, 2).sum() + np.power(W2, 2).sum())/(2*M)

    # ****** BACKPROP
    # layer3
    d3 = (y - labels)/M


    # layer3 -> layer2
    # ∂cost/∂z2  shape=(M,Dy)
    gradW2 = np.dot(np.transpose(h), d3)
    gradW2 += W2*penalty/M
    gradb2 = d3.sum(axis=0)
    d2 = np.dot(d3, np.transpose(W2)) * sigmoid_grad(h)

    # layer2 -> layer3
    # ∂cost/∂h shape=(M,H)
    gradW1 = np.dot(np.transpose(X), d2)
    gradW1 += W1*penalty/M
    gradb1 = d2.sum(axis=0)

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def train(X, labels, dimensions, 
        alpha=0.1, 
        batch_size=10, 
        epoch=1000, 
        momentum=0,tol=1e-10,
        verbose=False):
    """Train neural network with one hidden layer only

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """
    assert len(dimensions) == 3

    batch_size = min(batch_size, X.shape[0])
    params = np.random.randn(params_count(dimensions))
    velocities = np.zeros_like(params)
    loss = []

    for i in range(epoch):
        sum_cost = 0
        for batch_slice in _gen_batches(X.shape[0], batch_size):
            _X, _y = X[batch_slice], labels[batch_slice]
            cost, grad = _forward_backward_prop(_X, _y, params, dimensions)

            # apply gradients
            updates = momentum*velocities - alpha*grad
            velocities = updates
            params = params + updates

            sum_cost += cost * (batch_slice.stop - batch_slice.start)
        
        sum_cost /= X.shape[0]
        loss.append(sum_cost)

        if i > 2 and loss[i] - loss[i-1] > -tol and loss[i-1] - loss[i-2] > -tol:
            break

        if verbose:
            print('Iteration {:d}, cost = {:f}'.format(i+1, sum_cost))

    return params

def predict(X, params, dimensions):
    assert len(dimensions) == 3

    W1, b1, W2, b2 = unpack_parmas(params, dimensions)

    z1 = np.dot(X, W1) + b1
    h = sigmoid(z1)

    z2 = np.dot(h, W2) + b2
    y = softmax(z2)
    y = y.argmax(axis=1)
    n_label = np.max(y) + 1
    y = np.eye(n_label,dtype=np.int64)[y]
    
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
        _forward_backward_prop(data, labels, params, dimensions), params)
    
    print("Success!!\n")


def _gen_batches(n, batch_size):
    """Generator to create slices containing batch_size elements, from 0 to n.
    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.
    --------
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)

if __name__ == "__main__":
    ann_test()
