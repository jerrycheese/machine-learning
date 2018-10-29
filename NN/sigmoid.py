#!/usr/bin/env python

import numpy as np


def sigmoid(x):
    """Compute the sigmoid function for the input x.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    s = 1/(1+np.exp(-x))

    return s


def sigmoid_grad(s):
    """Compute the gradient for the sigmoid function.
    Note that the input s should be the sigmoid 
    function value of the input x.

    Arguments:
    s -- s = sigmoid(x)

    Return:
    ds -- sigmoid gradient at x
    """

    ds = s*(1-s)

    return ds


def test_sigmoid():
    """
    Some simple tests
    """
    print("Running sigmoid tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)

    print(f)
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)

    print(g)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print("Success!!\n")


if __name__ == "__main__":
    test_sigmoid()
