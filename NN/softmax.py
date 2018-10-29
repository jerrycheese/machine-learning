import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.
    Note that x may a D dimensional vector or N x D dimensional numpy matrix.

    Arguments:
    x -- vector or numpy matrix

    Return:
    x. modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Ndarray
        N = x.shape[0]
        x -= x.max(axis=1).reshape(N, 1)
        x = np.exp(x)/np.exp(x).sum(axis=1).reshape(N, 1)
    else:
        # Vector
        x -= x.max()
        x = np.exp(x)/np.exp(x).sum()

    assert x.shape == orig_shape
    return x


def test_softmax():
    """
    Some simple tests
    """
    print("Running softmax tests...")

    test1 = softmax(np.array([1,2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)
    print("Success!!\n")

if __name__ == "__main__":
    test_softmax()
