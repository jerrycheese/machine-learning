
import numpy as np
import random

def gradcheck(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- function which output the y, grad = f(x)
    x -- the x point value (numpy array)

    Return:
    True or False
    """

    # I use central difference to get the 
    #  approximate gradient of f(x) at point x
    #  so the numdgrad = (f(x+h)-f(x-h))/(2*h)
    h = 1e-4

    # If f(x) is using random module, there may be some bugs on 
    #  calling f(x), f(x+h) and f(x-h).Using random.setstate before
    #  calling f(x) to avoid these bugs.
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Beacuse the x is muilty dimentional array,
        #  so I actually get the partial derivative ∂y/∂xi here

        x[ix] += h
        random.setstate(rndstate)
        y1 = f(x)[0]

        x[ix] -= 2*h
        random.setstate(rndstate)
        y2 = f(x)[0]

        x[ix] += h
        numgrad = (y1 - y2)/(2 * h)

        # Compare gradients 
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Function gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return False

        it.iternext() # Step to next dimension
    
    return True

def test_gradcheck():
    """
    Some simple tests.
    """
    

    print("Running gradcheck tests...")

    f1 = lambda x: (np.sum(x ** 3), 3 * x**2)

    assert gradcheck(f1, np.array(123.456))      # scalar test
    assert gradcheck(f1, np.random.randn(3,))    # 1-D test
    assert gradcheck(f1, np.random.randn(4,5))   # 2-D test

    print("Success!!\n")


if __name__ == "__main__":
    test_gradcheck()       
