import numpy as np


def gen_batches(n, batch_size):
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

def params_unpack(params, sizes):
    """Unpack a one dimentional array to a matrixes

    Parameters:
    params --  one dimentional array
    sizes --  matirx sizes

    Return:
    matrixes
    """
    ofs = 0
    packed = []
    for size in sizes:
        length = size[0]*size[1]
        mat = np.reshape(params[ofs:ofs+ length], (size[0], size[1]))
        ofs += length
        packed.append(mat)
    return packed

def params_pack(params):
    """Pack a params matrixes into a one dimentional list sequential

    Parameters:
    params --  matirxes or array-like

    Return:
    one dimentional array
    """
    one_dims = []
    for p in params:
        if not hasattr(p, 'flatten'):
            p = np.array(p)
        one = p.flatten()
        one_dims.append(one)
    return np.concatenate(one_dims)