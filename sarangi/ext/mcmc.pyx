cdef extern from "mcmc_.h":
    void propagate_bistable_(float *x0, float *y0, int n_steps, unsigned short rng_seed);

import numpy as np
cimport numpy as np


def propagate_bistable(x, y, n_steps=100, random_seed=None):
    cdef float[:] X = np.zeros((1,), np.float32)
    cdef float[:] Y = np.zeros((1,), np.float32)
    if random_seed is None:
        random_seed = np.random.randint(0x10000)
    X[0] = x
    Y[0] = y
    propagate_bistable_(&X[0], &Y[0], n_steps, random_seed)
    return X[0], Y[0]
