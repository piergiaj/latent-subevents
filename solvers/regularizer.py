import theano
import theano.tensor as T
import numpy as np


def l2norm(x, axis=1, e=1e-8):
    return T.sqrt(T.sum(T.sqr(x), axis=axis, keepdims=True) + e)

class Regularizer(object):

    def __init__(self, l1=0.0, l2=0.0, maxnorm=0.0, l2norm=False, frobnorm=False):
        self.l1 = l1
        self.l2 = l2
        self.maxnorm = maxnorm
        self.l2norm = l2norm
        self.frobnorm = frobnorm

    def max_norm(self, p, maxnorm):
        if maxnorm > 0:
            norms = T.sqrt(T.sum(T.sqr(p), axis=0))
            desired = T.clip(norms, 0, maxnorm)
            p = p * (desired / (1e-7 + norms))
        return p

    def l2_norm(self, p):
        return p / l2norm(p, axis=0)

    def frob_norm(self, p, nrows):
        return (p / T.sqrt(T.sum(T.sqr(p)))) * T.sqrt(nrows)

    def gradient_regularize(self, p, g):
        g += p * self.l2
        g += T.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        p = self.max_norm(p, self.maxnorm)
        if self.l2norm:
            p = self.l2_norm(p)
        if self.frobnorm:
            p = self.frob_norm(p, self.frobnorm)
        return p
