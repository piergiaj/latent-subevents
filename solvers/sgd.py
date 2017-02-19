import theano
import theano.tensor as T
import numpy as np
from regularizer import Regularizer as R
from utils import clip_norms

def SGD(cost, params, lr=0.01, regularizer=R(), clipnorm=0.0):
    updates = []
    grads = T.grad(cost, params)
    grads = clip_norms(grads, clipnorm)
    for p,g in zip(params,grads):
        g = regularizer.gradient_regularize(p, g)
        updated_p = p - lr * g
        updated_p = regularizer.weight_regularize(updated_p)
        updates.append((p, updated_p))
    return updates
