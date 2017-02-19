import theano
import theano.tensor as T
import numpy as np
from regularizer import Regularizer as R
from utils import clip_norms

def Momentum(cost, params, lr=0.01, momentum=0.9, regularizer=R(), clipnorm=0.0):
    updates = []
    grads = T.grad(cost, params)
    grads = clip_norms(grads, clipnorm)
    for p,g in zip(params,grads):
        g = regularizer.gradient_regularize(p, g)
        m = theano.shared(p.get_value() * 0.)
        v = (momentum * m) - (lr * g)
        updates.append((m, v))
        
        updated_p = p + v
        updated_p = regularizer.weight_regularize(updated_p)
        updates.append((p, updated_p))
    return updates
