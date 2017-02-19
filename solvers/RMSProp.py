import theano
import theano.tensor as T
import numpy as np
from regularizer import Regularizer as R
from utils import clip_norms

def RMSProp(cost, params, lr=0.001, rho=0.9, epsilon=1e-6, regularizer=R(), clipnorm=0.0):
    updates = []
    grads = T.grad(cost, params)
    grads = clip_norms(grads, clipnorm)
    for p,g in zip(params,grads):
        g = regularizer.gradient_regularize(p, g)
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        updated_p = p - lr * (g / T.sqrt(acc_new + epsilon))
        updated_p = regularizer.weight_regularize(updated_p)
        updates.append((p, updated_p))
    return updates
