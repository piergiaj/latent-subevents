import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def dropout(srng, X, p=0., shape=None):
    if shape is None:
        shape = X.shape
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob # inverse dropout, so we don't have to do any scaling at test time
    return X


#def dropout(rng, x, p=0.5):
#    """ Zero-out random values in x with probability p using rng """
#    if p > 0. and p < 1.:
#        seed = int(rng.normal([1]) * 2**30)
#        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
#        mask = srng.binomial(n=1, p=1.-p, size=x.shape,
#                dtype=theano.config.floatX)
#        return x * mask
#    return x


#def fast_dropout(rng, x):
#    """ Multiply activations by N(1,1) """
#    seed = rng.randint(2 ** 30)
#    srng = RandomStreams(seed)
#    mask = srng.normal(size=x.shape, avg=1., dtype=theano.config.floatX)
#    return x * mask
