import numpy as np
import theano


def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        fan_in = np.prod(shape[1:])
        fan_out = shape[0]
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


class Constant(object):
    def __init__(self, val):
        self.val = val

    def init(self, rng, shape):
        t = np.empty(shape, dtype=theano.config.floatX)
        t[...] = self.val
        return t

class Uniform(object):
    def __init__(self, mean, width=None, std=None):
        if (width is not None) == (std is not None):
            raise ValueError("must specify width or std, "
                             "but not both")
        if std is not None:
            # Variance of a uniform is 1/12 * width^2
            self.width = np.sqrt(12) * std
        else:
            self.width = width
        self.mean = mean

    def init(self, rng, shape):
        w = self.width / 2.0
        return rng.uniform(self.mean - w, self.mean + w, size=shape).astype(theano.config.floatX)

class IsotropicGaussian(object):

    def __init__(self, std=1, mean=0):
        self.std = std
        self.mean = mean

    def init(self, rng, shape):
        return rng.normal(self.mean, self.std, size=shape).astype(theano.config.floatX)


class GlorotNormal(object):

    def init(self, rng, shape):
        fan_in, fan_out = get_fans(shape)
        s = np.sqrt(2./(fan_in+fan_out))
        return rng.normal(0, s, size=shape).astype(theano.config.floatX)


class GlorotUniform(object):

    def init(self, rng, shape):
        fan_in, fan_out = get_fans(shape)
        s = np.sqrt(6./(fan_in+fan_out))
        return rng.uniform(-s, s, size=shape).astype(theano.config.floatX)


class HeNormal(object):

    def init(self, rng, shape):
        fan_in, fan_out = get_fans(shape)
        s = np.sqrt(2./fan_in)
        return rng.normal(0, s, size=shape).astype(theano.config.floatX)


class HeUniform(object):

    def init(self, rng, shape):
        fan_in, fan_out = get_fans(shape)
        s = np.sqrt(6./fan_in)
        return rng.uniform(-s, s, size=shape).astype(theano.config.floatX)
