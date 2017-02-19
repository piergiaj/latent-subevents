import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.sharedvar import TensorSharedVariable
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
import numpy as np
import custom_pickle as pkl

import activations as act
import dropout as drp
import initialize as i
import logging
log = logging.getLogger(__name__)

srng = RandomStreams()
np_rng = np.random.RandomState()


class HiddenLayer(object):

    def __init__(self, input_size=0, hidden_size=0, batch_size=1, activation=act.Identity, name='',
                 initializers=[i.IsotropicGaussian(0.01), i.Constant(0.)], dropout=0.0, device='gpu'):
        """
        Standard hidden layer.
        """
            
        self.activation = activation
        if type(activation) is str:
            if activation == 'sigmoid':
                self.activation = T.nnet.sigmoid
        self.hidden_size = hidden_size
        self.input_size = input_size
        if isinstance(input_size, list):
            self.input_size = reduce(lambda x, y: x*y, self.input_size)
        self.batch_size = batch_size
        self.name = name
        self.initializers = initializers
        self.dropout = dropout
        self.device = device
        srng.seed(np.random.randint(0,(2**30)))

        self.init_params()

    def load_pretrained(self, vals, i):
        self.w = theano.shared(value=vals[i], name='Hidden.w.'+self.name, borrow=True, target=self.device)
        self.b = theano.shared(value=vals[i+1], name='Hidden.b.'+self.name, borrow=True, target=self.device)
        return i+2

    def init_params(self):
        w = self.initializers[0].init(np_rng, (self.input_size, self.hidden_size))
        self.w = theano.shared(value=w, name='Hidden.w.'+self.name, borrow=True, target=self.device)
        b = self.initializers[1].init(np_rng, (self.hidden_size,))
        self.b = theano.shared(value=b, name='Hidden.b.'+self.name, borrow=True, target=self.device)


    def run(self, x, dropout=True):
        if x.ndim > 2:
            # x isn't a matrix, make it one.
            x = x.flatten(2)
        #d = self.dropout
        if not hasattr(self, 'dropout'):
            d = 0
        else:
            d = self.dropout

        if not dropout:
            d = 0
        out = self.activation(T.dot(x, self.w) + self.b)
        return drp.dropout(srng, out, d, (out.shape[0], self.hidden_size))

    @property
    def params(self):
        if not (type(self.w) is TensorSharedVariable or type(self.w) is CudaNdarraySharedVariable):
            self.w = theano.shared(value=self.w.astype(theano.config.floatX), name='Hidden.w.'+self.name, borrow=True, target=self.device)
            self.b = theano.shared(value=self.b.astype(theano.config.floatX), name='Hidden.b.'+self.name, borrow=True, target=self.device)

        return [self.w, self.b]

    @property
    def params_val(self):
        return [self.w.get_value(borrow=True), self.b.get_value(borrow=True)]

    def set_params(self, params):
#        if params[0].shape[0] != self.input_size:
#            log.error('Input weights shape input size mismatch '+str(params[0].shape[0])+
#                      ' expected '+str(self.input_size))
#            return
#        if params[1].shape[0] != self.hidden_size:
#            log.error('Input bias shape mismatch '+str(params[1].shape[0])+
#                      ' expected '+str(self.hidden_size))
#            return
#        if params[0].shape[1] != self.hidden_size:
#            log.error('Input weights shape hidden mismatch '+str(params[0].shape[1])+
#                      ' expected '+str(self.hidden_size))
#            return
#        print self.w.get_value(borrow=True), params[0].get_value(borrow=True)
        print 'setting params'
        self.w = theano.shared(params[0], 'Hidden.w.'+self.name, borrow=True)
        self.b = theano.shared(params[1], 'Hidden.b.'+self.name, borrow=True)
        return

    @params.setter
    def params(self, params):
        self.set_params(params)


    def __getstate__(self):
        return {k: pkl.pickle(v) for k,v in self.__dict__.iteritems()}

    def __setstate__(self, state):
        # super(HiddenLayer, self).__setstate__(state) (doesn't exist...)
        self.__dict__ = state
        self.params = self.params # nifty trick to get the params as shared variables again


    def print_layer(self):
        v = '--------------------\n'
        v += 'Hidden Layer '+self.name+'\n'
        v += 'Input Size: '+str(self.input_size)+'\n'
        v += 'Hidden Size: '+str(self.hidden_size)+'\n'
        return v
