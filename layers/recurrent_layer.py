import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

import activations as act
from hidden_layer import HiddenLayer

srng = RandomStreams()
np_rng = np.random.RandomState()


class RecurrentLayer(HiddenLayer):

    def __init__(self, init_hidden_random=False, include_hidden_param=True, *args, **kwargs):
        self.init_hidden_random = init_hidden_random
        self.include_hidden_param = include_hidden_param
        super(RecurrentLayer, self).__init__(*args, **kwargs)

    def init_params(self):
        w = self.initializers[0].init(np_rng, (self.input_size+self.hidden_size, self.hidden_size))
        self.w = theano.shared(value=w, name='RNN.w.'+self.name, borrow=True, target=self.device)
        b = self.initializers[1].init(np_rng, (self.hidden_size,))
        self.b = theano.shared(value=b, name='RNN.b.'+self.name, borrow=True, target=self.device)
        if not self.init_hidden_random:
            h = np.asarray(np.zeros((1,self.hidden_size)), 
                           dtype=theano.config.floatX)
        else:
            h = self.initializers[0].init(np_rng, (1,self.hidden_size))
        self.hidden = theano.shared(value=h, name='RNN.hidden.'+self.name, borrow=True, target=self.device)


    def run(self, x, h):
        return self.activation(T.dot(T.concatenate([x,h], axis=1), self.w) + self.b)

    @property
    def get_initial_hidden(self):
        return self.hidden.repeat(self.batch_size, axis=0)

    @property
    def params(self):
        return [self.w, self.b] + ([self.hidden] if self.include_hidden_param else [])

    @property
    def params_val(self):
        return [self.w.get_value(borrow=True), self.b.get_value(borrow=True)] + \
            ([self.hidden.get_value(borrow=True)] if self.include_hidden_param else [])

    @params.setter
    def params(self, params):
        super(RecurrentLayer, self).set_params(params[0:1])
        if len(params) >= 3:
            self.hidden = theano.shared(params[2], name='RNN.hidden.'+self.name, borrow=True, target=self.device)

    def __setstate__(self, state):
        tmp = state['include_hidden_param']
        state['include_hidden_param'] = True
        super(RecurrentLayer, self).__setstate__(state)
        self.include_hidden_param = tmp
