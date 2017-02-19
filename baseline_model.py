# ========= STD Libs  ============
from __future__ import division
import logging

# ========= Theano/npy ===========
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

# ========= Tools  ==============
from model import Model

# ========= Layers  ==============
import layers.activations as act
from layers.dropout import dropout
from layers.hidden_layer import HiddenLayer
from layers.softmax_layer import SoftmaxLayer
from initialize import IsotropicGaussian as init

np.random.seed(np.random.randint(1 << 30))
rng = RandomStreams(seed=np.random.randint(1 << 30))
np_rng = np.random.RandomState()

def zeros(shape):
    return np.zeros(shape).astype(theano.config.floatX)


class TemporalModel(Model):
    def __init__(self, inputs, bs, max_time, classes, feature_dim, hidden_size, method='max', seed=12345):
        self._inputs = inputs
        self.method = method
        self.batch_size = bs
        self.classes = classes
        self.max_time = max_time
        self.feature_dim = feature_dim
        self.dropout = True
        self.hidden = HiddenLayer(input_size=feature_dim, hidden_size=hidden_size,
                                  batch_size=bs, name='hidden', dropout=0.5, activation=act.LeakyRelu())
        self.softmax = SoftmaxLayer(input_size=hidden_size, classes=self.classes, 
                                    batch_size=bs, name='softmax', dropout=0.5)


    @property
    def params(self):
        return self.softmax.params+self.hidden.params

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs
    
    @property
    def updates(self):
        return self._updates

    @property
    def test_algorithm(self):
        if not hasattr(self, '_talgorithm'):
            d = self.dropout
            self.dropout = False
            o = self.run(*self.inputs)
            for i,ot in enumerate(self.outputs):
                o[i].name = ot.name
            self._talgorithm = theano.function(inputs=self.inputs,
                                               outputs=o, on_unused_input='warn')
            self.dropout = d
        return self._talgorithm



    def run(self, x, mask, y):
        # get the max/mean/sum of x for each feature
        # from all frame
        if self.method == 'max':
            m = (-100*(1 - mask)).dimshuffle([0,1,'x'])
            x = T.max(x+m, axis=1)
        elif self.method == 'sum' or self.method == 'mean':
            x = T.sum(x, axis=1)
        elif self.method == 'mean':
            x = x/T.sum(mask, axis=1).dimshuffle([0,'x'])

        x = x.astype(theano.config.floatX)
        x = self.hidden.run(x, self.dropout)

        prob, pred = self.softmax.run(x, self.dropout)
        y = y.reshape((y.shape[0],))
        loss = self.softmax.loss(prob, y) + T.sum(self.hidden.w**2)*0.001 + T.sum(self.softmax.w**2)*0.0001
        y = T.extra_ops.to_one_hot(y, 51)
        error = self.softmax.error(pred, y)
        acc = 1-error

        return prob, pred, loss, error, acc
