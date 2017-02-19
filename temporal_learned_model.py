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
from layers.temporal_attention import TemporalAttentionLayer
from layers.hidden_layer import HiddenLayer
from layers.softmax_layer import SoftmaxLayer
from initialize import IsotropicGaussian as init

np.random.seed(np.random.randint(1 << 30))
rng = RandomStreams(seed=np.random.randint(1 << 30))
np_rng = np.random.RandomState()

def zeros(shape):
    return np.zeros(shape).astype(theano.config.floatX)


class TemporalModel(Model):
    def __init__(self, inputs, bs, max_time, classes, feature_dim, hidden_size, levels, N=1, pool=None, seed=12345):
        self._inputs = inputs
        self.N = N
        self.batch_size = bs
        self.classes = classes
        self.max_time = max_time
        self.levels = levels
        self.feature_dim = feature_dim
        self.pool = pool
        self.dropout = True

        self.temporal_pyramid = []
        for l in range(self.levels):
            for f in range(2**l):
                tf = TemporalAttentionLayer(batch_size=bs, N=N, channels=feature_dim, 
                                            name='temporal-attention-layer-'+str(l)+'-filter-'+str(f))
                tf.test = True
                tf.d = theano.shared(value=np.asarray([1./2**(l+1)]).astype('float32'), name='d', borrow=True,
                                     broadcastable=[True])
                tf.g = theano.shared(value=np.asarray([((1./2**l)+(2*f/2.**l))]).astype('float32'), name='g', borrow=True,
                                     broadcastable=[True])
                tf.sigma = theano.shared(value=np.asarray([5.0]).astype('float32'), name='sigma', borrow=True,
                                         broadcastable=[True])
                self.temporal_pyramid.append(tf)

        input_size = feature_dim*N*(len(self.temporal_pyramid) if pool == None else 1)
        self.hidden = HiddenLayer(input_size=input_size, hidden_size=hidden_size, activation=act.LeakyRelu(),
                                  batch_size=bs, name='hidden', dropout=0.5)
        self.softmax = SoftmaxLayer(input_size=hidden_size, classes=self.classes,
                                    batch_size=bs, name='softmax', dropout=0.5)


    @property
    def params(self):
        return self.softmax.params+self.hidden.params+[p for f in self.temporal_pyramid for p in f.params]

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
        # use temporal filters
        results = []
        # make x to be batch x features x time
        x = x.transpose([0,2,1])
        for tf in self.temporal_pyramid:
            # results is batch x features x N
            # flatten to batch x features*N
            res, (g,s2,d) = tf.run(x, mask)
            if self.pool == None:
                results.append(res.reshape((x.shape[0], self.feature_dim*self.N)))
            else:
                results.append(res.reshape((x.shape[0], 1, self.feature_dim*self.N)))
        # concatenate on axis 1 to get batch x features*N*filters
        x = T.concatenate(results, axis=1)

        if self.pool == 'max':
            x = T.max(x, axis=1)
        elif self.pool == 'sum':
            x = T.sum(x, axis=1)
        elif self.pool == 'mean':
            x = T.mean(x, axis=1)

        x = self.hidden.run(x, self.dropout)
        prob, pred = self.softmax.run(x, self.dropout)
        loss = self.softmax.loss(prob, y)
        error = self.softmax.error(pred, y)
        acc = 1-error

        return prob, pred, loss, error, acc
