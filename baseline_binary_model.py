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
    def __init__(self, inputs, bs, max_time, class_num, feature_dim, hidden_size, method='max', seed=12345):
        self._inputs = inputs
        self.method = method
        self.name = 'baseline_'+str(class_num)
        self.batch_size = bs
        self.class_num = theano.shared(class_num)
        self.max_time = max_time
        self.feature_dim = feature_dim
        self.dropout = True
        self.hidden = HiddenLayer(input_size=feature_dim, hidden_size=hidden_size,
                                  batch_size=bs, name='hidden', dropout=0.5, activation=act.LeakyRelu())

        self.classify = HiddenLayer(input_size=hidden_size, hidden_size=1,
                                    batch_size=bs, name='classify', dropout=0.0, activation=act.sigmoid)


    @property
    def params(self):
        return self.classify.params+self.hidden.params

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs
    
    @property
    def updates(self):
        return self._updates


    def run(self, x, mask, y):
        # get the max/mean/sum of x for each feature
        # from all frame
        if self.method == 'max':
            # apply the mask so that if the feature is all negative
            # the 0s don't affect it
            m = (-100*(1 - mask)).dimshuffle([0,1,'x'])
            x = T.max(x+m, axis=1)
        elif self.method == 'sum' or self.method == 'mean':
            x = T.sum(x, axis=1)
        if self.method == 'mean':
            # divide by the number of valid frames
            x = x/T.sum(mask, axis=1).dimshuffle([0,'x'])

        x = x.astype(theano.config.floatX)
        x = self.hidden.run(x, self.dropout)
        prob = self.classify.run(x, False)

        y = T.switch(T.eq(self.class_num.repeat(y.shape[0]).reshape((y.shape[0], 1)), y), 1, 0)
        preds = T.switch(T.gt(prob, 0.5), 1, 0)
        
        true_pos = (T.eq(y, 1) * T.eq(preds, 1)).sum() 
        true_neg = (T.neq(y, 1) * T.neq(preds, 1)).sum() 
        false_pos = (T.neq(y, 1) * T.eq(preds, 1)).sum() 
        false_neg = (T.eq(y, 1) * T.neq(preds, 1)).sum() 
        
        loss = T.nnet.binary_crossentropy(prob, y)
        loss = T.switch(T.eq(y,1), loss, 0.02*loss)
        loss = loss.mean()
        
        return prob, loss, (true_pos, true_neg, false_pos, false_neg)
