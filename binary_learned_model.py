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
    def __init__(self, inputs, bs, max_time, class_num, feature_dim, hidden_size, levels, N=1, pool=None, seed=12345):
        self._inputs = inputs
        self.N = N
        self.batch_size = bs
        self.name = 'learned_'+str(class_num)
        self.class_num = theano.shared(class_num)
        self.max_time = max_time
        self.filters = levels
        self.feature_dim = feature_dim
        self.pool = pool
        self.dropout = True

        self.temporal_pyramid = []
        for f in range(self.filters):
            tf = TemporalAttentionLayer(batch_size=bs, N=N, channels=feature_dim, 
                                        name='af-'+str(f))
            self.temporal_pyramid.append(tf)

        input_size = feature_dim*len(self.temporal_pyramid)#*N
        self.hidden = HiddenLayer(input_size=input_size, hidden_size=hidden_size, activation=act.LeakyRelu(),
                                  batch_size=bs, name='hidden', dropout=0.5)
        self.classify = HiddenLayer(input_size=hidden_size, hidden_size=1,
                                    batch_size=bs, name='classify', dropout=0.0, activation=act.sigmoid)


    @property
    def params(self):
        return self.classify.params+self.hidden.params+[p for f in self.temporal_pyramid for p in f.params]

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
        # use temporal filters
        results = []
        # make x to be batch x features x time
        x = x.transpose([0,2,1])
        for tf in self.temporal_pyramid:
            # results is batch x features x N
            # flatten to batch x features*N
            res, (g,s2,d) = tf.run(x, mask)
            res = res.reshape((x.shape[0], self.feature_dim, self.N))
            results.append(T.mean(res, axis=2)) # take mean along N to get feature x 1 representation for sub-event
#            if self.pool == None:
#                results.append(res.reshape((x.shape[0], self.feature_dim*self.N)))
#            else:
#                results.append(res.reshape((x.shape[0], 1, self.feature_dim*self.N)))
        # concatenate on axis 1 to get batch x features*N*filters
        x = T.concatenate(results, axis=1)

        if self.pool == 'max':
            x = T.max(x, axis=1)
        elif self.pool == 'sum':
            x = T.sum(x, axis=1)
        elif self.pool == 'mean':
            x = T.mean(x, axis=1)

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
