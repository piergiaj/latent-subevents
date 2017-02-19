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
from layers.temporal_attention_lstm import TemporalAttentionLayer
from layers.lstm2_layer import LSTMLayer
from layers.hidden_layer import HiddenLayer
from layers.softmax_layer import SoftmaxLayer
from initialize import IsotropicGaussian as init

np.random.seed(np.random.randint(1 << 30))
rng = RandomStreams(seed=np.random.randint(1 << 30))
np_rng = np.random.RandomState()

def zeros(shape):
    return np.zeros(shape).astype(theano.config.floatX)


class TemporalModel(Model):
    def __init__(self, inputs, bs, max_time, classes, feature_dim, hidden_size, filters, N=1, pool=None, lstm_dim=4096, steps=8, seed=12345):
        self._inputs = inputs
        self.N = N
        self.batch_size = bs
        self.classes = classes
        self.max_time = max_time
        self.filters = filters
        self.feature_dim = feature_dim
        self.pool = pool
        self.dropout = True
        self.steps = steps

        self.temporal_filters = []
        for f in range(filters):
            tf = TemporalAttentionLayer(batch_size=bs, N=N, channels=feature_dim, 
                                        input_hidden_size=lstm_dim,
                                        name='temporal-attention-layer-filter-'+str(f))
            self.temporal_filters.append(tf)

        input_size = feature_dim*len(self.temporal_filters)*(N if pool == None else 1)

        self.lstm_in = HiddenLayer(input_size=input_size, hidden_size=lstm_dim*4, batch_size=bs)
        self.lstm = LSTMLayer(input_size=lstm_dim, hidden_size=lstm_dim)

        self.hidden = HiddenLayer(input_size=lstm_dim, hidden_size=hidden_size, activation=act.relu,
                                  batch_size=bs, name='hidden', dropout=0.5)
        self.softmax = SoftmaxLayer(input_size=hidden_size, classes=self.classes,
                                    batch_size=bs, name='softmax', dropout=0.5)


    @property
    def params(self):
        return self.softmax.params+self.hidden.params+self.lstm_in.params+self.lstm.params+[p for f in self.temporal_filters for p in f.params]
        
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

        # make x to be batch x features x time
        x = x.transpose([0,2,1])

        h,c = self.lstm.get_initial_hidden(x)

        outputs_info = [dict(initial=h, taps=[-1]), # h
                        dict(initial=c, taps=[-1])] # c

        [h,c], _ = theano.scan(fn=self.step,
                               non_sequences=[x, mask],
                               outputs_info=outputs_info,
                               n_steps=self.steps)

        x = self.hidden.run(h[-1], self.dropout)
        prob, pred = self.softmax.run(x, self.dropout)
        loss = self.softmax.loss(prob, y)
        error = self.softmax.error(pred, y)
        acc = 1-error

        return prob, pred, loss, error, acc


    def step(self, h, c, x, mask):
        results = []
        for tf in self.temporal_filters:
            # results is batch x features x N
            # flatten to batch x features*N
            res, (g,s2,d) = tf.run(x, h, mask)
            if self.pool == None:
                results.append(res.reshape((x.shape[0], self.feature_dim*self.N)))
            elif self.pool == 'max':
                results.append(T.max(res, axis=2).reshape((x.shape[0], self.feature_dim)))
            elif self.pool == 'sum':
                results.append(T.sum(res, axis=2).reshape((x.shape[0], self.feature_dim)))
            elif self.pool == 'mean':
                results.append(T.mean(res, axis=2).reshape((x.shape[0], self.feature_dim)))

        # concatenate on axis 1 to get batch x features*N*filters
        x = T.concatenate(results, axis=1)
        x = self.lstm_in.run(x)
        h, c = self.lstm.run(x, h, c)
        
        return h, c

