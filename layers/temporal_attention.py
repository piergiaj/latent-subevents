from __future__ import division
import theano
from theano import tensor as T
from theano.tensor.sharedvar import TensorSharedVariable
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import initialize as i

np_rng = np.random.RandomState()

class TemporalAttentionLayer(object):

    def __init__(self, batch_size, N, channels, name='', use_gpu=True, test=False,
                 initializers=[i.IsotropicGaussian(mean=1., std=0.1), # g
                               i.IsotropicGaussian(mean=0.5, std=0.1), # d
                               i.IsotropicGaussian(mean=1., std=0.1)]): # sigma
        """
        Temporal Read Layer Based on DRAW paper
        """

        self.batch_size = batch_size
        self.N = N
        self.channels = channels
        self.name = name
        self.output_shape = [batch_size, channels, N]
        self.use_gpu = use_gpu
        self.initializers = initializers
        self.test = test

        self.init_params()

    def load_pretrained(self, v, i):
        return i

    def init_params(self):
        g = self.initializers[0].init(np_rng, (1,))
        self.g = theano.shared(value=g, name='g.'+self.name, borrow=True,
                               broadcastable=[True])

        d = self.initializers[1].init(np_rng, (1,))
        self.d = theano.shared(value=d, name='d.'+self.name, borrow=True,
                               broadcastable=[True])

        sigma = self.initializers[1].init(np_rng, (1,))
        self.sigma = theano.shared(value=sigma, name='sigma.'+self.name, borrow=True)


    def batched_dot(self, A, B):
        if self.use_gpu:
            return theano.sandbox.cuda.blas.batched_dot(A, B)
        else:
            return T.batched_dot(A,B)
#        C = A.dimshuffle([0,1,2,'x']) * B.dimshuffle([0,'x',1,2])
#        return C.sum(axis=-2)

    def get_params(self, time):
        g = time*((self.g+1)*0.5)
        g = g.astype(theano.config.floatX)
        s2 = (self.sigma/2.0).repeat(time.shape[0], axis=0)
        s2 = s2.astype(theano.config.floatX)
        d = time / (max(self.N-1.0,1.0)) *  (self.d)
        d = d.astype(theano.config.floatX)
         
        return g,s2,d

    def get_params_test(self, time):
        return (time*self.g), (self.sigma).repeat(time.shape[0], axis=0), (time/(max(self.N-1.0,1.0)) * self.d).astype(theano.config.floatX)

    def run(self, features, time_mask):
        channels = self.channels
        # assumes that features are batch x dim (channels) x time
        # time mask is batch x time and is binary
        
        # time mask is a binary matrix that is 1 when the input
        # is a valid frame and 0 when the input is not. 
        # This allows the shape of a minibatch be the same
        # even though videos are of various lengths.

        # we sum along axis 1 to get the time length of
        # the individual clips
        time = T.sum(time_mask, axis=1)
        
        if not self.test:
            g,s2,d= self.get_params(time)
        else:
            g,s2,d = self.get_params_test(time)
            g = g.astype(theano.config.floatX)
            s2 = s2.astype(theano.config.floatX)
            d = d.astype(theano.config.floatX)

        I = features.reshape((features.shape[0]*self.channels, features.shape[2], 1))

        mu = g.dimshuffle([0,'x']) + d.dimshuffle([0,'x']) * \
             (T.arange(self.N).astype(theano.config.floatX) - self.N/2. - 0.5)

        a = T.arange(features.shape[2]).astype(theano.config.floatX)

        # I is batch*channels x time x 1
        # F is batch[*channels] x N x time
        # batch*channels x N x 1
        F = T.exp(-(a-mu.dimshuffle([0,1,'x']))**2 / 2. / s2.dimshuffle([0,'x','x'])**2)
        # need to mask F
        F = F * time_mask.dimshuffle([0,'x',1])
        # normalize F
        F = F / (F.sum(axis=-1).dimshuffle([0,1,'x']) + 1e-4)
        F = T.repeat(F, channels, axis=0)

        res = self.batched_dot(F, I).reshape((features.shape[0], self.channels, self.N))

        return res, (g,s2,d)

    @property
    def params(self):
        if not (type(self.g) is TensorSharedVariable or type(self.g) is CudaNdarraySharedVariable):
            self.g = theano.shared(value=self.g.astype(theano.config.floatX), name='g.'+self.name, borrow=True,
                               broadcastable=[True])

            self.d = theano.shared(value=self.d.astype(theano.config.floatX), name='d.'+self.name, borrow=True,
                                   broadcastable=[True])

            self.sigma = theano.shared(value=self.sigma.astype(theano.config.floatX), name='sigma.'+self.name, borrow=True)
            
        return [self.g, self.sigma, self.d]
            

    @params.setter
    def params(self, params):
        g = params[0]
        self.g = theano.shared(value=g, name='g.'+self.name, borrow=True,
                               broadcastable=[True])

        d = params[1]
        self.d = theano.shared(value=d, name='d.'+self.name, borrow=True,
                               broadcastable=[True])
        sigma = params[2]
        self.sigma = theano.shared(value=sigma, name='sigma.'+self.name, borrow=True)


    def print_layer(self):
        v = '--------------------\n'
        v += 'Read Layer '+self.name+'\n'
        v += 'Input Shape: '+str((self.width, self.height))+'\n'
        return v + 'Output Shape: '+str((self.N, self.N))+'\n'


if __name__ == '__main__':
    # testing
    theano.config.optimizer='fast_compile'
    attn = TemporalAttentionLayer(batch_size=10, N=5, channels=6, use_gpu=False)

    time_mask = T.imatrix('time_mask')
    features = T.tensor3('features')

    res, (g, s2, d) = attn.run(features, time_mask)
    
    f = theano.function([features, time_mask], [res, g, s2, d], on_unused_input='warn')

    fts = np.random.random((10,6,12))
    tm = np.ones((10,12))
    tm[0,6:] = 0
    tm[1,4:] = 0
    tm[2,2:] = 0
    tm[3,8:] = 0
    tm[4,9:] = 0
    tm[5,1:] = 0
    tm[6,3:] = 0
    tm[7,5:] = 0
    tm = tm.astype('int32')

    print f(fts, tm)
