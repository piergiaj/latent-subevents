import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.sharedvar import TensorSharedVariable
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
import numpy as np

import activations as act
import dropout as drp
import initialize as ini

srng = RandomStreams()
np_rng = np.random.RandomState()

def get_random(size):
    return (np_rng.standard_normal(size) * 1. / size[0]).astype(theano.config.floatX)


class SoftmaxLayer(object):

    def __init__(self, input_size=0, classes=0, batch_size=1, activation=act.Identity, name='',
                 initializers=[ini.IsotropicGaussian(0.01), ini.Constant(0.)], dropout=0, device='gpu'):
        """
        Standard hidden layer.
        """
            
        self.activation = activation
        self.classes = classes
        self.input_size = input_size
        self.batch_size = batch_size
        self.name = name
        self.initializers = initializers
        self.dropout = dropout
        self.device = device
        srng.seed(np.random.randint(0,(2**30)))

        self.init_params()

    def load_pretrained(self, vals, i):
        self.w = theano.shared(value=vals[i], name='Softmax.w.'+self.name, borrow=True, target=self.device)
        self.b = theano.shared(value=vals[i+1], name='Softmax.b.'+self.name, borrow=True, target=self.device)
        return i+2

    def init_params(self):
        w = self.initializers[0].init(np_rng, (self.input_size, self.classes))
        self.w = theano.shared(value=w, name='Softmax.w.'+self.name, borrow=True, target=self.device)
        b = self.initializers[1].init(np_rng, (self.classes,))
        self.b = theano.shared(value=b, name='Softmax.b.'+self.name, borrow=True, target=self.device)


    def run(self, x, dropout=True):
        if x.ndim > 2:
            x = x.flatten(2)
        if not hasattr(self, 'dropout'):
            d = 0
        else:
            d = self.dropout

        if not dropout:
            d = 0
        out = self.activation(T.dot(x, self.w) + self.b)
        out = drp.dropout(srng, out, d, (out.shape[0], self.classes))
        prob = T.nnet.softmax(out)
        pred = T.argmax(prob, axis=1)
        return prob, pred

    def loss(self, prob, y):
        return T.nnet.categorical_crossentropy(prob,y).mean()#-T.mean(T.log(prob)[y])

    def error(self, pred, y):
        return T.mean(T.neq(pred, T.argmax(y,axis=1)), dtype=theano.config.floatX, acc_dtype=theano.config.floatX)
        
    def all_errors(self, pred, y):
        return T.neq(pred, T.argmax(y,axis=1))

    @property
    def params(self):
        if not (type(self.w) is TensorSharedVariable or type(self.w) is CudaNdarraySharedVariable):
            self.w = theano.shared(value=self.w.astype(theano.config.floatX), name='Softmax.w.'+self.name, borrow=True, target=self.device)
            self.b = theano.shared(value=self.b.astype(theano.config.floatX), name='Softmax.b.'+self.name, borrow=True, target=self.device)

        return [self.w, self.b]

    @params.setter
    def params(self, params):
        self.w.set_value(params[0])#.get_value())
        self.b.set_value(params[1])#.get_value())

    def print_layer(self):
        v = '--------------------\n'
        v += 'Softmax Layer '+self.name+'\n'
        v += 'Input Shape: '+str(self.input_size)+'\n'
        return v + 'Classes: '+str(self.classes)+'\n'



if __name__ == '__main__':
    from solvers import RMSProp as sol

    x = T.matrix()
    y = T.vector(dtype='int64')

    s = SoftmaxLayer(10, 2, 5)

    prob, pred = s.run(x)
    loss = s.loss(prob, y)

    updates = sol.RMSProp(cost=loss, params=s.params)

    f = theano.function([x,y], outputs=[loss,pred], updates=updates)
    trn = np.random.random((5,10))
    lb = [0,1,0,1,0]
    for x in range(1000):
        print f(trn,lb)
