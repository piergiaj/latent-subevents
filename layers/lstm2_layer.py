import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.sharedvar import TensorSharedVariable
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
import numpy as np

import activations as act
from hidden_layer import HiddenLayer
from recurrent_layer import RecurrentLayer

srng = RandomStreams()
np_rng = np.random.RandomState()


class LSTMLayer(RecurrentLayer):
    """
    Activation should be set to Tanh usually
    """

    def init_params(self):
        w_state = self.initializers[0].init(np_rng, (self.hidden_size, 4*self.hidden_size))
        self.w_state = theano.shared(value=w_state, name='LSTM.w.State.'+self.name,
                                     borrow=True, target=self.device)

        w_c_t_i = self.initializers[0].init(np_rng, (self.hidden_size,))
        self.w_cell_to_in = theano.shared(value=w_c_t_i, name='LSTM.w.cell_to_in.'+self.name,
                                          borrow=True, target=self.device)

        w_c_t_f = self.initializers[0].init(np_rng, (self.hidden_size,))
        self.w_cell_to_forget = theano.shared(value=w_c_t_f,
                                              name='LSTM.w.cell_to_forget.'+self.name,
                                              borrow=True, target=self.device)

        w_c_t_o = self.initializers[0].init(np_rng, (self.hidden_size,))
        self.w_cell_to_out = theano.shared(value=w_c_t_o,
                                           name='LSTM.w.cell_to_out.'+self.name,
                                           borrow=True, target=self.device)
        


        if not self.init_hidden_random:
            h = np.asarray(np.zeros((self.hidden_size,)), 
                           dtype=theano.config.floatX)
            c = np.asarray(np.zeros((self.hidden_size,)), 
                           dtype=theano.config.floatX)
        else:
            h = self.initializers[0].init(np_rng, (self.hidden_size,))
            c = self.initializers[0].init(np_rng, (self.hidden_size,))
        self.hidden = theano.shared(value=h, name='LSTM.hidden.'+self.name, borrow=True, target=self.device)
        self.cells = theano.shared(value=c, name='LSTM.cells.'+self.name, borrow=True, target=self.device)


    def run(self, x, hidden, cells, dropout=None):
        def slice_last(x, no):
            return x[:, no*self.hidden_size:(no+1)*self.hidden_size]

        w_state = self.w_state
        if dropout is not None:
            # apply dropout as in Y. Gal's paper
            # input should be the dropout matrix
            # and it should stay constant for the whole loop
            # and x should have already had dropout applied
                        
            # w_state is (self.hidden_size, 4*self.hidden_size)
            # so we want val to be hidden_drop_1 x wstate_1, 2, 3, 4
            # val should be of shape batch x hidden*4
            # so we want to take hidden*drop_1 dot w_state_1
            # and concatenate 2,3,4 to it
            
            # unfortuantly, this cannot be done with 1 matrix multiply :(
            val0 = T.dot(hidden * slice_last(dropout, 0), slice_last(self.w_state, 0))
            val1 = T.dot(hidden * slice_last(dropout, 1), slice_last(self.w_state, 1))
            val2 = T.dot(hidden * slice_last(dropout, 2), slice_last(self.w_state, 2))
            val3 = T.dot(hidden * slice_last(dropout, 3), slice_last(self.w_state, 3))
            val = T.concatenate([val0,val1,val2,val3], axis=1) + x
            val /= (1-self.dropout)
        else:
            val = T.dot(hidden, self.w_state) + x

        in_gate = T.nnet.sigmoid(slice_last(val, 0) +
                                 cells * self.w_cell_to_in)
        forget_gate = T.nnet.sigmoid(slice_last(val, 1) +
                                     cells * self.w_cell_to_forget)
        next_cells = (forget_gate * cells +
                      in_gate * self.activation(slice_last(val, 2)))
        out_gate = T.nnet.sigmoid(slice_last(val, 3) +
                                  next_cells * self.w_cell_to_out)
        next_states = out_gate * self.activation(next_cells)

        return next_states, next_cells

    
    def get_initial_hidden(self, x=None):
        if x == None:
            return [T.repeat(self.hidden[None, :], self.batch_size, 0),
                    T.repeat(self.cells[None, :], self.batch_size, 0)]
        else:
            return [T.repeat(self.hidden[None, :], x.shape[0], 0),
                    T.repeat(self.cells[None, :], x.shape[0], 0)]

    @property
    def params(self):
        if not (type(self.w_state) is TensorSharedVariable or type(self.w_state) is CudaNdarraySharedVariable):
            self.w_state = theano.shared(value=self.w_state.astype(theano.config.floatX), name='LSTM.w.State.'+self.name,
                                         borrow=True, target=self.device)

            self.w_cell_to_in = theano.shared(value=self.w_cell_to_in.astype(theano.config.floatX), name='LSTM.w.cell_to_in.'+self.name,
                                              borrow=True, target=self.device)

            self.w_cell_to_forget = theano.shared(value=self.w_cell_to_forget.astype(theano.config.floatX),
                                                  name='LSTM.w.cell_to_forget.'+self.name,
                                                  borrow=True, target=self.device)

            self.w_cell_to_out = theano.shared(value=self.w_cell_to_out.astype(theano.config.floatX),
                                               name='LSTM.w.cell_to_out.'+self.name,
                                               borrow=True, target=self.device)
        
            self.hidden = theano.shared(value=self.hidden.astype(theano.config.floatX), name='LSTM.hidden.'+self.name, borrow=True, target=self.device)
            self.cells = theano.shared(value=self.cells.astype(theano.config.floatX), name='LSTM.cells.'+self.name, borrow=True, target=self.device)

        return [self.w_state, self.w_cell_to_in, self.w_cell_to_forget, self.w_cell_to_out] + \
            ([self.hidden, self.cells] if self.include_hidden_param else [])

    @params.setter
    def params(self, params):
        if self.w_state.name == 'LSTM.w.State.'+self.name:
            return # this sohuld be fixed at some point...
        self.w_state = theano.shared(value=params[0], name='LSTM.w.State.'+self.name,
                                     borrow=True, target=self.device)
        self.w_cell_to_in = theano.shared(value=params[1], name='LSTM.w.cell_to_in.'+self.name,
                                          borrow=True, target=self.device)
        self.w_cell_to_forget = theano.shared(value=params[2],
                                              name='LSTM.w.cell_to_forget.'+self.name,
                                              borrow=True, target=self.device)
        self.w_cell_to_out = theano.shared(value=params[3],
                                           name='LSTM.w.cell_to_out.'+self.name,
                                           borrow=True, target=self.device)
        if len(params) >= 6:
            self.hidden = theano.shared(params[4], name='LSTM.hidden.'+self.name, borrow=True, target=self.device)
            self.cells = theano.shared(params[5], name='LSTM.cells.'+self.name, borrow=True, target=self.device)

    def print_layer(self):
        v = '--------------------\n'
        v += 'LSTM Layer '+self.name+'\n'
        v += 'Input Shape: '+str(self.input_size)+'\n'
        return v + 'Hidden Size: '+str(self.hidden_size)+'\n'

