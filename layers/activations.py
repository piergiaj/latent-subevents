from theano import tensor as T

def Identity(x):
    return x

# Standard ReLU function max(x,0)
def relu(x):
    return T.maximum(x, 0.)

class Relu(object):
    def __init__(self, name='relu'):
        self.name = name

    def run(self, x):
        return T.maximum(x, 0.)

    def load_pretrained(self, x,i):
        return i

    @property
    def params(self):
        return []

    def print_layer(self):
        return 'Relu Layer\n'


# sigmoid function
def sigmoid(x):
    return T.nnet.sigmoid(x)



class LeakyRelu(object):

    def __init__(self, name='leaky-relu', leak=0.01):
        self.leak = leak
        self.name = name

    def run(self, x):
        self.shape = x.shape
        return T.switch(T.ge(x, 0), x, self.leak*x)

    def __call__(self, x):
        return self.run(x)
