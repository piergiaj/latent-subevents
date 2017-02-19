# ========= STD Libs  ============
from __future__ import division
from collections import OrderedDict
import os
import shutil
import sys
import logging
#import ipdb
import cPickle
import argparse
import time
sys.setrecursionlimit(100000000)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= Theano/npy ===========
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import fuel
from fuel.streams import DataStream, ServerDataStream
from fuel.schemes import SequentialScheme

# ========= Tools  ==============
from main_loop import MainLoop
from extensions import Printing, FinishAfter, TrackBest, Track, Report
from extensions.timing import TimeProfile
from extensions.plot import PlotLocal, Plot
from extensions.data_monitor import DataStreamTrack
from extensions.save_model import SaveModel, SaveBestModel

import importlib

def create_config(name, max_time, feature_dim, filename, args, model, lr=0.0001):
    return {'name':name, 'max_time':max_time, 'feature_dim':feature_dim, 'filename':filename, 'model_kwargs':args, 'model':model, 'lr':lr}


def run():
    configs = [0]
    for config in configs:
        bs = 48
        feature_dim = 4000

        from uniform_dataset import UniformDataset
        data_test = UniformDataset(bs=bs, filename='/ssd2/hmdb/hmdb-tdd-1.hdf5', which_sets=['test'], sources=['features','time_mask','labels'])

        test_stream = DataStream.default_stream(data_test, iteration_scheme=SequentialScheme(data_test.num_examples, bs))
        
        x = T.tensor3('features')
        time_mask = T.wmatrix('time_mask')
        y = T.imatrix('labels')
        
        classes = eval(sys.argv[1])
        outputs = []
        for clas in classes:
            print 'Loading',clas
            model = cPickle.load(open('models/learned_'+str(clas), 'rb'))
            prob, loss, (tp,tn,fp,fn) = model.run(x,time_mask,y)
            prob.name = 'prob_'+str(clas)
        
            outputs += [prob]
        # prob is Nx1
        # outputs is 51xNx1
        # stack and take max along 51-class index
        outputs = T.stacklists(outputs)
        preds = T.argmax(outputs, axis=0)

        # predicted class is now outputs
        # which is shape Nx1, reshape to vector of N
        preds = preds.reshape((preds.shape[0],1))

        num_err = T.neq(preds, y).sum()
        acc = 1-(num_err/y.shape[0])

        test_func = theano.function([x,time_mask,y], outputs, on_unused_input='warn')

        data = test_stream.get_epoch_iterator(as_dict=True)
        total_acc = 0
        num = 0
        res = None
        labs = None
        for batch in data:
            o = test_func(batch['features'],batch['time_mask'],batch['labels'])
            if res is None:
                res = o
                labs = batch['labels']
            else:
                # append on axis 1, to get 51xDs_size
                res = np.append(res, o, axis=1)
                labs = np.append(labs, batch['labels'], axis=0)
            continue

            total_acc += acc
            num += 1
            print acc
        np.save('results'+sys.argv[2], res)
        np.save('labs'+sys.argv[2], labs)
        #print 'Mean acc:',total_acc/num
        
        

        

if __name__ == '__main__':
    #theano.config.optimizer='fast_compile'
    run()
