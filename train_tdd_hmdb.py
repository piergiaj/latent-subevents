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
    report = file('report-hmdb-tdd.txt', 'w')
    max_time = 200
    configs = []
    cc = create_config
    for d in ['1','2','3']:
        configs.append(cc('tdd-max-pool-h-4000 '+d, max_time, 4000, 'hmdb-tdd-1.hdf5', {'method':'max', 'hidden_size':4000}, 'hidden_2_layer_model',   0.0001))
        configs.append(cc('tdd-mean-pool-h-4000 '+d, max_time, 4000, 'hmdb-tdd.hdf5', {'method':'mean', 'hidden_size':4000}, 'hidden_2_layer_model', 0.0001))
        configs.append(cc('tdd-sum-pool-h-4000 '+d, max_time, 4000, 'hmdb-tdd.hdf5', {'method':'sum', 'hidden_size':4000}, 'hidden_2_layer_model',   0.0005))
        
        configs.append(cc('tdd-spyramid-1-h-1000', max_time, 4000, 'hmdb-tdd.hdf5', {'levels':1, 'hidden_size':1000}, 'temporal_pyramid_model'))
        configs.append(cc('tdd-spyramid-4-h-4000 '+d, max_time, 4000, '/ssd2/hmdb/hmdb-tdd.hdf5', {'levels':4, 'hidden_size':4000}, 'temporal_pyramid_model', 0.0001))


    for d in ['1','2','3']:
        for model in ['temporal_learned_model']:
            s = s+' split='+d
            for num_f in [3]:
                configs.append(cc('tdd-pyramid-1-N-'+str(num_f)+'-h-1000'+s, max_time, 4000, 'hmdb-tdd.hdf5', {'levels':1, 'hidden_size':1000, 'N':num_f}, model, 0.05))


    for config in configs:
        name = config['name']
        epochs = 250
        subdir = name + "-" + time.strftime("%Y%m%d-%H%M%S")
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
            
        bs = 100#int(sys.argv[1])
        max_time = config['max_time']#int(sys.argv[2])
        feature_dim = config['feature_dim']#int(sys.argv[3])

        from uniform_dataset import UniformDataset
        data_train = UniformDataset(bs=bs, filename=config['filename'], which_sets=['train'], sources=['features','time_mask','labels'])
        data_test = UniformDataset(bs=bs, filename=config['filename'], which_sets=['test'], sources=['features','time_mask','labels'])

        train_stream = DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, bs))
        test_stream = DataStream.default_stream(data_test, iteration_scheme=SequentialScheme(data_test.num_examples, bs))

        x = T.tensor3('features')
        time_mask = T.wmatrix('time_mask')
        y = T.imatrix('labels')
        
        mod = importlib.import_module(config['model'])
        classes = 51
        model = mod.TemporalModel([x,time_mask,y], bs, max_time, classes, feature_dim, **config['model_kwargs'])
        
        prob, pred, loss, error, acc = model.run(x,time_mask,y)
        prob.name = 'prob'
        acc.name = 'acc'
        pred.name = 'pred'
        loss.name = 'loss'
        error.name = 'error'
        
        model._outputs = [prob, pred, loss, error, acc]
        
        params = model.params
        
#        from solvers.sgd import SGD as solver
        from solvers.RMSProp import RMSProp as solver
        updates = solver(loss, params, lr=config['lr'], clipnorm=10.0)
        for i,u in enumerate(updates):
            if u[0].name == 'g' or u[0].name == 'sigma' or u[0].name == 'd':
                updates[i] = (u[0], T.mean(u[1]).dimshuffle(['x']))

        model._updates = updates
        
        # ============= TRAIN =========
        plots = [['train_loss','test_loss'],['train_acc','test_acc']]
        main_loop = MainLoop(model, train_stream,
                             [FinishAfter(epochs),
                              Track(variables=['loss','error','acc'], prefix='train'),
                              DataStreamTrack(test_stream, ['loss','error','acc'], prefix='test', best_method=[min,min,max]),
                              #SaveModel(subdir, name+'.model'),
                              TimeProfile(),
                              Report(os.path.join(subdir, 'report.txt'), name=name),
                              Printing()])
        main_loop.run()
        config['best_acc'] = main_loop.log.current_row['best_test_acc']
        print >> report, config['name'], 'best test acc', config['best_acc']
        report.flush()

    print ''.join(79 * '-')
    print 'FINAL REPORT'
    print ''.join(79 * '-')

    for config in configs:
        print config['name'], 'best test acc',config['best_acc']
        

if __name__ == '__main__':
    #theano.config.optimizer='fast_compile'
    run()
