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
from extensions import Printing, FinishAfter, TrackBest, Track, Report, Extension
from extensions.timing import TimeProfile
from extensions.plot import PlotLocal, Plot
from extensions.data_monitor import DataStreamTrack
from extensions.save_model import SaveModel, SaveBestModel

class SaveAfter(Extension):

    def __init__(self, models, **kwargs):
        kwargs.setdefault('after_epoch', True)
        super(SaveAfter, self).__init__(**kwargs)
        self.models = models
        self.best_f1 = [0]*len(self.models)

    def do(self, *args, **kwargs):
        for i,model in enumerate(self.models):
            recall = self.main_loop.log.current_row['test_recall_'+str(model.class_num.get_value())]
            prec = self.main_loop.log.current_row['test_prec_'+str(model.class_num.get_value())]
            f1 = (recall*prec)/(recall+prec)
            if f1 > self.best_f1[i]:
                self.best_f1[i] = f1
                f = file(os.path.join('models', model.name), 'wb')
                cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()

import importlib

def create_config(name, max_time, feature_dim, filename, args, model, lr=0.0001):
    return {'name':name, 'max_time':max_time, 'feature_dim':feature_dim, 'filename':filename, 'model_kwargs':args, 'model':model, 'lr':lr}


def run():
    report = file('report-hmdb-tdd-binary.txt', 'w')
    max_time = 200
    configs = []
    cc = create_config
    for d in ['1','2','3']:
        configs.append(cc('tdd-max-pool-h-4000 '+d, max_time, 4000, 'hmdb-tdd-1.hdf5', {'method':'max', 'hidden_size':4000}, 'baseline_binary_model',   0.01))
        configs.append(cc('tdd-mean-pool-h-4000 '+d, max_time, 4000, 'hmdb-tdd.hdf5', {'method':'mean', 'hidden_size':4000}, 'baseline_binary_model', 0.0001))
        configs.append(cc('tdd-sum-pool-h-4000 '+d, max_time, 4000, 'hmdb-tdd.hdf5', {'method':'sum', 'hidden_size':4000}, 'baseline_binary_model',   0.0005))
        
        #configs.append(cc('tdd-spyramid-1-h-1000', max_time, 4000, 'hmdb-tdd.hdf5', {'levels':1, 'hidden_size':1000}, 'temporal_pyramid_model'))
#        configs.append(cc('tdd-spyramid-4-h-4000 '+d, max_time, 4000, 'hmdb-tdd.hdf5', {'levels':3, 'hidden_size':4000}, 'temporal_pyramid_binary_model', 0.01))
#

    for d in ['1','2','3']:
        for model in ['binary_learned_model']:#, 'temporal_random_model']:
            s = s+' split='+d
            for num_f in [3]:
                configs.append(cc('plot-attention-', max_time, 4000, 'hmdb-tdd.hdf5', {'levels':6, 'hidden_size':4000, 'N':num_f}, model, 0.005))



    for config in configs:
        name = config['name']+sys.argv[1]
        epochs = 150
        subdir = name + "-" + time.strftime("%Y%m%d-%H%M%S")
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
            
        bs = 64#int(sys.argv[1])
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
        models = []
        b_model = None
        classes = eval(sys.argv[1])
        for clas in classes:
            model = mod.TemporalModel([x,time_mask,y], bs, max_time, clas, feature_dim, **config['model_kwargs'])
            models.append(model)
            if not b_model:
                b_model = model
                b_model._outputs = []
                b_model._updates = []

            prob, loss, (tp,tn,fp,fn) = model.run(x,time_mask,y)
            prob.name = 'prob_'+str(clas)
            loss.name = 'loss_'+str(clas)
            tp.name = 'tp_'+str(clas)
            tn.name = 'tn_'+str(clas)
            fp.name = 'fp_'+str(clas)
            fn.name = 'fn_'+str(clas)
        
            b_model._outputs += [prob, loss, tp, tn, fp, fn]

            #for filt in model.temporal_pyramid:
            #    print filt.g.name, filt.d.name, filt.sigma.name
            #    b_model._outputs += [filt.g, filt.d, filt.sigma]
        
            params = model.params
        
            #        from solvers.sgd import SGD as solver
            from solvers.RMSProp import RMSProp as solver
            updates = solver(loss, params, lr=config['lr'], clipnorm=10.0)
            for i,u in enumerate(updates):
                if u[0].name is None:
                    continue
                if 'g.' in u[0].name or 'shhigma.' in u[0].name or 'd.' in u[0].name:
                    updates[i] = (u[0], T.mean(u[1]).dimshuffle(['x']))

            b_model._updates += updates
        
        # ============= TRAIN =========
        tc = classes
        #plots = [['_plt_g.af-0','_plt_g.af-1','_plt_g.af-2'],['_plt_d.af-0','_plt_d.af-1','_plt_d.af-2'],['_plt_sigma.af-0','_plt_sigma.af-1','_plt_sigma.af-2']]
        #track_plot = [(x[5:],'last') for sl in plots for x in sl]
        var = [['loss_'+str(i),('tp_'+str(i),'sum'),('tn_'+str(i),'sum'),('fp_'+str(i),'sum'),('fn_'+str(i),'sum'),('recall_'+str(i),'after','tp_'+str(i),'fn_'+str(i),lambda x,y:x/(x+y)), ('prec_'+str(i),'after','tp_'+str(i),'fp_'+str(i),lambda x,y:x/(x+y))] for i in tc]
        var = [item for sublist in var for item in sublist]

        bm = [[min, max, max, min, min, max, max] for i in tc]
        bm = [item for sublist in bm for item in sublist]

        main_loop = MainLoop(b_model, train_stream,
                             [FinishAfter(epochs),
                              Track(variables=var, prefix='train'),
                              #Track(variables=track_plot, prefix='_plt'),
                              DataStreamTrack(test_stream, var, prefix='test', best_method=bm),
                              TimeProfile(),
                              SaveAfter(models),
                              #PlotLocal(name, subdir, plots),
                              Report(os.path.join(subdir, 'report.txt'), name=name),
                              Printing()])
        main_loop.run()
        config['best_prec'] = main_loop.log.current_row['best_test_prec']
        print >> report, config['name'], 'best test prec', config['best_prec']
        report.flush()

    print ''.join(79 * '-')
    print 'FINAL REPORT'
    print ''.join(79 * '-')

    for config in configs:
        print config['name'], 'best test prec',config['best_prec']
        

if __name__ == '__main__':
    #theano.config.optimizer='fast_compile'
    run()
