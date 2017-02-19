import logging
from toolz import first
import numpy as np
logger = logging.getLogger(__name__)


class Extension(object):
    
    def __init__(self, before_epoch=False, before_batch=False, 
                 after_batch=False, after_epoch=False,
                 before_training=False, after_training=False):
        self.conditions = []
        if before_training:
            self.conditions.append('before_training')
        if before_epoch:
            self.conditions.append('before_epoch')
        if after_epoch:
            self.conditions.append('after_epoch')
        if before_batch:
            self.conditions.append('before_batch')
        if after_batch:
            self.conditions.append('after_batch')
        if after_training:
            self.conditions.append('after_training')

    def notify(self, method, *args):
        if method in self.conditions:
            self.do(method, *args)

    def do(self, method, *args):
        pass


class FinishAfter(Extension):

    def __init__(self, num_epochs, **kwargs):
        kwargs.setdefault('after_epoch', True)
        super(FinishAfter, self).__init__(**kwargs)
        self.num_epochs = num_epochs

    def do(self, *args, **kwargs):
        if self.main_loop.status['epochs_done'] >= self.num_epochs:
            self.main_loop.log.current_row['training_finish_requested'] = True

class Printing(Extension):

    def __init__(self, **kwargs):
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_epoch", True)
        super(Printing, self).__init__(**kwargs)


    def print_attributes(self, atts):
        for attr, val in sorted(atts.items(), key=first):
            if not attr.startswith('_'):# or True:
                print '\t', '{}:'.format(attr), val

    def do(self, method, *args):
        log = self.main_loop.log
        print ''.join(79 * '-')
        if method == 'before_epoch' and log.status['epochs_done'] == 0:
            print 'BEFORE FIRST EPOCH'
        elif method == 'after_training':
            print 'TRAINING HAS BEEN FINISHED:'
        elif method == 'after_epoch':
            print 'AFTER EPOCH', log.status['epochs_done']
        print ''.join(79 * '-')

        print 'Training status:'
        self.print_attributes(log.status)
        print 'Log records from the iteration {}:'.format(
            log.status['iterations_done'])
        self.print_attributes(log.current_row)
        print


class Report(Extension):

    def __init__(self, report_file, name='', **kwargs):
        self.f = open(report_file, 'w')
        print >> self.f, ''.join(79 * '-')
        print >> self.f, 'Training Report for', name
        print >> self.f, ''.join(79 * '-')
        print >> self.f
        self.f.flush()

        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_epoch", True)
        super(Report, self).__init__(**kwargs)


    def print_attributes(self, atts):
        for attr, val in sorted(atts.items(), key=first):
            if not attr.startswith('_'):
                print >> self.f, '\t', '{}:'.format(attr), val

    def do(self, method, *args):
        log = self.main_loop.log
        print >> self.f, ''.join(79 * '-')
        if method == 'before_epoch' and log.status['epochs_done'] == 0:
            print >> self.f, 'BEFORE FIRST EPOCH'
        elif method == 'after_training':
            print >> self.f, 'TRAINING HAS BEEN FINISHED:'
        elif method == 'after_epoch':
            print >> self.f, 'AFTER EPOCH', log.status['epochs_done']
        print >> self.f, ''.join(79 * '-')

        print >> self.f, 'Training status:'
        self.print_attributes(log.status)
        print >> self.f, 'Log records from the iteration {}:'.format(
            log.status['iterations_done'])
        self.print_attributes(log.current_row)
        print >> self.f
        self.f.flush()



class Track(Extension):
    def __init__(self, variables, prefix='', **kwargs):
        kwargs.setdefault('before_training', True)
        kwargs.setdefault('before_epoch', True)
        kwargs.setdefault('after_batch', True)
        kwargs.setdefault('after_epoch', True)
        super(Track, self).__init__(**kwargs)
        self.variables = variables
        for i, ent in enumerate(self.variables):
            if type(ent) is not tuple:
                self.variables[i] = (ent, 'mean')
        self.prefix = prefix

    def get_name(self, name):
        return self.prefix + '_' + name if self.prefix != '' else name

    def do(self, method, *args):
        log = self.main_loop.log
        if method == 'before_epoch':
            for i in range(len(self.variables)):
                v = self.variables[i][0]
                log.current_row['_'+self.get_name(v)+'_tmp'] = 0
                log.current_row['_'+self.get_name(v)+'_batches'] = 0
        elif method == 'after_batch':
            for i in range(len(self.variables)):
                v = self.variables[i][0]
                if self.variables[i][1] == 'after':
                    continue
                val = self.main_loop.outputs[self.main_loop.output_map[v]]
                pval = log.previous_row['_'+self.get_name(v)+'_tmp']
                if self.variables[i][1] in ['mean','sum']:
                    log.current_row['_'+self.get_name(v)+'_tmp'] = val + pval
                elif self.variables[i][1] == 'last':
                    if hasattr(val,'get_value'):
                        val = val.get_value()
                    log.current_row['_'+self.get_name(v)+'_tmp'] = val
                elif self.variables[i][1] == 'all':
                    if not hasattr(pval, 'shape') and pval == 0:
                        log.current_row['_'+self.get_name(v)+'_tmp'] = val
                    else:
                        log.current_row['_'+self.get_name(v)+'_tmp'] = np.concatenate([val,pval],axis=0)
                log.current_row['_'+self.get_name(v)+'_batches'] = log.previous_row['_'+
                                                        self.get_name(v)+'_batches'] + 1
        elif method == 'after_epoch':
            for i in range(len(self.variables)):
                v = self.variables[i][0]
                if self.variables[i][1] != 'after':
                    nv = log.current_row['_'+self.get_name(v)+'_tmp']
                if self.variables[i][1] == 'mean':
                    nvm = nv / log.current_row['_'+self.get_name(v)+'_batches']
                    if hasattr(nvm, 'shape'):
                        nvm = nvm.sum()
                    log.current_row[self.get_name(v)] = nvm
                if self.variables[i][1] == 'sum':
                    log.current_row[self.get_name(v)] = nv
                if self.variables[i][1] == 'after':
                    log.current_row[self.get_name(v)] = self.variables[i][4](log.current_row[self.get_name(self.variables[i][2])],log.current_row[self.get_name(self.variables[i][3])])
                elif self.variables[i][1] == 'last' or self.variables[i][1] == 'all':
                    log.current_row[self.get_name(v)] = nv
                if self.variables[i][1] == 'all':
                    pass


class TrackBest(Track):

    def __init__(self, best_method=None, **kwargs):
        super(TrackBest, self).__init__(**kwargs)
        self.best_method = best_method
        if best_method is None:
            self.best_method = [min]*len(self.variables)

    def do(self, method, *args):
        log = self.main_loop.log
        if method == 'before_training':
            for i in range(len(self.variables)):
                v = self.variables[i][0]
                m = float("inf") if self.best_method[i] == min else float("-inf")
                log.current_row[self.get_name(v)] = m
                log.current_row['_'+self.get_name(v)+'_bstiter'] = 0
        elif method == 'before_epoch':
            for i in range(len(self.variables)):
                v = self.variables[i][0]
                if log.status['iterations_done'] > 0:
                    log.current_row[self.get_name(v)] = log.current_row[self.get_name(v)]
                    log.status['_best_epoch'+self.get_name(v)] = False
                log.current_row['_'+self.get_name(v)+'_tmp'] = 0
                log.current_row['_'+self.get_name(v)+'_batches'] = 0
        elif method == 'after_batch':
            for i in range(len(self.variables)):
                v = self.variables[i][0]
                val = self.main_loop.outputs[self.main_loop.output_map[v]]
                pval = log.previous_row['_'+self.get_name(v)+'_tmp']
                log.current_row['_'+self.get_name(v)+'_tmp'] = val + pval
                log.current_row['_'+self.get_name(v)+'_batches'] = log.previous_row['_'+
                                                        self.get_name(v)+'_batches'] + 1
                log.current_row[self.get_name(v)] = log.previous_row[self.get_name(v)]
                log.current_row['_'+self.get_name(v)+'_bstiter'] = log.previous_row['_'+
                                                                self.get_name(v)+'_bstiter']
        elif method == 'after_epoch':
            for i in range(len(self.variables)):
                v = self.variables[i][0]
                nv = log.current_row['_'+self.get_name(v)+'_tmp']
                nv /= log.current_row['_'+self.get_name(v)+'_batches']
                if hasattr(nv, 'shape'):
                    nv = nv.sum()
                cv = log.current_row[self.get_name(v)]
                log.current_row[self.get_name(v)] = self.best_method[i](cv, nv)
                if nv == self.best_method[i](cv, nv):
                    log.current_row['_'+self.get_name(v)+'_bstiter'] = log.status['epochs_done']
                    log.current_row['_'+self.get_name(v)+'_new_best'] = True
                    log.status['_best_epoch'+self.get_name(v)] = True
                    log.current_row['best_'+self.get_name(v)] = (nv,cv)
        
