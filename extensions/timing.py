from __future__ import division
import time

from extensions import Extension



class TimeProfile(Extension):

    def __init__(self, **kwargs):
        kwargs.setdefault('before_epoch', True)
        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault('before_batch', True)
        kwargs.setdefault('after_batch', True)
        super(TimeProfile, self).__init__(**kwargs)

    def do(self, method, *args, **kwargs):
        log = self.main_loop.log
        if method == 'before_epoch':
            log.status['_epoch_start_time'] = time.clock()
            log.status['_batch_time_sum'] = 0
            log.status['_batch_time_count'] = 0
        elif method == 'before_batch':
            log.status['_batch_start_time'] = time.clock()
        elif method == 'after_batch':
            log.status['_batch_time_sum'] += time.clock()-log.status['_batch_start_time']
            log.status['_batch_time_count'] += 1
        elif method == 'after_epoch':
            log.current_row['Average Batch Time'] = 1000*log.status['_batch_time_sum'] /\
                                                    log.status['_batch_time_count']
            log.current_row['Epoch Time'] = 1000*(time.clock() - log.status['_epoch_start_time'])
