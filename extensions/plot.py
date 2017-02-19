from collections import namedtuple
from functools import total_ordering
import logging
import os
import signal
import time
from six.moves.queue import PriorityQueue
from subprocess import Popen, PIPE
from threading import Thread
from extensions import Extension

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bokeh.plotting import (curdoc, cursession, figure, output_server,
                                push, show)
logger = logging.getLogger(__name__)


class Plot(Extension):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, document, channels, server, **kwargs):
        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault('after_training', True)
        super(Plot, self).__init__(**kwargs)
        self.plots = {}
        self.document = document
        self.server = server

        output_server(self.document, url=self.server)

        self.p = []
        self.p_indx = {}
        self.color_indx = {}
        for i, channel_set in enumerate(channels):
            channel_set_opts = {}
            if isinstance(channel_set, dict):
                channel_set_opts = channel_set
                channel_set = channel_set_opts.pop('channels')
            channel_set_opts.setdefault('title',
                                        '{} #{}'.format(document, i + 1))
            channel_set_opts.setdefault('x_axis_label', 'epochs')
            channel_set_opts.setdefault('y_axis_label', 'value')
            self.p.append(figure(**channel_set_opts))
            for j, channel in enumerate(channel_set):
                self.p_indx[channel] = i
                self.color_indx[channel] = j

    @property
    def push_thread(self):
        if not hasattr(self, '_push_thread'):
            self._push_thread = PushThread()
            self._push_thread.start()
        return self._push_thread


    def do(self, method, *args):
        log = self.main_loop.log
        epoch = log.status['epochs_done']
        for k,v in log.current_row.items():
            if k in self.p_indx:
                if k not in self.plots:
                    line_color = self.colors[self.color_indx[k] % len(self.colors)]
                    fig = self.p[self.p_indx[k]]
                    fig.line([epoch], [v], legend=k, name=k, line_color=line_color)
                    renderer = fig.select(dict(name=k))
                    self.plots[k] = renderer[0].data_source
                else:
                    self.plots[k].data['x'].append(epoch)
                    self.plots[k].data['y'].append(v)
                    self.push_thread.put(self.plots[k], PushThread.PUT)
        self.push_thread.put(method, PushThread.PUSH)


@total_ordering
class _WorkItem(namedtuple('BaseWorkItem', ['priority', 'obj'])):
    __slots__ = ()

    def __lt__(self, other):
        return self.priority < other.priority




class PushThread(Thread):
    # Define priority constants
    PUSH = 1
    PUT = 2

    def __init__(self):
        super(PushThread, self).__init__()
        self.queue = PriorityQueue()
        self.setDaemon(True)

    def put(self, obj, priority):
        self.queue.put(_WorkItem(priority, obj))

    def run(self):
        while True:
            priority, obj = self.queue.get()
            if priority == PushThread.PUT:
                cursession().store_objects(obj)
            elif priority == PushThread.PUSH:
                push()
                # delete queued objects when training has finished
                if obj == "after_training":
                    with self.queue.mutex:
                        del self.queue.queue[:]
                    break
            self.queue.task_done()



class PlotLocal(Extension):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, document, subdir, channels, **kwargs):
        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault('after_training', True)
        super(PlotLocal, self).__init__(**kwargs)
        self.document = document
        self.subdir = subdir
        if not os.path.isdir(self.subdir):
            os.mkdir(self.subdir)

        self.p = []
        self.p_indx = {}
        self.color_indx = {}
        self.data = {}
        self.plots = {}

        self.fig = plt.figure(figsize=plt.figaspect(len(channels)))
        for i, channel_set in enumerate(channels):
            ax = self.fig.add_subplot(len(channels),1,i+1)
            ax.set_ylabel('value')
            ax.set_xlabel('epochs')
            ax.set_title('{} #{}'.format(document, i + 1))
            self.p.append(ax)
            for j, channel in enumerate(channel_set):
                self.p_indx[channel] = i
                self.color_indx[channel] = j
                self.data[channel] = []

    def do(self, method, *args):
        log = self.main_loop.log
        epoch = log.status['epochs_done']
        for k,v in log.current_row.items():
            if k in self.p_indx:
                if k not in self.plots:
                    line_color = self.colors[self.color_indx[k] % len(self.colors)]
                    ax = self.p[self.p_indx[k]]
                    #print k,v
                    v = np.mean(v)
                    self.data[k].append(v)
                    line,=ax.plot(np.arange(len(self.data[k])), self.data[k], line_color, label=k)
                    ax.legend()
                    self.plots[k] = (ax,line)
                else:
                    (ax,line) = self.plots[k]
                    line.set_xdata(np.append(line.get_xdata(), epoch))
                    line.set_ydata(np.append(line.get_ydata(), v))
                    ax.relim()
                    ax.autoscale_view()
        plt.draw()
#        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(os.path.join(self.subdir,str(self.document)+'.pdf'), transparent=True)
