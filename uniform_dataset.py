# -*- coding: utf-8 -*-
from __future__ import division
import os
from fuel.datasets import Dataset
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
import theano
import numpy as np

class UniformDataset(Dataset):
    def __init__(self, bs, sources, filename, which_sets, **kwargs):
        self.bs = bs
        self.provides_sources = sources
        self.filename = filename
        super(UniformDataset, self).__init__(**kwargs)
        self.train_set = H5PYDataset(self.filename, which_sets=which_sets, load_in_memory=True)
        self.handle = self.train_set.open()
        self.num_examples = self.train_set.num_examples
        

    def reset(self, state):
        self.train_set.reset(state)
    
    def get_data(self, state=None, request=None):
        features, labels = self.train_set.get_data(self.handle, request)
        return features, np.ones((features.shape[0],1)).astype('int16'), labels.astype('int32')
        max_time = 0
        for feature in features:
            if feature.shape[0] > max_time:
                max_time = feature.shape[0]
        bs = len(features)
        if bs == 0:
            return np.zeros((1,8096)).astype('float32'), np.zeros((1,1)).astype('int16'), np.zeros((1,51)).astype('int32')
        feats = np.zeros((bs, max_time, features[0].shape[-1]))
        time_mask = np.zeros((bs, max_time))
        labs = np.zeros((bs,1))
        neg = 0
        for i,feature in enumerate(features):
            if feature.shape[0] == 0:
                neg += 1
                continue
            feats[i,:feature.shape[0],:] = feature
            time_mask[i-neg, :feature.shape[0]] = 1
            labs[i-neg] = labels[i]
        return feats[:bs-neg].astype('float32'), time_mask[:bs-neg].astype('int16'), labels[:bs-neg].astype('int32')
