import os
import sys
import gzip
from random import shuffle
from subprocess import Popen, PIPE

import numpy as np
import cPickle

import h5py
from fuel.datasets.hdf5 import H5PYDataset
import theano
import theano.tensor as T
from skimage import io

from skimage.transform import resize

from os import listdir
from os.path import isfile, join


def load_data(train, test, s, overlap=1):
    train_filelist = open(train, 'r').read().split('\n')
    print len(train_filelist)
    shuffle(train_filelist)

    test_filelist = open(test, 'r').read().split('\n')
    print len(test_filelist)
    shuffle(test_filelist)

    dataset_size = len(train_filelist)+len(test_filelist)


    f = h5py.File('/ssd2/hmdb/hmdb-tdd.hdf5', mode='w')
    dtype = h5py.special_dtype(vlen=np.dtype('float32'))  
    features = f.create_dataset('features', (dataset_size,), dtype=dtype, compression='gzip', compression_opts=7)
    features_shapes = f.create_dataset('features_shapes', (dataset_size,2), dtype=int)
    features_shape_labels = f.create_dataset('features_shape_labels', (2,), dtype='S7')

    labels = f.create_dataset('labels', (dataset_size,1), dtype=int, compression='gzip')

    for i,fn in enumerate(train_filelist+test_filelist):
        if len(fn) == 0:
            continue
        fn, label = fn.split(' ')
        fn = fn.replace('/fast-data/hmdb/image-file/', '/ssd2/hmdb/feats/')
        print i, fn, label

        labels[i] = int(label)

        output = np.genfromtxt(fn, skip_header=1)

        features[i] = output.flatten()
        print output.shape
        features_shapes[i] = output.shape
        
    features.dims.create_scale(features_shapes, 'shapes')
    features.dims[0].attach_scale(features_shapes)
    
    features_shape_labels[...] = ['frames'.encode('utf8'), 'channels'.encode('utf8')]

    features.dims.create_scale(features_shape_labels, 'shape_labels')
    features.dims[0].attach_scale(features_shape_labels)

    features.dims[0].label = 'batch'
    labels.dims[0].label = 'batch'


    trn = len(train_filelist)
    tst = len(test_filelist)
    split_dict = {'train': {'features': (0, trn),
                            'labels': (0, trn)},
                  'test': {'features': (trn, trn+tst),
                           'labels': (trn, trn+tst)}}

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()


if __name__ == '__main__':
    for s in ['1']:
        load_data('/ssd2/hmdb/train_'+s+'.txt', '/ssd2/hmdb/test_'+s+'.txt', s)



