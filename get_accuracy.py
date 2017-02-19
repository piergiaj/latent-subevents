import numpy as np

files = ['results'+s+'.npy' for s in ['A','B','C','D','E','F','G','H','I']]

results = None
for f in files:
    x = np.load(f)
    if results is None:
        results = x
    else:
        results = np.concatenate([results,x], axis=0)

print results.shape

labs = np.load('labsA.npy')

pred = np.argmax(results,axis=0)

print sum(labs == pred)

print sum(labs == pred)/1530.
