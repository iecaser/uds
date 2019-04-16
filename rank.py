import ipdb
import os
import numpy as np
import matplotlib.pyplot as plt

output = 'ranks'
filedir = '/home/zxf/workspace/DiscriminativeActiveLearning/mnist.500.2400.1'
methods = os.listdir(filedir)
mps, qs = [], []
for method in methods:
    if method in ['results', 'EGL', 'Random', 'DualDensityBeam2', 'DualDensityBeam3', 'DualDensityBeam4']:
        continue
    try:
        filename = method + '_mnist_500_2400_0_entropy.pkl'
        filepath = os.path.join(filedir, method, 'results', filename)
        [_, _, query] = np.load(filepath)
        query = np.array(query).reshape(-1)
        # if len(query) == 3:
        # query = np.concatenate(query)
        if method == 'DualDensity':
            method = 'DWDS'
        elif method == 'Uncertainty':
            method = 'US'
        elif method == 'UncertaintyDensity':
            method = 'DWUS'
        mps.append(method)
        qs.append(query/3000.)
    except:
        pass
for m1, x1 in zip(mps, qs):
    for m2, x2 in zip(mps, qs):
        if m1 == m2:
            continue
        if x1.shape[0] > 2400:
            x1 = x1[:2400]
        if x2.shape[0] > 2400:
            x2 = x1[:2400]
        if not x1.shape == x2.shape:
            import ipdb
            ipdb.set_trace()
        plt.figure()
        print(m1+m2)
        plt.title(m1+' v.s. '+m2)
        plt.xlabel(f'rank by {m1}')
        plt.ylabel(f'rank by {m2}')
        plt.scatter(x1, x2, marker='.', alpha=0.8)
        plt.savefig(os.path.join(output, m1+m2))
