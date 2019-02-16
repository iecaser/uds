import ipdb
import matplotlib.pyplot as plt
import os
import pickle
import glob
import numpy as np

basepath = '/home/zxf/workspace/DiscriminativeActiveLearning/exp/'
method = os.listdir(basepath)
for mp in method:
    filepath = glob.glob(os.path.join(basepath, mp, 'results/*100_1000_[0-9].pkl'))
    acc = []
    for fp in filepath:
        print(fp)
        with open(fp, 'rb') as f:
            accuracies, initial_size, batch_size = pickle.load(f)
        acc.append(accuracies)
    acc = np.array(acc).mean(axis=0)
    plt.plot(acc)
plt.legend(method)
plt.savefig('mnist_100_1000.png')
