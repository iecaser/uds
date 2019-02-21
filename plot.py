import ipdb
import matplotlib.pyplot as plt
import os
import pickle
import glob
import numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'mnist', 'Specified dataset')
flags.DEFINE_integer('init', 100, 'initial size')

def main(argv):
    basepath = '/home/zxf/workspace/DiscriminativeActiveLearning/exp/'
    methods = os.listdir(basepath)
    valid_methods = []
    for mp in methods:
        filepath = glob.glob(os.path.join(basepath, mp, 'results/*{}_{}_1000_[0-9].pkl'.format(FLAGS.dataset,FLAGS.init)))
        acc = []
        for fp in filepath:
            print(fp)
            with open(fp, 'rb') as f:
                accuracies, initial_size, batch_size = pickle.load(f)
            acc.append(accuracies)
        if len(acc)>0:
            valid_methods.append(mp)
            acc = np.array(acc).mean(axis=0)
            plt.plot(acc)
    plt.legend(valid_methods)
    plt.savefig('{}_{}_1000.png'.format(FLAGS.dataset, FLAGS.init))
if __name__ == '__main__':
    app.run(main)

