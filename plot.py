import ipdb
import matplotlib.pyplot as plt
import os
import pickle
import glob
import numpy as np
from absl import app
from absl import flags
from loguru import logger

FLAGS = flags.FLAGS

flags.DEFINE_string('exp', 'exp', 'exp_director')
flags.DEFINE_integer('idx', 999, 'specify exp idx to plot')
flags.DEFINE_string('dataset', 'mnist', 'Specified dataset')
flags.DEFINE_integer('init', 100, 'initial size')
flags.DEFINE_integer('batch', 100, 'batch size')

MARKERS = ['+:', 'x-.', 'd--', ',--',  '.:', '|-.', 's-.', '1-', 'p:', '*-.', '^--', 'o-']


def main(_):
    save_dir = 'results'
    basepath = os.path.join('/home/zxf/workspace/DiscriminativeActiveLearning/', FLAGS.exp)
    methods = os.listdir(basepath)
    methods.remove(save_dir)
    logger.info('plot exp/idx: {}/{}...'.format(FLAGS.exp, FLAGS.idx))
    valid_methods = []
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    methods = ['Random', 'Uncertainty', 'UncertaintyDensity', 'CoreSet', 'EGL',
               'DualDensity', 'DualDensityBeam2', 'DualDensityBeam3']
    for i, mp in enumerate(methods):
        if mp in ['Adversarial', 'DualDensityBeam4',
                  'UncertaintyDualDensity', 'DynamicUncertaintyDualDensity']:
            continue
        logger.info('METHOD: {}'.format(mp))
        if FLAGS.idx == 999:
            filepath = glob.glob(os.path.join(
                basepath, mp, 'results/*{}_{}_{}_[0-9]*.pkl'.format(FLAGS.dataset, FLAGS.init, FLAGS.batch)))
        else:
            filepath = glob.glob(os.path.join(
                basepath, mp, 'results/*{}_{}_{}_{}.pkl'.format(FLAGS.dataset, FLAGS.init, FLAGS.batch, FLAGS.idx)))
        acc = []
        for fp in filepath:
            if 'entropy' in fp:
                continue
            if 'label' in fp:
                continue
            # print(fp)
            with open(fp, 'rb') as f:
                accuracies, initial_size, batch_size = pickle.load(f)
            acc.append(accuracies)
        if len(acc) > 0:
            if mp == 'DualDensity':
                valid_methods.append('DWDAL-1')
            elif mp == 'DualDensityBeam2':
                valid_methods.append('DWDAL-2')
            elif mp == 'DualDensityBeam3':
                valid_methods.append('DWDAL-3')
            else:
                valid_methods.append(mp)
            acc = np.array(acc)
            acc = acc.mean(axis=0)
            # acc = np.median(acc, axis=0)
            plt.plot(acc*100, MARKERS[i], linewidth=1, markersize=2.5)
    plt.legend(valid_methods)
    dataset, init, batch = FLAGS.dataset, FLAGS.init, FLAGS.batch
    plt.title(f'Dataset: {dataset.upper()}, Initial size: {init}, Batch size: {batch}')
    plt.xlabel('Iterations')
    plt.ylabel('Classification Accuray (%)')
    plt.savefig('{}/{}/{}_{}_{}_{}.png'.format(FLAGS.exp, save_dir,
                                               FLAGS.dataset, FLAGS.init,
                                               FLAGS.batch, FLAGS.idx))


if __name__ == '__main__':
    app.run(main)
