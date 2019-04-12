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

MARKERS = ['+:', 'x-.', 'd--', 's-', '.:', '|-.', ',--', '1-', 'p:', '*-.', '^--', 'o-']


def get_outlier_mask(x, outlierConstant=0.5):
    # mask = np.ones(shape=(acc.shape[0],)).astype('bool')
    # for iteration in range(acc.shape[1]):
    #     mask &= get_outlier_mask(acc[:, iteration], )
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    mask = (a >= quartileSet[0]) & (a <= quartileSet[1])
    logger.info('mask percent: {}'.format(mask.sum()/mask.shape[0]))
    return mask


def get_bias_mask(x, bias, c=0.05):
    mask = np.abs(np.array(x[:, 0]) - bias) < c
    logger.info('mask percent: {}'.format(mask.sum()/mask.shape[0]))
    return mask


def get_increase_mask(x, c=-0.05):
    diff = x[:, 1:]-x[:, :-1]
    mask = diff.min(axis=1) >= c
    logger.info('mask percent: {}'.format(mask.sum()/mask.shape[0]))
    return mask


def main(_):
    save_dir = 'results'
    basepath = os.path.join('/home/zxf/workspace/DiscriminativeActiveLearning/', FLAGS.exp)
    methods = os.listdir(basepath)
    methods.remove(save_dir)
    valid_methods = []
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    for i, mp in enumerate(methods):
        logger.info('method : {}'.format(mp))
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
            # print(fp)
            with open(fp, 'rb') as f:
                accuracies, initial_size, batch_size = pickle.load(f)
            acc.append(accuracies)
        if len(acc) > 0:
            valid_methods.append(mp)
            acc = np.array(acc)
            mask = get_increase_mask(acc, c=-0.035)
            # mask = get_bias_mask(acc, bias=0.95, c=0.05)
            masked_acc = acc[mask]
            result = masked_acc.mean(axis=0)
            plt.plot(result, MARKERS[i], linewidth=1, markersize=2.5)
    plt.legend(valid_methods)
    plt.savefig('{}/{}/{}_{}_{}_{}b.png'.format(FLAGS.exp, save_dir,
                                                FLAGS.dataset, FLAGS.init,
                                                FLAGS.batch, FLAGS.idx))


if __name__ == '__main__':
    app.run(main)
