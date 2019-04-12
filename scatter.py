import os
import numpy as np
from loguru import logger
from sklearn.datasets import load_iris as load_iris_dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('dir', None, 'method')


# # import some data to play with
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
SEED = 1234


def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]


def load_iris():
    iris = load_iris_dataset()
    X, y = iris['data'], iris['target']
    # ss = StandardScaler()
    # X = ss.fit_transform(X)
    # enc = OneHotEncoder()
    # y = enc.fit_transform(y.reshape(-1, 1)).toarray()
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=SEED)
    return (X_train, y_train), (X_val, y_val)


def main(_):
    filedir = f'{FLAGS.dir}/results'
    (x, y), (_, __) = load_iris()
    logger.info(f'scatter {filedir}')
    for filename in os.listdir(filedir):
        if 'labeled.pkl' in filename:
            filepath = os.path.join(filedir, filename)

            pca = PCA(n_components=2)
            x = pca.fit_transform(x)
            logger.info(f'explained: {pca.explained_variance_ratio_}')
            with open(filepath, 'rb') as f:
                perm, initial_size, batch_size = pickle.load(f)
            mark = initial_size
            batch_iterations = (perm.shape[0]-initial_size)//batch_size
            for batch in range(batch_iterations):
                old_labeled = perm[:mark]
                new_idx = perm[mark:mark + batch_size]
                unlabeled_idx = get_unlabeled_idx(x, perm[:mark+batch_size])
                mark += batch_size

                plt.figure(batch)
                plt.scatter(x[unlabeled_idx, 0], x[unlabeled_idx, 1],
                            color='gray', marker='x', alpha=0.2)
                plt.scatter(x[old_labeled, 0], x[old_labeled, 1], color='r', facecolors='none')
                plt.scatter(x[new_idx, 0], x[new_idx, 1], color='r', alpha=0.5)
                # for i, idx in enumerate(new_idx):
                #     plt.text(x[idx, 0], x[idx, 1]+0.01, i, fontsize=8)
                plt.savefig(f'scatter/{filename[:-4]}.{batch}.png')


if __name__ == '__main__':
    app.run(main)
