from modAL.density import information_density
from sklearn.preprocessing import normalize
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from tqdm import tqdm

L = np.random.randint(-100, 100, (1000, 128))
U = np.random.randint(-100, 100, (50000, 128))
# LU = distance_matrix(L, U)
# UU = distance_matrix(U, U)
print('cal distance matrix')
LU = cdist(L, U)
UU = cdist(U, U)

print('cal similarity')
# labeled density
sim_labeled = LU.min(axis=0)
# unlabeled density
dis_unlabeled = UU.mean(axis=0)
sim_unlabeled = 1 / (1+dis_unlabeled)
sample_indices = []
amount = 100
N = UU.shape[0]
for i in tqdm(range(amout)):
    # sample
    sim = sim_labeled * sim_unlabeled
    sample_index = np.argmax(sim)
    sample_indices.append(sample_index)

    # update
    sample = UU[sample_index, :]
    sim_labeled = np.c_[sim_labeled, sample].min(axis=1)
    sim_labeled[sample_index] = 0
    dis_unlabeled = (dis_unlabeled * N - sample) / (N-1)
    # dis_unlabeled[a] = 9999
    sim_unlabeled = 1 / (1+dis_unlabeled)
    N -= 1
# print(sample_indices)
# print(len(set(sample_indices)))
