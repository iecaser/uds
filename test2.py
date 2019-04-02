import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from loguru import logger
from tqdm import tqdm

dim = 10
l_neighbors = -1
u_neighbors = -1


def init_dis(DM, n):
    if n == -1:
        dis = DM.mean(axis=0)
    elif n == 0:
        dis = DM.min(axis=0)
    else:
        dis = np.partition(DM, n, axis=0)[:n+1].mean(axis=0)
    return dis


def init_sim(SM, n):
    if n == -1:
        sim = SM.mean(axis=0)
    elif n == 0:
        sim = SM.max(axis=0)
    else:
        sim = -np.partition(-SM, n, axis=0)[:n+1].mean(axis=0)
    return sim


# params
amount = 10
EPSILON = 0.00001
rng = np.random.RandomState(12345)
L = rng.randn(100, dim)
U = rng.randn(500, dim)
LU = cosine_distances(L, U)
UU = cosine_similarity(U)
M, N = LU.shape[0], UU.shape[0]
BLANK = 0


logger.info(f'cal similarity...')
selected_indices, scores = [], []

dis_l = init_dis(LU, l_neighbors)
sim_u = init_sim(UU, u_neighbors)
for i in tqdm(range(amount)):
    # sample
    # score = (dis_labeled)**self.alpha / (EPSILON+dis_unlabeled)
    score = dis_l * sim_u
    sample_index = np.argmax(score)
    scores.append(score[sample_index])
    sample_sim = UU[sample_index, :]
    sample_dis = 1 - sample_sim
    selected_indices.append(sample_index)
    # update labeled
    if l_neighbors == -1:
        dis_l = (dis_l*M + sample_dis) / (M+1)
        M += 1
    elif l_neighbors == 0:
        dis_l = np.c_[dis_l, sample_dis].min(axis=1)
    else:
        LU = np.r_[LU, sample_dis.reshape(1, -1)]
        LU[:, sample_index] = MIN_DISTANCE
        dis_labeled = np.partition(LU, l_neighbors, axis=0)[:l_neighbors+1].mean(axis=0)
    # update labeled
    if u_neighbors == -1:
        sim_u = (sim_u * N - sample_sim) / (N-1)
        N -= 1
    else:
        UU[sample_index, :], UU[:, sample_index] = BLANK, BLANK
        sim_u = -np.partition(-UU, u_neighbors, axis=0)[:u_neighbors+1].mean(axis=0)
    dis_l[sample_index], sim_u[sample_index] = BLANK, BLANK
print(scores)
logger.info(f'scores sum: {sum(scores)}')
