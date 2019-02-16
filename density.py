import numpy as np
import copy

n_samples = 10000
n_features = 2
bin_size = 0.2
amount = 100
representation = np.random.rand(n_samples, n_features)
information = np.random.rand(n_samples, 1).reshape(n_samples,)
bins = np.arange(0, 1+bin_size, bin_size)
n_bins = bins.shape[0]-1
index = np.digitize(representation, bins) - 1
index[index >= n_bins] = n_bins-1
index_bin = [[] for __ in range(n_bins**n_features)]
information_bin = copy.deepcopy(index_bin)
for i, idx in enumerate(index):
    index = np.ravel_multi_index(idx, dims=tuple(np.ones(n_features, dtype='int')*n_bins))
    index_bin[index].append(i)
    information_bin[index].append(information[i])
information_index, information_val = [], []
for infob, idxb in zip(information_bin, index_bin):
    idx, val = max(zip(idxb, infob), key=lambda x: x[1])
    information_index.append(idx)
    information_val.append(val)
pdf = [len(idxb) for idxb in index_bin]
choose_mat = np.multiply(information_val, pdf)
for i in range(amount):
    # chose
    which_bin = np.argmax(choose_mat)
    which_index = information_index[which_bin]

    # udpate
    pdf[which_bin] -= 1
    information_bin[which_bin].remove(information[which_index])
    index_bin[which_bin].remove(which_index)
    information_index[which_bin], information_val[which_bin] = max(
        zip(index_bin[which_bin], information_bin[which_bin]), key=lambda x: x[1])
    choose_mat[which_bin] = pdf[which_bin]*information_val[which_bin]
