import os
import random
import pickle as pkl
from collections import defaultdict
import json
import numpy as np
from scipy import sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import *



# ============= load adjs for randomwalk ============
def matrix2list(matrix):
    """
    将邻接矩阵转换成邻接链表
    """
    matrix = matrix.toarray()
    adj_lists = defaultdict(set)
    for i, col in enumerate(matrix):
        adj_lists[i] = set(np.where(col>0)[0])
    return adj_lists


def get_adjs(matrix):
    """
    input: apk_tpl的邻接矩阵
    return: apk_tpl, tpl_apk的邻接链表
    """
    return matrix2list(matrix), matrix2list(matrix.transpose())


def load_adjs(data_dir):
    adjs_fromApk = []
    adjs_toApk = []
    for mp_id in metapaths:
        view = prefix[mp_id]
        if os.path.exists(os.path.join(data_dir, 'adj_app_{}.pkl'.format(view))):
            with open(os.path.join(data_dir, 'adj_app_{}.pkl'.format(view)), 'rb') as f:
                adj = pkl.load(f)
            with open(os.path.join(data_dir, 'adj_{}_app.pkl'.format(view)), 'rb') as f:
                mirror_adj = pkl.load(f)
        else:
            matrix = sp.load_npz(os.path.join(data_dir, '{}_feats.npz'.format(view)))
            adj, mirror_adj = get_adjs(matrix)
            with open(os.path.join(data_dir, 'adj_app_{}.pkl'.format(view)), 'wb') as f:
                pkl.dump(adj, f)
            with open(os.path.join(data_dir, 'adj_{}_app.pkl'.format(view)), 'wb') as f:
                pkl.dump(mirror_adj, f)
        adjs_fromApk.append(adj)
        adjs_toApk.append(mirror_adj)
    return adjs_fromApk, adjs_toApk



# ========== load dataset for classifer =============
def load_data_for_optim(data_dir, batch_size):
    with open(os.path.join(data_dir, 'split_info.json'), 'r') as f:
        split_info = json.load(f)
    with open(os.path.join(data_dir, 'images'), 'rb') as f:
        data = np.array(pkl.load(f))

    train_x = data[split_info['train_ids']]
    val_x = data[split_info['val_ids']]
    test_x = data[split_info['test_ids']]

    train_y = split_info['train_y']
    val_y = split_info['val_y']
    test_y = split_info['test_y']

    train_dataset = TensorDataset(torch.Tensor(train_x), torch.LongTensor(train_y))
    test_dataset = TensorDataset(torch.Tensor(test_x), torch.LongTensor(test_y))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader
