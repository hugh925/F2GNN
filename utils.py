import dgl
import numpy as np
import scipy.sparse as sp
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score
import os
import random


def load_config(dataset):
    with open('config/config.yml', 'r') as file:
        config = yaml.safe_load(file)
    return config.get(dataset, {})

def update_args_from_config(args, config):
    for key, value in config.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)

# random seed setting
def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    dgl.seed(seed)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    mx = np.array(mx)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = torch.FloatTensor(mx)
    return mx

# threshold adjusting for best macro 'in cuda'
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in torch.linspace(0.05, 0.95, 19):
        preds = torch.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres.item()
    return best_f1, best_thre

