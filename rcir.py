#%%
# =================== imports ===================
import argparse
import importlib
import json
import os
import pickle
import shutil
import sys
from multiprocessing import Pool
from os.path import abspath, dirname, exists, join

import faiss
import matin
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.utils as utils
from utils.cp_utils import cal_recall
from options import FixRandom, Options

np.set_printoptions(precision=3, suppress=True)
plt.style.use('ggplot')

sys.argv = ['']
parser = argparse.ArgumentParser(description="Options")
parser.add_argument('--dbs', type=str, default='cub200', help='choose dataset.', choices=['pitts', 'cub200', 'car196', 'chestx'])
parser.add_argument('--unc', type=str, default='btl', help='choose dataset.', choices=['euc', 'btl', 'dul', 'mcd'])
opt_local = parser.parse_args()


hyps = {'dbs': None, 'score': None, 'unc': None}
if sys.argv == ['']:
    hyps['dbs'] = 'cub200'                       # 'cub200' | 'car196' | 'pitts'| 'chestx' | 'sop'
    hyps['unc'] = 'btl'                          # 'btl' | 'dul' | 'mcd' | 'euc'|
else:
    hyps['dbs'] = opt_local.dbs                  # 'cub200' | 'car196' | 'pitts'| 'chestx'
    hyps['unc'] = opt_local.unc                  # 'btl' | 'dul' | 'mcd' | 'euc'|

print(f"==> {list(hyps.values())}")

resume, dataset = utils.load_cir_v2(hyps)
print(f"==> load {resume}")

with open(join(resume, 'flags.json')) as f:
    opt = edict(json.load(f))
    opt.split = 'test'
    opt.resume = resume

torch.cuda.set_device(opt.cGPU)
device = torch.device("cuda")
fix_random = FixRandom(opt.seed)
seed_worker = fix_random.seed_worker()

meta_dir = join(opt.resume, "meta_rcrs")
# # remvoe meta_dir if exists
# if exists(meta_dir):
#     shutil.rmtree(meta_dir)
os.makedirs(meta_dir, exist_ok=True)

# Load embeddings/predictions
if hyps['dbs'] in ['pitts']:
    whole_train_set = dataset.Whole('train', data_path=opt.data_path, img_size=(opt.height, opt.width))
    whole_training_data_loader = DataLoader(dataset=whole_train_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=opt.cuda, worker_init_fn=seed_worker)
    whole_val_set = dataset.Whole('val', data_path=opt.data_path, img_size=(opt.height, opt.width))
    whole_val_data_loader = DataLoader(dataset=whole_val_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=opt.cuda, worker_init_fn=seed_worker)
    whole_test_set = dataset.Whole('test', data_path=opt.data_path, img_size=(opt.height, opt.width))
    whole_test_data_loader = DataLoader(dataset=whole_test_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=opt.cuda, worker_init_fn=seed_worker)
    print(f"==> train: db={whole_train_set.get_databases().shape}, q={whole_train_set.get_queries().shape}")
    print(f"==> val: db={whole_val_set.get_databases().shape}, q={whole_val_set.get_queries().shape}")
    print(f"==> test: db={whole_test_set.get_databases().shape}, q={whole_test_set.get_queries().shape}")

    # VAL ================================ #
    with open(join(opt.resume, 'val_embeddings_best.pickle'), 'rb') as handle:
        q_mu_cal = pickle.load(handle)
        db_mu_cal = pickle.load(handle)
        q_sigma_cal = pickle.load(handle)
        db_sigma_cal = pickle.load(handle)
        preds_cal = pickle.load(handle)
        dists_cal = pickle.load(handle)
        positives_cal = pickle.load(handle)
    print(f'==> check shape: q_mu={q_mu_cal.shape}, db_mu_val={db_mu_cal.shape}; q_sigma_val={q_sigma_cal.shape}, db_sigma_val={db_sigma_cal.shape}')
    if hyps['unc'] == 'mcd':
        q_sigma_cal = q_sigma_cal.mean(axis=1, keepdims=True)
        db_sigma_cal = db_sigma_cal.mean(axis=1, keepdims=True)

    # n_values = [1, 5, 10, 20, 30, 40, 50]
    n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
    dists_cal, preds_cal = utils.find_nn(q_mu_cal, db_mu_cal, max(n_values))                     # override prediciton with larger k
    recall_at_k = cal_recall(preds_cal, positives_cal, n_values)
    recall_at_k = {n_values[i]: f'{recall_at_k[i]:.2f}' for i in range(len(n_values))}
    print(f'==> recall on VAL (len={q_mu_cal.shape[0]}):', recall_at_k)

    # TEST =============================== #
    with open(join(opt.resume, 'test_embeddings_best.pickle'), 'rb') as handle:
        q_mu_test = pickle.load(handle)
        db_mu_test = pickle.load(handle)
        q_sigma_test = pickle.load(handle)
        db_sigma_test = pickle.load(handle)
        preds_test = pickle.load(handle)
        dists_test = pickle.load(handle)
        positives_test = pickle.load(handle)
    print(f'==> check shape: q_mu_test={q_mu_test.shape}, db_mu_test={db_mu_test.shape}; q_sigma_test={q_sigma_test.shape}, db_sigma_test={db_sigma_test.shape}')
    if hyps['unc'] == 'mcd':
        q_sigma_test = q_sigma_test.mean(axis=1, keepdims=True)
        db_sigma_test = db_sigma_test.mean(axis=1, keepdims=True)

    dists_test, preds_test = utils.find_nn(q_mu_test, db_mu_test, max(n_values))                   # the results is sorted
    recall_at_k = cal_recall(preds_test, positives_test, n_values)
    recall_at_k = {n_values[i]: f'{recall_at_k[i]:.2f}' for i in range(len(n_values))}
    print(f'==> recall on TEST (len={q_mu_cal.shape[0]}):', recall_at_k)

    # preds = preds_test
    # gt = positives_test
    # q_sigma_sq = q_sigma_test

elif hyps['dbs'] in ['car196', 'cub200', 'chestx', 'sop']:

    '''
    1. Restore pretrained model paratmeters
    2. Load dataset
    3. Genrate embeddings, variances and labels
    4. Sanity check for the whole val set
    '''
    whole_train_set = dataset.Whole('train', data_path=opt.data_path, aug=True, return_label=True)
    whole_training_data_loader = DataLoader(dataset=whole_train_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=True, pin_memory=opt.cuda, worker_init_fn=seed_worker)
    whole_val_set = dataset.Whole('val', data_path=opt.data_path, aug=False, return_label=True)
    whole_val_data_loader = DataLoader(dataset=whole_val_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=opt.cuda, worker_init_fn=seed_worker)
    whole_test_set = dataset.Whole('test', data_path=opt.data_path, aug=False, return_label=True)
    whole_test_data_loader = DataLoader(dataset=whole_test_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=opt.cuda, worker_init_fn=seed_worker)

    with open(join(opt.resume, f'{opt.split}_embeddings_best.pickle'), 'rb') as handle:
        q_mu = pickle.load(handle)
        db_mu = pickle.load(handle)
        q_sigma_sq = pickle.load(handle)
        db_sigma_sq = pickle.load(handle)
        preds = pickle.load(handle)
        dists = pickle.load(handle)
        gt = pickle.load(handle)
    print(f'==> embeddings={q_mu.shape}, sigmas={q_sigma_sq.shape}')
    if hyps['unc'] == 'mcd':
        q_sigma_sq = q_sigma_sq.mean(axis=1, keepdims=True)
        db_sigma_sq = db_sigma_sq.mean(axis=1, keepdims=True)

    n_values = [1, 5, 10, 20, 30, 40, 50]
    # n_values = np.arange(1, 50, 1)
    dists, preds = utils.find_nn(q_mu, db_mu, max(n_values) + 1)
    dists = dists[:, 1:]                             # -1: to exclude query itself
    preds = preds[:, 1:]
    gt = whole_test_set.get_positives()

    # Load CAL and TEST set
    # Split the cal set from the val set
    n_cal = int(0.5 * q_mu.shape[0])
    n_test = q_mu.shape[0] - n_cal
    idx = np.array([1] * n_cal + [0] * (q_mu.shape[0] - n_cal)) > 0
    np.random.shuffle(idx)
    images_cal, images_test = whole_test_set.image_list[idx], whole_test_set.image_list[~idx]
    labels_cal, labels_test = whole_test_set.image_label[idx], whole_test_set.image_label[~idx]

    q_mu_cal, q_mu_test = q_mu[idx, :], q_mu[~idx, :]
    db_mu_cal, db_mu_test = db_mu[idx, :], db_mu[~idx, :]
    # labels_cal, labels_test = labels[idx], labels[~idx]
    q_sigma_cal, q_sigma_test = q_sigma_sq[idx, :], q_sigma_sq[~idx, :]
    db_sigma_cal, db_sigma_test = db_sigma_sq[idx, :], db_sigma_sq[~idx, :]
    # vars_cal, vars_test = variances[idx, :], variances[~idx, :]

    n_values = [1, 5, 10, 20, 30, 40, 50]
    # Check recall
    positives_cal = []
    for i, label in enumerate(labels_cal):
        if hyps['dbs'] in ['car196', 'cub200', 'sop']:
            positive = np.where(labels_cal == label)[0]                                                # find same-label samples
        elif hyps['dbs'] == 'chestx':
            positive = np.where(labels_cal & label)[0]                                                 # find same-label samples
        positive = np.delete(positive, np.where(positive == i)[0])                                     # delete self
        positives_cal.append(positive)
    dists_cal, preds_cal = utils.find_nn(q_mu_cal, db_mu_cal, max(n_values) + 1)
    dists_cal = dists_cal[:, 1:]                                                                       # -1: to exclude query itself
    preds_cal = preds_cal[:, 1:]
    recall_at_k_cal = cal_recall(preds_cal, positives_cal, n_values)
    recall_at_k_cal = {n_values[i]: f'{recall_at_k_cal[i]:.2f}' for i in range(len(n_values))}
    print(f'==> recall on VAL (len={q_mu_cal.shape[0]}):', recall_at_k_cal)


    positives_test = []
    for i, label in enumerate(labels_test):
        if hyps['dbs'] in ['car196', 'cub200', 'sop']:
            positive = np.where(labels_test == label)[0]                                               # find same-label samples
        elif hyps['dbs'] == 'chestx':
            positive = np.where(labels_test & label)[0]                                                # find same-label samples
        positive = np.delete(positive, np.where(positive == i)[0])                                     # delete self
        positives_test.append(positive)
    dists_test, preds_test = utils.find_nn(q_mu_test, db_mu_test, max(n_values) + 1)                   # +1 because query itself occupies a position                    # (N, n), (N, n) the results is sorted
    dists_test = dists_test[:, 1:]                                                                     # -1: to exclude query itself
    preds_test = preds_test[:, 1:]
    recall_at_k_test_ml = cal_recall(preds_test, positives_test, n_values)
    recall_at_k_test_ml = {n_values[i]: f'{recall_at_k_test_ml[i]:.2f}' for i in range(len(n_values))}
    print(f'==> recall on TEST (len={q_mu_test.shape[0]}):', recall_at_k_test_ml)


# RCIR
def parallel_evaluate(func, args):
    with Pool(processes=128) as pool:
        async_result = pool.starmap(func, args)
    return async_result


def loss_k(preds, gts, sigma, k_max, random=False):
    assert len(preds) == len(gts) == len(sigma), 'lengths of preds, gts, and sigma should be the same'
    num_query = len(preds)
    num_recall = 0
    k_pool = np.zeros(num_query)
    for i, pred in enumerate(tqdm(preds, leave=False)):
        unc_level = (sigma[i] - sigma.min()) / (sigma.max() - sigma.min())
        if random:
            ki = np.random.choice(np.arange(1, k_max + 1))
        else:
            ki = int(np.ceil(unc_level * k_max))
        if np.in1d(pred[:ki], gts[i]).sum() > 0:
            num_recall += 1
        k_pool[i] = ki
    recall_rate = num_recall / num_query
    return k_pool, recall_rate


SANITITY_CHECK = False
if SANITITY_CHECK ==True:
    args = [[preds, gt, q_sigma_sq, k_, True] for k_ in range(1, 40, 1)]
    recall_atk_random = np.array([(x.mean(), y) for x, y, in parallel_evaluate(loss_k, args)])
    args = [[preds, gt, q_sigma_sq, k_, False] for k_ in range(1, 30, 1)]
    recall_atk_ada = np.array([(x.mean(), y) for x, y, in parallel_evaluate(loss_k, args)])
    recall_atk_fix = cal_recall(preds, gt, np.arange(1, 20, 1)) / 100.0

    n_row, n_col = 1, 1
    fig, axs = plt.subplots(n_row, n_col, figsize=(5 * n_col, 5 * n_row), dpi=200, squeeze=False)
    axs = axs.ravel()
    ax = axs[0]
    ax.plot(np.arange(1, 20, 1), recall_atk_fix, marker='o', markersize=2, label='Fixed K')
    ax.plot(recall_atk_random[:, 0], recall_atk_random[:, 1], marker='o', markersize=3, label='Random K')
    ax.plot(recall_atk_ada[:, 0], recall_atk_ada[:, 1], marker='o', markersize=2, label='Adaptive K')
    ax.set_xlabel('$K$')
    ax.set_ylabel('Recall@$K$')
    matin.ax_default_style(ax, font_size=12, show_legend=True, show_grid=True)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylim([0.40, 1.00])


def calibrate_kappa(preds, positives, q_sigma, d_kappa=0.2, alpha=0.2, delta=0.1):
    n = len(preds)
    n_pred = len(preds[0])
    kappa = 1
    ucb = 1
    while ucb > alpha:
        kappa += d_kappa
        k_pool, recall, = loss_k(preds, positives, q_sigma, k_max=kappa)
        ucb = 1 - recall + np.sqrt(np.log(1 / delta) / (2 * n))
        if kappa > n_pred:
            break
    return kappa

def calibrate_kappa_parallel(preds, positives, q_sigma, d_kappa=0.2, alpha=0.2, delta=0.1):
    n = len(preds)
    n_pred = len(preds[0])
    kappa = 1
    ucb = 1
    kappa_list = np.arange(1, n_pred, d_kappa)
    args = [(preds, positives, q_sigma, kappa) for kappa in kappa_list]
    results = parallel_evaluate(loss_k, args)
    k_pool_list, recall_list = zip(*results)
    ucb_list = 1 - np.array(recall_list) + np.sqrt(np.log(1 / delta) / (2 * n))
    if ucb_list.min() > alpha:
        return kappa_list.max()
    else:
        return kappa_list[np.where(ucb_list < alpha)[0][0]]

# k1 = calibrate_kappa(preds_cal, positives_cal, q_sigma_cal, d_kappa=0.2, alpha=0.1, delta=0.1)
# k2 = calibrate_kappa_parallel(preds_cal, positives_cal, q_sigma_cal, d_kappa=0.2, alpha=0.1, delta=0.1)


#%% EXP1: Risk vs. Different Alpha + Fixed Delta

delta_test = 0.1
alphas_test = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90]

# slow
def infer(preds_cal, positives_cal, q_sigma_cal, preds_test, positives_test, q_sigma_test, alpha, delta=0.1):
    kappa = calibrate_kappa(preds_cal, positives_cal, q_sigma_cal, d_kappa=0.2, alpha=alpha, delta=delta)
    k_pool_test, recall_test = loss_k(preds_test, positives_test, q_sigma_test, kappa)
    return k_pool_test, recall_test
args = [(preds_cal, positives_cal, q_sigma_cal, preds_test, positives_test, q_sigma_test, alpha, delta_test) for alpha in alphas_test]
results = parallel_evaluate(infer, args)
k_pool_test, recalls_test = zip(*results)

## fast - Not verfied
# def infer_parallel(preds_cal, positives_cal, q_sigma_cal, preds_test, positives_test, q_sigma_test, alpha, delta=0.1):
#     kappa = calibrate_kappa_parallel(preds_cal, positives_cal, q_sigma_cal, d_kappa=0.2, alpha=alpha, delta=delta)
#     k_pool_test, recall_test = loss_k(preds_test, positives_test, q_sigma_test, kappa)
#     return k_pool_test, recall_test
# results = [infer_parallel(preds_cal, positives_cal, q_sigma_cal, preds_test, positives_test, q_sigma_test, alpha, delta_test) for alpha in alphas_test]
# k_pool_test, recalls_test = zip(*results)

recall_k = cal_recall(preds_test, positives_test, np.arange(1, 30, 1)) / 100.0

# save results to disk
OUTPUT_DIR = "figures_rcir"
np.save(os.path.join(OUTPUT_DIR, f'k_pool_test_{hyps["dbs"]}.npy'), k_pool_test)
np.save(os.path.join(OUTPUT_DIR, f'recalls_test_{hyps["dbs"]}.npy'), recalls_test)
np.save(os.path.join(OUTPUT_DIR, f'recall_k_{hyps["dbs"]}.npy'), recall_k)

#%% EXP2: Risk vs. Different Alpha + Different Delta

alphas_test_2d = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90]
delta_test_2d = [0.001, 0.01, 0.1]
risks = [(x, y) for x in alphas_test_2d for y in delta_test_2d]
args = [(preds_cal, positives_cal, q_sigma_cal, preds_test, positives_test, q_sigma_test, risk[0], risk[1]) for risk in risks]
results = parallel_evaluate(infer, args)
k_pool_test, recalls_test = zip(*results)

# save results to disk
OUTPUT_DIR = "figures_rcir"
np.save(os.path.join(OUTPUT_DIR, f'k_pool_test_{hyps["dbs"]}_delta.npy'), k_pool_test)
np.save(os.path.join(OUTPUT_DIR, f'recalls_test_{hyps["dbs"]}_delta.npy'), recalls_test)
