#%%
# =================== imports ===================
import ipdb
import argparse
import importlib
import json
import os
import pickle
import shutil
import sys
from glob import glob
from multiprocessing import Pool
from os.path import abspath, dirname, exists, join
from rich import print
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.progress import Progress
import random
import pickle

import utils.utils as utils
from utils.cp_utils import cal_recall

np.set_printoptions(precision=3, suppress=True)
plt.style.use('ggplot')

parser = argparse.ArgumentParser(description="Options")
parser.add_argument('--dbs', type=str, default='cub200', help='choose dataset', choices=['pitts', 'cub200', 'car196', 'chestx', 'sop'])
parser.add_argument('--unc', type=str, default='btl', help='choose method', choices=['triplet', 'btl', 'mcd', 'ensemble', 'random'])
parser.add_argument('--cnt', type=int, default=0, help='set seed')
parser.add_argument('--basic', action='store_true')                                                # basic: Risk with different alpha and fixed delta
parser.add_argument('--delta', action='store_true')                                                # delta: Risk with different alpha and different delta
parser.add_argument('--ece', action='store_true')
parser.add_argument('--heu_unc', action='store_true')                                              # heu_unc: average K vs. recall
parser.add_argument('--qua', action='store_true')
parser.add_argument('--bound', action='store_true')                                                # bound: visulization of rho and rho_plus
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--random', action='store_true')                                           # random: random uncertainty guess
parser.add_argument('--constant', action='store_true')                                           # random: random uncertainty guess
opts = parser.parse_args()

hyps = {'dbs': None, 'unc': None}
if sys.argv == ['']:
    hyps['dbs'] = 'cub200'                       # 'cub200' | 'car196' | 'pitts'| 'chestx' | 'sop'
    hyps['unc'] = 'btl'                          # 'btl' | 'triplet' | 'mcd' | 'ensemble'|
else:
    hyps['dbs'] = opts.dbs                       # 'cub200' | 'car196' | 'pitts'| 'chestx'
    hyps['unc'] = opts.unc                       # 'btl' | 'dul' | 'mcd' | 'ensemble'|

print(f"{list(hyps.values())}")
def get_resume(dbs, unc):
    dicts = pd.read_json('baselines_beta.json')
    logs_dir = dicts['logs_dir'][0]
    if unc == 'ensemble':
        exp_folder = []
        resume_folder = []
        hparams_file = []
        for i in range(len(dicts[dbs][unc])):
            exp_folder = join(logs_dir, dicts[dbs][unc][i])
            resume_folder.append(glob(join(exp_folder, 'RCIR','*', 'checkpoints'))[0])
            hparams_file.append(glob(join(exp_folder, 'train_csv', '*', 'hparams.yaml'))[0])
        with open(hparams_file[0]) as f:
            cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    elif unc in ['btl', 'mcd', 'triplet']:
        exp_folder = join(logs_dir, dicts[dbs][unc])
        resume_folder = glob(join(exp_folder, 'RCIR','*', 'checkpoints'))[0]
        hparams_file = glob(join(exp_folder, 'train_csv', '*', 'hparams.yaml'))[0]
        with open(hparams_file) as f:
            cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    return resume_folder, cfg


resume, cfg = get_resume(hyps['dbs'], hyps['unc'])
dataset = importlib.import_module(f"datasets.{hyps['dbs']}")

worker_seed = opts.cnt
np.random.seed(worker_seed)
random.seed(worker_seed)

OUTPUT_DIR = f"figures/rcir/{hyps['unc']}"
if opts.heu_unc:
    OUTPUT_DIR += '_heu'
if opts.qua or opts.bound:
    OUTPUT_DIR += '_qua'
if opts.random:
    OUTPUT_DIR += '_random'
if opts.constant:
    OUTPUT_DIR += '_constant'

print(f"{resume}")
print(f"OUTPUT_DIR={OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#%%

# Load embeddings/predictions
if hyps['dbs'] in ['pitts']:
    split_method = 'on_test' # 'on_test' | 'seperate'
    # whole_val_set = dataset.Whole('val', data_path=cfg.dataset.data_path)
    # whole_test_set = dataset.Whole('test', data_path=cfg.dataset.data_path)
    # print(f"val: db={whole_val_set.get_databases().shape}, q={whole_val_set.get_queries().shape}")
    # print(f"test: db={whole_test_set.get_databases().shape}, q={whole_test_set.get_queries().shape}")

    if hyps['unc'] == 'ensemble':
        q_mu_list = []
        db_mu_list = []
        for i in range(5):
            with open(join(resume[i], f'embeddings_test.pickle'), 'rb') as handle:
                q_mu = pickle.load(handle)
                db_mu = pickle.load(handle)
                _ = pickle.load(handle)
                _ = pickle.load(handle)
                _ = pickle.load(handle)
                _ = pickle.load(handle)
                positives_test = pickle.load(handle)
            q_mu_list.append(q_mu)
            db_mu_list.append(db_mu)
        q_mu_list = np.array(q_mu_list)
        db_mu_list = np.array(db_mu_list)
        q_mu_test = q_mu_list.mean(axis=0)
        db_mu_test = db_mu_list.mean(axis=0)
        q_sigma_test = q_mu_list.var(axis=0)
        db_sigma_test = db_mu_list.var(axis=0)
    elif hyps['unc'] in ['btl', 'mcd', 'triplet']:
        # TEST =============================== #
        with open(join(resume, 'embeddings_test.pickle'), 'rb') as handle:
            q_mu_test = pickle.load(handle)
            db_mu_test = pickle.load(handle)
            q_sigma_test = pickle.load(handle)
            db_sigma_test = pickle.load(handle)
            preds_test = pickle.load(handle)
            dists_test = pickle.load(handle)
            positives_test = pickle.load(handle)

    # print(f'check shape: q_mu_test={q_mu_test.shape}, db_mu_test={db_mu_test.shape}; q_sigma_test={q_sigma_test.shape}, db_sigma_test={db_sigma_test.shape}')
    if hyps['unc'] in ['mcd', 'ensemble']:
        q_sigma_test = q_sigma_test.mean(axis=1, keepdims=True)
        db_sigma_test = db_sigma_test.mean(axis=1, keepdims=True)
    n_values = [1, 5, 10, 20, 30, 40, 50]
    # n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
    dists_test, preds_test = utils.find_nn(q_mu_test, db_mu_test, max(n_values))                   # the results is sorted
    recall_at_k = cal_recall(preds_test, positives_test, n_values)
    # recall_at_k = {n_values[i]: f'{recall_at_k[i]:.2f}' for i in range(len(n_values))}
    print(f'recall on TEST (len={q_mu_test.shape[0]}):', recall_at_k)

    if split_method == 'seperate':
        # VAL ================================ #
        with open(join(resume, 'embeddings_val.pickle'), 'rb') as handle:
            q_mu_cal = pickle.load(handle)
            db_mu_cal = pickle.load(handle)
            q_sigma_cal = pickle.load(handle)
            db_sigma_cal = pickle.load(handle)
            preds_cal = pickle.load(handle)
            dists_cal = pickle.load(handle)
            positives_cal = pickle.load(handle)
        # print(f'check shape: q_mu={q_mu_cal.shape}, db_mu_val={db_mu_cal.shape}; q_sigma_val={q_sigma_cal.shape}, db_sigma_val={db_sigma_cal.shape}')
        if hyps['unc'] in ['mcd', 'ensemble']:
            q_sigma_cal = q_sigma_cal.mean(axis=1, keepdims=True)
            db_sigma_cal = db_sigma_cal.mean(axis=1, keepdims=True)

        dists_cal, preds_cal = utils.find_nn(q_mu_cal, db_mu_cal, max(n_values))                     # override prediciton with larger k
        recall_at_k = cal_recall(preds_cal, positives_cal, n_values)
        # recall_at_k = {n_values[i]: f'{recall_at_k[i]:.2f}' for i in range(len(n_values))}
        print(f'recall on VAL (len={q_mu_cal.shape[0]}):', recall_at_k)

    elif split_method == 'on_test':
        n_total = q_mu_test.shape[0]
        n_cal = int(0.5 * n_total)
        n_test = n_total - n_cal
        idx = np.array([1] * n_cal + [0] * (n_total - n_cal)) > 0
        np.random.shuffle(idx)
        q_mu_cal = q_mu_test[idx]
        db_mu_cal = db_mu_test
        q_sigma_cal = q_sigma_test[idx]
        db_sigma_cal = db_sigma_test
        preds_cal = preds_test[idx]
        dists_cal = dists_test[idx]
        positives_cal = positives_test[idx]
        q_mu_test = q_mu_test[~idx]
        db_mu_test = db_mu_test
        q_sigma_test = q_sigma_test[~idx]
        db_sigma_test = db_sigma_test
        preds_test = preds_test[~idx]
        dists_test = dists_test[~idx]
        positives_test = positives_test[~idx]

        # dists_test, preds_test = utils.find_nn(q_mu_test, db_mu_test, max(n_values))                   # the results is sorted
        recall_at_k = cal_recall(preds_cal, positives_cal, n_values)
        # recall_at_k = {n_values[i]: f'{recall_at_k[i]:.2f}' for i in range(len(n_values))}
        print(f'recall on VAL (len={q_mu_cal.shape[0]}):', recall_at_k)
        recall_at_k = cal_recall(preds_test, positives_test, n_values)
        # recall_at_k = {n_values[i]: f'{recall_at_k[i]:.2f}' for i in range(len(n_values))}
        print(f'recall on TEST (len={q_mu_test.shape[0]}):', recall_at_k)


elif hyps['dbs'] in ['car196', 'cub200', 'chestx', 'sop']:

    '''
    1. Restore pretrained model paratmeters
    2. Load dataset
    3. Genrate embeddings, variances and labels
    4. Sanity check for the whole val set
    '''
    whole_test_set = dataset.Whole('test', data_path=cfg.dataset.data_path, aug=False, return_label=True)

    if hyps['unc'] == 'ensemble':
        q_mu_list = []
        db_mu_list = []
        for i in range(len(resume)):
            with open(join(resume[i], f'embeddings_test.pickle'), 'rb') as handle:
                q_mu = pickle.load(handle)
                db_mu = pickle.load(handle)
            q_mu_list.append(q_mu)
            db_mu_list.append(db_mu)
        q_mu_list = np.array(q_mu_list)
        db_mu_list = np.array(db_mu_list)
        q_mu = q_mu_list.mean(axis=0)
        db_mu = db_mu_list.mean(axis=0)
        q_variance = q_mu_list.var(axis=0)
        db_variance = db_mu_list.var(axis=0)
    elif hyps['unc'] in ['mcd', 'btl', 'triplet']:
        with open(join(resume, f'embeddings_test.pickle'), 'rb') as handle:
            q_mu = pickle.load(handle)
            db_mu = pickle.load(handle)
            q_variance = pickle.load(handle)
            db_variance = pickle.load(handle)
            preds = pickle.load(handle)
            dists = pickle.load(handle)
            gt = pickle.load(handle)

    q_sigma_sq = q_variance
    db_sigma_sq = db_variance
    print(f'embeddings={q_mu.shape}, sigmas={q_sigma_sq.shape}')
    if hyps['unc'] in ['mcd', 'ensemble']:
        q_sigma_sq = q_sigma_sq.mean(axis=1, keepdims=True)
        db_sigma_sq = db_sigma_sq.mean(axis=1, keepdims=True)
    # Load CAL and TEST set
    # Split the cal set from the val set
    n_total = q_mu.shape[0]
    n_cal = int(0.5 * n_total)
    n_test = n_total - n_cal
    idx = np.array([1] * n_cal + [0] * (n_total - n_cal)) > 0
    np.random.shuffle(idx)
    images_cal, images_test = whole_test_set.image_list[idx], whole_test_set.image_list[~idx]
    labels_cal, labels_test = whole_test_set.image_label[idx], whole_test_set.image_label[~idx]

    q_mu_cal, q_mu_test = q_mu[idx, :], q_mu[~idx, :]
    db_mu_cal, db_mu_test = db_mu[idx, :], db_mu[~idx, :]
    q_sigma_cal, q_sigma_test = q_sigma_sq[idx, :], q_sigma_sq[~idx, :]
    db_sigma_cal, db_sigma_test = db_sigma_sq[idx, :], db_sigma_sq[~idx, :]

    if hyps['dbs'] in ['sop']:
        n_values = [1, 10, 50, 100, 150, 200, 250]
    elif hyps['dbs'] in ['cub200']:
        n_values = [1, 5, 10, 20, 30, 40, 50, 60, 70]
    elif hyps['dbs'] in ['car196', 'chestx']:
        n_values = [1, 5, 10, 20, 30, 40, 50]

    print(n_values)
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
    # recall_at_k_cal = {n_values[i]: f'{recall_at_k_cal[i]:.2f}' for i in range(len(n_values))}
    print(f'recall on VAL (len={q_mu_cal.shape[0]}):', recall_at_k_cal)


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
    # recall_at_k_test_ml = {n_values[i]: f'{recall_at_k_test_ml[i]:.2f}' for i in range(len(n_values))}
    print(f'recall on TEST (len={q_mu_test.shape[0]}):', recall_at_k_test_ml)
    distributional_diffs = np.abs(recall_at_k_cal[0] - recall_at_k_test_ml[0])
    print(f'diffs={distributional_diffs:.3f}')

# ipdb.set_trace()
sigma_profile_cal = np.min([q_sigma_cal.min(), db_sigma_cal.min()]), np.max([q_sigma_cal.max(), db_sigma_cal.max()])

if opts.tsne == True:
    import pickle
    with open(join(OUTPUT_DIR, f'vis_tsne_{hyps["dbs"]}_{opts.cnt}.pickle'), 'wb') as handle:
        pickle.dump(q_mu, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(whole_test_set.image_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(whole_test_set.image_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%

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
        if random == True:
            ki = int(np.random.choice(np.arange(1, k_max + 1)))
        else:
            ki = int(np.ceil(unc_level[0] * k_max))
        if np.in1d(pred[:ki], gts[i]).sum() > 0:
            num_recall += 1
        k_pool[i] = ki
    recall_rate = num_recall / num_query
    return k_pool, recall_rate

def compute_rho(preds, gts, sigma, k_max, random=False):
    _, recall_avg = loss_k(preds, gts, sigma, k_max, random)
    rho = 1 - recall_avg
    return rho

def compute_rho_plus(preds, gts, sigma, k_max, random=False, delta=0.1):
    rho = compute_rho(preds, gts, sigma, k_max, random)
    rho_plus_ = rho + np.sqrt(np.log(1 / delta) / (2 * len(preds)))
    return rho, rho_plus_


def calibrate_kappa(preds, positives, q_sigma, d_kappa=0.2, alpha=0.2, delta=0.1, random=False):
    n = len(preds)
    n_pred = len(preds[0])
    kappa = 1
    ucb = 1
    while ucb > alpha:
        kappa += d_kappa
        k_pool, recall, = loss_k(preds, positives, q_sigma, k_max=kappa, random=random)
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


def infer(preds_cal, positives_cal, q_sigma_cal, preds_test, positives_test, q_sigma_test, alpha, delta=0.1, random=False):
    kappa = calibrate_kappa(preds_cal, positives_cal, q_sigma_cal, d_kappa=0.2, alpha=alpha, delta=delta, random=random)
    k_pool_test, recall_test = loss_k(preds_test, positives_test, q_sigma_test, kappa, random=random)
    return k_pool_test, recall_test

def locate_var_level(var_stat, var):
    '''
    given the maximum and minimum value of var, decide the percentile of a new var
    '''
    var_min, var_max = var_stat
    if var < var_min:
        return 0
    elif var > var_max:
        return 1
    else:
        return (var - var_min) / (var_max - var_min)

def uncertainty_recall(preds, positives, sigma_profile_cal, q_sigma, db_sigma, alpha, k_max):
    num_recall = 0
    k_pool = []
    unc_level_pool = []
    preds_new = []
    for i, pred in enumerate(tqdm(preds, leave=False)):
        pred_refine = []
        unc_level_ = []
        for j in pred[:k_max]:
            unc_level = locate_var_level(sigma_profile_cal, (q_sigma[i] + db_sigma[j]) / 2)
            if unc_level < alpha:
                pred_refine.append(j)
                unc_level_.append(unc_level)
        pred_refine = np.array(pred_refine)
        k_pool.append(len(pred_refine))
        unc_level_pool.append(unc_level_)
        preds_new.append(pred_refine)
        if np.in1d(pred_refine, positives[i]).sum() > 0:
            num_recall += 1
    recall = num_recall / len(preds)
    k_pool = np.array(k_pool)
    return k_pool, recall, unc_level_pool, preds_new


with open("rcir_paras.yaml", 'r') as stream:
    data = yaml.safe_load(stream)
    delta_test = data['delta_test']
    alphas_test = data['alphas_test']
    delta_test_2d = data['delta_test_2d']
    alphas_test_2d = data['alphas_test_2d']

#%%

if hyps['unc'] == 'triplet':
    recall_k = cal_recall(preds_test, positives_test, np.arange(1, 30, 1)) / 100.0
    np.save(os.path.join(OUTPUT_DIR, f'recall_k_{hyps["dbs"]}_{opts.cnt}.npy'), recall_k)

#%% ece: visualizaiton of calibration
if opts.ece == True:
    import pickle
    with open(join(OUTPUT_DIR, f'ece_{hyps["dbs"]}_{opts.cnt}.pickle'), 'wb') as handle:
        pickle.dump(preds_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_sigma_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(positives_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% qua: visualizaiton of retrieval
if opts.qua == True:
    delta_test_qua = 0.1
    alphas_test_qua = 0.2
    print('Experiment Qualitative')
    print(f"delta: {delta_test_qua}")
    print(f"alphas: {alphas_test_qua}")

    k_pool_test, recall_test, unc_level_pool, preds_new = uncertainty_recall(preds_test, positives_test, sigma_profile_cal, q_sigma_test, db_sigma_test, alphas_test_qua, len(preds_test[0]))
    with open(os.path.join(OUTPUT_DIR, f'h_{hyps["dbs"]}_{hyps["unc"]}_{opts.cnt}.pickle'), 'wb') as handle:
        pickle.dump(k_pool_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(recall_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(preds_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(positives_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(unc_level_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)

    kappa = calibrate_kappa(preds_cal, positives_cal, q_sigma_cal, d_kappa=0.2, alpha=alphas_test_qua, delta=delta_test_qua)
    k_pool_test, recall_test = loss_k(preds_test, positives_test, q_sigma_test, kappa)
    with open(os.path.join(OUTPUT_DIR, f'rg_{hyps["dbs"]}_{hyps["unc"]}_{opts.cnt}.pickle'), 'wb') as handle:
        pickle.dump(k_pool_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(recall_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(preds_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(positives_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(images_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(labels_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% bound: visulization of rho and rho_plus
if opts.bound == True:
    # loss curve
    # kappa_list = []
    # loss_list = []
    # for kappa in np.arange(0, 50, 0.1):
    #     k_pool, recall = loss_k(preds_cal, positives_cal, q_sigma_cal, k_max=kappa)
    #     kappa_list.append(kappa)
    #     loss_list.append(1 - recall)
    # kappa_list = np.array(kappa_list)
    # loss_list = np.array(loss_list)

    kappa_list = np.arange(0, 50, 0.1)
    args = [(preds_cal, positives_cal, q_sigma_cal, kappa, False, 0.1) for kappa in kappa_list]
    results = parallel_evaluate(compute_rho_plus, args)
    rho_list, rho_plus_list = zip(*results)
    # compute_rho_plus(preds_cal, positives_cal, q_sigma_cal, 5, 0.1)
    np.save(os.path.join(OUTPUT_DIR, f'kappa_list.npy'), kappa_list)
    np.save(os.path.join(OUTPUT_DIR, f'rho.npy'), rho_list)
    np.save(os.path.join(OUTPUT_DIR, f'rho_plus.npy'), rho_plus_list)

#%% basic: Risk vs. Different Alpha + Fixed Delta
if opts.basic == True:
    print('Experiment 1: Risk vs. Different Alpha + Fixed Delta')
    print(f"delta:{delta_test}")
    print(f"alphas:{alphas_test}")

    # ipdb.set_trace()
    if opts.random == True:
        q_sigma_cal = np.random.rand(q_sigma_cal.shape[0], 1)
        q_sigma_test = np.random.rand(q_sigma_test.shape[0], 1)
    if opts.constant == True:
        q_sigma_cal = np.ones_like(q_sigma_cal)
        q_sigma_test = np.ones_like(q_sigma_test)
    args = [(preds_cal, positives_cal, q_sigma_cal, preds_test, positives_test, q_sigma_test, alpha, delta_test, False) for alpha in alphas_test]
    results = parallel_evaluate(infer, args)
    k_pool_test, recalls_test = zip(*results)

    recall_k = cal_recall(preds_test, positives_test, np.arange(1, 30, 1)) / 100.0

    # save results to disk
    np.save(os.path.join(OUTPUT_DIR, f'k_pool_test_{hyps["dbs"]}_{opts.cnt}.npy'), k_pool_test)
    np.save(os.path.join(OUTPUT_DIR, f'recalls_test_{hyps["dbs"]}_{opts.cnt}.npy'), recalls_test)
    np.save(os.path.join(OUTPUT_DIR, f'recall_k_{hyps["dbs"]}_{opts.cnt}.npy'), recall_k)


#%% delta: Risk vs. Different Alpha + Different Delta
if opts.delta == True:
    print('Experiment 1: Risk vs. Different Alpha + Fixed Delta')
    print(f"delta_test_2d:{delta_test_2d}")
    print(f"alphas_test_2d:{alphas_test}")

    risks = [(x, y) for x in alphas_test_2d for y in delta_test_2d]
    args = [(preds_cal, positives_cal, q_sigma_cal, preds_test, positives_test, q_sigma_test, risk[0], risk[1]) for risk in risks]
    results = parallel_evaluate(infer, args)
    k_pool_test, recalls_test = zip(*results)

    # save results to disk
    np.save(os.path.join(OUTPUT_DIR, f'k_pool_test_{hyps["dbs"]}_delta_{opts.cnt}.npy'), k_pool_test)
    np.save(os.path.join(OUTPUT_DIR, f'recalls_test_{hyps["dbs"]}_delta_{opts.cnt}.npy'), recalls_test)

#%% heu_unc: average K vs. recall
if opts.heu_unc:
    # recall, k = uncertainty_recall(preds_test, positives_test, sigma_profile_cal, q_sigma_test, db_sigma_test, 0.3, len(preds_test[0]))
    args = [(preds_test, positives_test, sigma_profile_cal, q_sigma_test, db_sigma_test, alpha, len(preds_test[0])) for alpha in alphas_test]
    results = parallel_evaluate(uncertainty_recall, args)
    k_pool_test, recalls_test, _, _ = zip(*results)

    # save results to disk
    np.save(os.path.join(OUTPUT_DIR, f'k_pool_test_{hyps["dbs"]}_{opts.cnt}.npy'), k_pool_test)
    np.save(os.path.join(OUTPUT_DIR, f'recalls_test_{hyps["dbs"]}_{opts.cnt}.npy'), recalls_test)
