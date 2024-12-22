import importlib
import os
import shutil
from collections import namedtuple
from os.path import dirname, exists, join

import faiss
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, sqeuclidean, squareform
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def compute_ece(preds, variances, gt):
    '''return: bins_recall: (10, 3)
                ece_bins_recall: (1, 3)    
    '''
    assert len(preds) == len(variances) == len(gt), 'length of preds, variances and gt should be the same :('
    num_bins = 11
    n_values = [1, 5, 10]
    if variances.shape[-1] != 1:
        variances = variances.mean(axis=-1)
    variance_reduced = variances
    indices, _ = get_bins(variance_reduced, num_bins)
    # indices, _, k = utils.get_zoomed_bins(variance_reduced, num_bins)
    bins_recall = np.zeros((num_bins - 1, len(n_values)))
    ece_bins_recall = np.zeros((num_bins - 1, len(n_values)))
    for index in range(num_bins - 1):
        if len(indices[index]) == 0:
            continue
        pred_bin = preds[indices[index]]
        gt_bin = [gt[i] for i in indices[index]]
        # calculate r@K
        recall_at_n = cal_recall(pred_bin, gt_bin, n_values)
        bins_recall[index] = recall_at_n
        ece_bins_recall[index] = np.array([len(indices[index]) / variance_reduced.shape[0] * np.abs(recall_at_n[i] - (num_bins - 1 - index) / ((num_bins - 1))) for i in range(len(n_values))], )
    return bins_recall, ece_bins_recall.sum(axis=0), np.array([len(x) for x in indices])


def compute_ece_cosine(preds, gt, q_mu, db_mu):
    '''return: bins_recall: (10, 3)
                ece_bins_recall: (1, 3)    
    '''
    assert len(preds) == len(gt) == len(q_mu), 'length of preds and gt should be the same :('

    q_cosine_matrix = 1 - q_mu @ db_mu.T
    variance_reduced = q_cosine_matrix[np.arange(q_cosine_matrix.shape[0]), preds[:, 0]]

    num_bins = 11
    n_values = [1, 5, 10]
    indices, _ = get_bins(variance_reduced, num_bins)
    bins_recall = np.zeros((num_bins - 1, len(n_values)))
    ece_bins_recall = np.zeros((num_bins - 1, len(n_values)))
    for index in range(num_bins - 1):
        if len(indices[index]) == 0:
            continue
        pred_bin = preds[indices[index]]
        gt_bin = [gt[i] for i in indices[index]]
        # calculate r@K
        recall_at_n = cal_recall(pred_bin, gt_bin, n_values)
        bins_recall[index] = recall_at_n
        ece_bins_recall[index] = np.array([len(indices[index]) / variance_reduced.shape[0] * np.abs(recall_at_n[i] - (num_bins - 1 - index) / ((num_bins - 1))) for i in range(len(n_values))], )
    return bins_recall, ece_bins_recall.sum(axis=0), np.array([len(x) for x in indices])


def schedule_device():
    info_per_card = os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split('\n')         # ['40536, 9948', '40536, 18191', '40536, 14492', '']

    card_memory_used = []
    for i in range(len(info_per_card)):
        if info_per_card[i] == '':
            continue
        else:
            total, used = int(info_per_card[i].split(',')[0]), int(info_per_card[i].split(',')[1])
            # print('Total GPU mem:', total, 'used:', used)
            card_memory_used.append(used)
    # print(card_memory_used.index(min(card_memory_used)))
    return int(card_memory_used.index(min(card_memory_used)))

def flatten_dict(nested_dict):
    flat_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flat_subdict = flatten_dict(value)
            flat_dict.update({f"{subkey}": subvalue for subkey, subvalue in flat_subdict.items()})
        else:
            flat_dict[key] = value
    return flat_dict


def find_nn(q_mu, db_mu, num_nn, device=''):
    """
    retrieve by L2 distance:||x-y||^2
    """
    if device == '':
        if q_mu.shape[0] >= 10000:
            # print('set device to gpu as size > 10000')
            device = 'gpu'
        else:
            device = 'cpu'

    if device == 'gpu':
        # print('Using GPU')
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 2
        faiss_index = faiss.GpuIndexFlatL2(res, q_mu.shape[1], flat_config)
    elif device == 'cpu':
        # print('Using GPU')
        faiss_index = faiss.IndexFlatL2(q_mu.shape[1])
    faiss_index.add(db_mu)
    dists, preds = faiss_index.search(q_mu, num_nn)
    return dists, preds

# install skimage using conda
# conda install -c conda-forge scikit-image
def load_cir_v2(hyps):
    dicts = pd.read_json('baselines.json')
    logs_dir = 'logs'
    resume = join(logs_dir, dicts[hyps['dbs']][hyps['score']][hyps['unc']])
    dataset = importlib.import_module(f"{resume.replace('/', '.')}.models.{hyps['dbs']}")
    return resume, dataset


def load_cir(hyps):
    if hyps['dbs'] == 'cub200':
        import datasets.cub200 as dataset
        if hyps['unc'] == 'btl':
            resume = 'logs/cub/0202_223209/ckpt_best.pth.tar'
        elif hyps['unc'] == 'dul':
            resume = 'logs/cub/0211_213116/ckpt_best.pth.tar'                                      # 1e-4
        elif hyps['unc'] == 'mcd':
            resume = 'logs/cub/0214_150330/ckpt_best.pth.tar'                                      # 0.1
    elif hyps['dbs'] == 'car196':
        import datasets.car196 as dataset
        if hyps['unc'] == 'btl':
            resume = 'logs/car/0202_214552/ckpt_best.pth.tar'
        elif hyps['unc'] == 'dul':
            resume = 'logs/car/0216_094113/ckpt_best.pth.tar'
        elif hyps['unc'] == 'mcd':
            resume = 'logs/car/0216_094224/ckpt_best.pth.tar'
    elif hyps['dbs'] == 'pitts':
        import datasets.pitts as dataset
        if hyps['unc'] == 'btl':
            resume = 'logs/pitts/0203_171233/ckpt_best.pth.tar'
        elif hyps['unc'] == 'dul':
            resume = 'logs/pitts/0216_152848/ckpt_best.pth.tar'
        elif hyps['unc'] == 'mcd':
            resume = 'logs/pitts/0216_152902/ckpt_best.pth.tar'
    else:
        print(hyps)
        raise NotImplementedError

    return resume, dataset

def load_resume(dataset, network, loss):
    resume = ''
    if dataset == 'cub200':
        if network == 'btl':
            resume = 'logs/cub/0202_223209/ckpt_best.pth.tar'
        elif network == 'dul':
            resume = 'logs/cub/0213_202429/ckpt_best.pth.tar'                                      # 1e-6
            resume = 'logs/cub/0213_194101/ckpt_best.pth.tar'                                      # 1e-5
            resume = 'logs/cub/0211_213116/ckpt_best.pth.tar'                                      # 1e-4
            resume = 'logs/cub/0211_213102/ckpt_best.pth.tar'                                      # 1e-3
        elif network == 'mcd':
            resume = 'logs/cub/0214_150330/ckpt_best.pth.tar'
    elif dataset == 'car196':
        if network == 'btl':
            resume = 'logs/car/0202_214552/ckpt_best.pth.tar'
        elif network == 'dul':
            resume = 'logs/car/0216_094113/ckpt_best.pth.tar'
        elif network == 'mcd':
            resume = ''
    elif dataset == 'pitts':
        if network == 'btl':
            resume = 'logs/pitts/0203_171233/ckpt_best.pth.tar'
        elif network == 'dul':
            resume = ''
        elif network == 'mcd':
            resume = ''
    else:
        raise NameError('undefined dataset')
    print(resume)
    return resume


def linear_fit(x, y, w, report_error=False):
    def cost(p, x, y, w):
        k = p[0]
        b = p[1]
        error = y - (k * x + b)
        error *= w
        return error

    p_init = np.array([-1, 1])
    ret = least_squares(cost, p_init, args=(x, y, w), verbose=0)
    # print(ret['x'][0], ret['x'][1], )
    y_fitted = ret['x'][0] * x + ret['x'][1]
    error = ret['cost']
    if report_error:
        return y_fitted, error
    else:
        return y_fitted


def reduce_sigma(sigma, std_or_sq, log_or_linear, hmean_or_mean):
    ''' 
    input sigma: sigma^2, ([1, D])
    output sigma: sigma, (1)
    '''
    if log_or_linear == 'log':
        print('log')
        sigma = np.log(sigma)
    elif log_or_linear == 'linear':
        pass
    else:
        raise NameError('undefined')

    if std_or_sq == 'std':
        sigma = np.sqrt(sigma)
    elif std_or_sq == 'sq':
        pass
    else:
        raise NameError('undefined')

    if hmean_or_mean == 'hmean':
        sigma = stats.hmean(sigma, axis=1)       # ([numQ,])
    elif hmean_or_mean == 'mean':
        sigma = np.mean(sigma, axis=1)           # ([numQ,])
    else:
        raise NameError('undefined')

    return sigma

def mls_pdist(q_mu, db_mu, q_sigma_sq, db_sigma_sq):
    '''
    q_mu: (q_len, D)
    db_mu: (db_len, D)
    q_sigma_sq: (q_len, D)
    db_sigma_sq: (db_len, D)
    '''
    # calculate mu matrix
    q_len = q_mu.shape[0]
    db_len = db_mu.shape[0]

    preds = np.zeros((q_len, db_len))
    dists = np.zeros((q_len, db_len))
    for ind in tqdm(range(q_mu.shape[0])):
        q_mu_c = q_mu[ind].reshape(1, -1)
        mu_dif = np.square((q_mu_c - db_mu))
        q_sigma_sq_c = q_sigma_sq[ind].reshape(1, -1)
        sigma_sq_sum = q_sigma_sq_c + db_sigma_sq
        mls = (mu_dif / sigma_sq_sum + np.log(sigma_sq_sum)).sum(axis=1)
        pred = np.argsort(mls)
        dist = mls[pred]
        preds[ind] = pred
        dists[ind] = dist

    return preds, dists


def exp_pdist(q_mu, db_mu, q_sigma_sq, db_sigma_sq):
    '''
    q_mu: (q_len, D)
    db_mu: (db_len, D)
    q_sigma_sq: (q_len, D)
    db_sigma_sq: (db_len, D)
    '''
    # calculate mu matrix
    q_len = q_mu.shape[0]
    db_len = db_mu.shape[0]
    concat_mu = np.concatenate((q_mu, db_mu))
    print('calculation sq_euclidean matrix..')
    e_z = squareform(pdist(concat_mu, metric='sqeuclidean'))
    print('done')
    e_z = e_z[:q_len, q_len:]

    # calculate sigma_sq matrix
    q_sigma_sq_sum = np.sum(q_sigma_sq, axis=1).reshape(-1, 1).repeat(db_len, axis=1)
    db_sigma_sq_sum = np.sum(db_sigma_sq, axis=1).reshape(-1, 1).repeat(q_len, axis=1).T
    e_sq = q_sigma_sq_sum + db_sigma_sq_sum

    e_mat = e_z + e_sq                           # (q_len, db_len)
    preds = np.argsort(e_mat, 1)
    dists = np.array([e_mat[i][preds[i]] for i in range(e_mat.shape[0])])

    return preds, dists



def light_log(path, args):
    with open(join(path, 'screen.log'), 'a') as f:
        for arg in args:
            f.write(arg)
            f.flush()
            print(arg, end='')


def cal_recall_from_embeddings(gt, qFeat, dbFeat):
    n_values = [1, 5, 10]

    # ---------------------------------------------------- sklearn --------------------------------------------------- #
    # knn = NearestNeighbors(n_jobs=-1)
    # knn.fit(dbFeat)
    # dists, predictions = knn.kneighbors(qFeat, len(dbFeat))

    # --------------------------------- use faiss to do NN search -------------------------------- #
    faiss_index = faiss.IndexFlatL2(qFeat.shape[1])
    faiss_index.add(dbFeat)
    dists, predictions = faiss_index.search(qFeat, max(n_values))                                  # the results is sorted

    recall_at_n = cal_recall(predictions, gt, n_values)
    return recall_at_n


def cal_recall(ranks, pidx, ks):

    recall_at_k = np.zeros(len(ks))
    for qidx in range(ranks.shape[0]):
        for i, k in enumerate(ks):
            if np.sum(np.in1d(ranks[qidx, :k], pidx[qidx])) > 0:
                recall_at_k[i:] += 1
                break

    recall_at_k /= ranks.shape[0]

    return recall_at_k


def cal_apk(pidx, rank, k):
    if len(rank) > k:
        rank = rank[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(rank):
        if p in pidx and p not in rank[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(pidx), k) * 100.0


def cal_mapk(ranks, pidxs, k):

    return np.mean([cal_apk(a, p, k) for a, p in zip(pidxs, ranks)])


def get_bins(q_sigma_sq_h, num_of_bins):
    q_sigma_sq_h_min = np.min(q_sigma_sq_h)
    q_sigma_sq_h_max = np.max(q_sigma_sq_h)
    # q_sigma_sq_h_max = q_sigma_sq_h_max - 0.3*(q_sigma_sq_h_max - q_sigma_sq_h_min)
    # print(q_sigma_sq_h_min, q_sigma_sq_h_max)
    bins = np.linspace(q_sigma_sq_h_min, q_sigma_sq_h_max, num=num_of_bins)
    indices = []
    for index in range(num_of_bins - 1):
        target_q_ind_l = np.where(q_sigma_sq_h >= bins[index])
        if index != num_of_bins - 2:
            target_q_ind_r = np.where(q_sigma_sq_h < bins[index + 1])
        else:
            # the last bin use close interval
            target_q_ind_r = np.where(q_sigma_sq_h <= bins[index + 1])

        target_q_ind = np.intersect1d(target_q_ind_l, target_q_ind_r)
        indices.append(target_q_ind)
    # print([len(x) for x in indices])
    return indices, bins


def get_equal_sample_bins(q_sigma_sq_h, num_of_bins):
    sorted_ind = np.argsort(q_sigma_sq_h)
    indices = np.array_split(sorted_ind, num_of_bins)
    return indices


def get_zoomed_bins(sigma, num_of_bins):
    s_min = np.min(sigma)
    s_max = np.max(sigma)
    print(s_min, s_max)
    bins_parent = np.linspace(s_min, s_max, num=num_of_bins)
    k = 0
    while True:
        indices = []
        bins_child = np.linspace(bins_parent[0], bins_parent[-1 - k], num=num_of_bins)
        for index in range(num_of_bins - 1):
            target_q_ind_l = np.where(sigma >= bins_child[index])
            if index != num_of_bins - 2:
                target_q_ind_r = np.where(sigma < bins_child[index + 1])
            else:
                target_q_ind_r = np.where(sigma <= bins_child[index + 1])
            target_q_ind = np.intersect1d(target_q_ind_l[0], target_q_ind_r[0])
            indices.append(target_q_ind)
        # if len(indices[-1]) > int(sigma.shape[0] * 0.0005):
        if len(indices[-1]) > int(sigma.shape[0] * 0.001) or k == num_of_bins - 2:
            break
        else:
            k = k + 1
    # print('{:.3f}'.format(sum([len(x) for x in indices]) / sigma.shape[0]), [len(x) for x in indices])
    # print('k=', k)
    return indices, bins_child, k


def bin_pr(preds, dists, gt, vis=False):
    # dists_m = np.around(dists[:, 0], 2)          # (4620,)
    # dists_u = np.array(list(set(dists_m)))
    # dists_u = np.sort(dists_u)                   # small > large

    dists_u = np.linspace(np.min(dists[:, 0]), np.max(dists[:, 0]), num=100)

    recalls = []
    precisions = []
    for th in dists_u:
        TPCount = 0
        FPCount = 0
        FNCount = 0
        TNCount = 0
        for index_q in range(dists.shape[0]):
            # Positive
            if dists[index_q, 0] < th:
                # True
                if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                    TPCount += 1
                else:
                    FPCount += 1
            else:
                if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                    FNCount += 1
                else:
                    TNCount += 1
        assert TPCount + FPCount + FNCount + TNCount == dists.shape[0], 'Count Error!'
        if TPCount + FNCount == 0 or TPCount + FPCount == 0:
            # print('zero')
            continue
        recall = TPCount / (TPCount + FNCount)
        precision = TPCount / (TPCount + FPCount)
        recalls.append(recall)
        precisions.append(precision)
    if vis:
        from matplotlib import pyplot as plt
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.plot(recalls, precisions)
        ax.set_title('Precision-Recall')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.savefig('pr.png', dpi=200)
    return recalls, precisions



    # def view_queries(indices, img_dir, structFile, root_dir):
    #     def parse_dbStruct(path):
    #         from collections import namedtuple
    #         dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ', 'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])
    #         from scipy.io import loadmat
    #         mat = loadmat(path)
    #         matStruct = mat['dbStruct'].item()

    #         dataset = 'nuscenes'

    #         whichSet = matStruct[0].item()

    #         # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    #         # dbImage = [f[0].item() for f in matStruct[1]]
    #         dbImage = matStruct[1]
    #         utmDb = matStruct[2].T

    #         # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    #         # qImage = [f[0].item() for f in matStruct[3]]
    #         qImage = matStruct[3]
    #         utmQ = matStruct[4].T

    #         numDb = matStruct[5].item()
    #         numQ = matStruct[6].item()

    #         posDistThr = matStruct[7].item()
    #         posDistSqThr = matStruct[8].item()
    #         nonTrivPosDistSqThr = matStruct[9].item()

    #         return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr, posDistSqThr, nonTrivPosDistSqThr)

    #     dbStruct = parse_dbStruct(structFile)
    #     query_images = [join(img_dir, dbIm) for dbIm in dbStruct.qImage]
    #     for ind in range(len(indices)):
    #         uncer_itev = join(root_dir,'{:0>2d}'.format(ind))
    #         if exists(uncer_itev):
    #             shutil.rmtree(uncer_itev)
    #             os.makedirs(uncer_itev)
    #         query_inds = indices[ind]
    #         for query_ind in query_inds:
    #             target_img = query_images[query_ind]
    #             shutil.copyfile(target_img, join(uncer_itev, '{:0>5d}_{}'.format(query_ind, target_img.split('/')[-1])))


def parse_dbStruct_nus(path):
    dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ', 'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    dataset = 'nuscenes'

    whichSet = matStruct[0].item()

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    # dbImage = [f[0].item() for f in matStruct[1]]
    dbImage = matStruct[1]
    utmDb = matStruct[2].T

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    # qImage = [f[0].item() for f in matStruct[3]]
    qImage = matStruct[3]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr, posDistSqThr, nonTrivPosDistSqThr)

def parse_dbStruct_pitts(path):
    dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ', 'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    dataset = 'nuscenes'

    whichSet = matStruct[0].item()

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    dbImage = [f[0].item() for f in matStruct[1]]
    # dbImage = matStruct[1]
    utmDb = matStruct[2].T
    # utmDb = matStruct[2]

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    qImage = [f[0].item() for f in matStruct[3]]
    # qImage = matStruct[3]
    utmQ = matStruct[4].T
    # utmQ = matStruct[4]

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr, posDistSqThr, nonTrivPosDistSqThr)


def view_queries(indices, passed, query_withGT_inds):
    img_dir = '/LOCAL/ramdrop/dataset/pitts'
    structFile = '/LOCAL/ramdrop/dataset/pitts/structure/pitts30k_test.mat'
    root_dir = 'postprocess_query/pitts'
    dbStruct = parse_dbStruct_pitts(structFile)
    db_images = [join(img_dir, 'database', dbIm) for dbIm in dbStruct.dbImage]
    query_images = np.array([join(img_dir, 'query', qIm) for qIm in dbStruct.qImage])
    query_images = query_images[query_withGT_inds]
    hs_list = np.zeros(len(indices))
    for ind in range(len(indices)):
        uncer_itev = join(root_dir,'{:0>2d}'.format(ind))
        if exists(uncer_itev):
            shutil.rmtree(uncer_itev)
        os.makedirs(uncer_itev)
        os.makedirs(join(uncer_itev, 'passed'))
        os.makedirs(join(uncer_itev, 'failed'))
        query_inds = indices[ind]
        for query_ind in query_inds:
            target_img = query_images[query_ind]
            hs_list[ind] += cal_hs(target_img)
            if query_ind in passed:
                shutil.copyfile(target_img, join(uncer_itev, 'passed', '{:0>5d}_{}'.format(query_ind, target_img.split('/')[-1])))
            else:
                shutil.copyfile(target_img, join(uncer_itev, 'failed', '{:0>5d}_{}'.format(query_ind, target_img.split('/')[-1])))
    n_list = np.array([len(inds) for inds in indices])
    return hs_list / n_list


def view_pairs(indices, bins_child, q_sigma_sq_h, db_sigma_sq_h, preds, passed, query_withGT_inds):
    img_dir = '/LOCAL/ramdrop/dataset/pitts'
    structFile = '/LOCAL/ramdrop/dataset/pitts/structure/pitts30k_test.mat'
    root_dir = 'postprocess_pair/pitts'
    dbStruct = parse_dbStruct_pitts(structFile)
    db_images = [join(img_dir, 'database', dbIm) for dbIm in dbStruct.dbImage]
    query_images = np.array([join(img_dir, 'query', qIm) for qIm in dbStruct.qImage])
    query_images = query_images[query_withGT_inds]
    for ind in range(len(indices)):
        uncer_itev = join(root_dir, '{:0>2d}'.format(ind))
        if exists(uncer_itev):
            shutil.rmtree(uncer_itev)
        os.makedirs(uncer_itev)
        os.makedirs(join(uncer_itev, 'passed'))
        os.makedirs(join(uncer_itev, 'failed'))
        query_inds = indices[ind]
        for query_ind in query_inds:
            q_img = query_images[query_ind]
            if query_ind in passed:
                shutil.copyfile(q_img, join(uncer_itev, 'passed', '{:0>5d}_{}'.format(query_ind, q_img.split('/')[-1])))

                # pair: query ~ top 1 candidate
                pair_uty = q_sigma_sq_h[query_ind] + db_sigma_sq_h[preds[query_ind][0]]
                uncertain = np.argwhere(bins_child < pair_uty)
                if uncertain.size == 0: continue
                uty_itv = uncertain[-1][0]
                shutil.copyfile(db_images[preds[query_ind][0]], join(uncer_itev, 'passed', '{:0>5d}n1_{}_{}'.format(query_ind, uty_itv, db_images[preds[query_ind][0]].split('/')[-1])))

                # pair: query ~ top 10 candidate
                pair_uty = q_sigma_sq_h[query_ind] + db_sigma_sq_h[preds[query_ind][9]]
                uncertain = np.argwhere(bins_child < pair_uty)
                if uncertain.size == 0: continue
                uty_itv = uncertain[-1][0]
                shutil.copyfile(db_images[preds[query_ind][9]], join(uncer_itev, 'passed', '{:0>5d}n2_{}_{}'.format(query_ind, uty_itv, db_images[preds[query_ind][9]].split('/')[-1])))
            else:
                shutil.copyfile(q_img, join(uncer_itev, 'failed', '{:0>5d}_{}'.format(query_ind, q_img.split('/')[-1])))
                # pair: query ~ top 1 candidate
                pair_uty = q_sigma_sq_h[query_ind] + db_sigma_sq_h[preds[query_ind][0]]
                uncertain = np.argwhere(bins_child < pair_uty)
                if uncertain.size == 0: continue
                uty_itv = uncertain[-1][0]
                shutil.copyfile(db_images[preds[query_ind][0]], join(uncer_itev, 'failed', '{:0>5d}n1_{}_{}'.format(query_ind, uty_itv, db_images[preds[query_ind][0]].split('/')[-1])))

def cal_hs(img_path):
    # TODO use PIL
    # img = io.imread(img_path, as_gray=True).reshape(-1, 1)
    counts, bins = np.histogram((img * 255).astype(np.int16), np.arange(0, 256, 1))
    counts = counts / np.sum(counts)
    cumulative = np.cumsum(counts)
    in_min = np.min((img*255).astype(np.int16))
    in_max = np.max((img*255).astype(np.int16))
    per_75 = np.argwhere(cumulative < 0.75)[-1]
    per_25 = np.argwhere(cumulative < 0.25)[-1]
    hs = (per_75 - per_25)/255
    return hs

if __name__ == '__main__':
    # view_queries([[1, 3], [2, 5]], 'nuscenes/7n5s_xy11/img', 'nuscenes/7n5s_xy11/nuscenes_test.mat', 'tmp/nus')
    view_queries([[1, 3], [2, 5]], '/LOCAL/ramdrop/dataset/pitts', '/LOCAL/ramdrop/dataset/pitts/structure/pitts30k_test.mat', 'tmp/pitts')
