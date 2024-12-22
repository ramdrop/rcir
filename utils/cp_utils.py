import numpy as np

def calculate_coverage(positives_cal, pred_cal_refine):
    recall_at_k_cal = 0
    for qidx in range(len(positives_cal)):
        if np.sum(np.in1d(positives_cal[qidx], pred_cal_refine[qidx])) > 0:
            recall_at_k_cal += 1
    recall_at_k_cal /= len(positives_cal) / 100
    return recall_at_k_cal


def calculate_distance_matrix(a, b):
    '''
    given two arrays, a and b, each of shape (N, M) and (K, M) respectively, calculate the distance matrix D such that D[i, j] is the euc distance between a[i] and b[j].
    '''
    a2 = np.sum(np.square(a), axis=1)
    b2 = np.sum(np.square(b), axis=1)
    ab = np.dot(a, b.T)
    D = np.sqrt(a2[:, None] + b2[None, :] - 2 * ab + 1e-6)
    # D = a2[:, None] + b2[None, :] - 2 * ab
    return D

def calculate_similarity_matrix(a, b):
    '''
    given two arrays, a and b, each of shape (N, M) and (K, M) respectively, calculate the cosine similarity matrix D such that D[i, j] is the cosine similarity between a[i] and b[j].
    '''
    a2 = np.sum(np.square(a), axis=1)
    b2 = np.sum(np.square(b), axis=1)
    ab = np.dot(a, b.T)
    D = ab / np.sqrt(a2[:, None] * b2[None, :])
    return D

def cal_recall(ranks, pidx, ks):
    recall_at_k = np.zeros(len(ks))
    for qidx in range(ranks.shape[0]):
        for i, k in enumerate(ks):
            if np.sum(np.in1d(ranks[qidx, :k], pidx[qidx])) > 0:
                recall_at_k[i:] += 1
                break
    recall_at_k /= ranks.shape[0]
    return recall_at_k * 100.0

def locate_var_level(var_stat, var):
    '''
    given the maximum and minimum value of var, decide the percentile of a new var
    '''
    var_min, var_max = var_stat
    if var < var_min:
        return 1
    elif var > var_max:
        return 0
    return 1 - (var - var_min) / (var_max - var_min)


def miscalibration_rate(x, y, sgn=True):
    '''
    calculate L1 distance between two curves
    '''
    if sgn:
        return np.sum(x - y) / len(x)
    else:
        return np.sum(np.abs(x - y)) / len(x)
