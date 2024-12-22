import torch
from torch.distributions.normal import Normal

def negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin = 0.0):

    muA2 = muA**2                                # ([D-1, N])   # N = S - 2
    muP2 = muP**2                                # ([D-1, N])
    muN2 = muN**2                                # ([D-1, N])
    varP2 = varP**2                              # (1, N)
    varN2 = varN**2                              # (1, N)

    mu = torch.sum(muP2 + varP - muN2 - varN - 2 * muA * (muP - muN), dim=0)                                           # ([5])broadcast from 1 to D-1
    T1 = varP2 + 2 * muP2 * varP + 2 * (varA + muA2) * (varP + muP2) - 2 * muA2 * muP2 - 4 * muA * muP * varP          # ([2047, 5])
    T2 = varN2 + 2 * muN2 * varN + 2 * (varA + muA2) * (varN + muN2) - 2 * muA2 * muN2 - 4 * muA * muN * varN          # ([2047, 5])
    T3 = 4 * muP * muN * varA                                                                                          # ([2047, 5])
    sigma2 = torch.sum(2 * T1 + 2 * T2 - 2 * T3, dim=0)                                            # ([5])
    sigma = sigma2**0.5

    probs = Normal(loc=mu, scale=sigma + 1e-8).cdf(margin)
    nll = -torch.log(probs + 1e-8)

    return nll.mean()

def kl_div_gauss(mu_q, var_q, mu_p, var_p): # (D, N), (1, N)

    # N, D = mu_q.shape

    # kl diverence for isotropic gaussian
    # kl = 0.5 * ((var_q / var_p) * D + \
    #     1.0 / (var_p) * torch.sum(mu_p**2 + mu_q**2 - 2 * mu_p * mu_q, axis=1) - D + \
    #         D * (torch.log(var_p) - torch.log(var_q)))
    D, N = mu_q.shape

    kl = 0.5 * ((var_q / var_p) * D + 1.0 / (var_p) * torch.sum(mu_p**2 + mu_q**2 - 2 * mu_p * mu_q, axis=0) - D + D * (torch.log(var_p) - torch.log(var_q)))

    return kl.mean()


def kl_div_vMF(mu_q, var_q):
    N, D = mu_q.shape

    # we are estimating the variance and not kappa in the network.
    # They are propertional
    kappa_q = 1.0 / var_q
    kl = kappa_q - D * torch.log(2.0)

    return kl.mean()


def triplet_loss(x, label, margin=0.1):
    # x is D x N
    dim = x.size(0)                              # D
    nq = torch.sum(label.data == -1).item()      # number of tuples
    S = x.size(1) // nq                          # number of images per tuple including query: 1+1+n

    xa = x[:, label.data == -1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, dim).permute(1, 0)                   # ([2048, 5])
    xp = x[:, label.data == 1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, dim).permute(1, 0)                    # ([2048, 5])
    xn = x[:, label.data == 0]                                                                                         # ([2048, 5])

    dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=0)
    dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=0)

    return torch.sum(torch.clamp(dist_pos - dist_neg + margin, min=0))
