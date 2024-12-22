from genericpath import exists
import os
import json
import argparse
import numpy as np
import random
import torch
from os.path import join

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Options")
        self.parser.add_argument('--phase', type=str, default='train', help='phase', choices=['train', 'test'])
        self.parser.add_argument('--dataset', type=str, default='cub200', help='choose dataset.', choices=['pitts', 'cub200', 'car196', 'chestx', 'sop'])
        self.parser.add_argument('--data_path', type=str, default='', help='choose dataset.')
        self.parser.add_argument('--height', type=int, default=200, help='number of sequence to use.')
        self.parser.add_argument('--width', type=int, default=200, help='number of sequence to use.')
        self.parser.add_argument('--net', type=str, default='', help='network')
        self.parser.add_argument('--setting', type=str, default='btl', help='network', choices=['btl', 'dul', 'mcd', 'triplet'])
        self.parser.add_argument('--loss', type=str, default='tri', help='triplet loss or bayesian triplet loss', choices=['triplet','bayes_triplet'])
        self.parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
        self.parser.add_argument('--batchSize', type=int, default=8, help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
        self.parser.add_argument('--cacheBatchSize', type=int, default=25, help='Batch size for caching and testing')
        self.parser.add_argument('--cacheRefreshRate', type=int, default=0, help='How often to refresh cache, in number of queries. 0 for off')
        self.parser.add_argument('--nEpochs', type=int, default=60, help='number of epochs to train for')
        self.parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
        self.parser.add_argument('--cGPU', type=int, default=2, help='core of GPU to use.')
        self.parser.add_argument('--optim', type=str, default='adam', help='optimizer to use', choices=['sgd', 'adam'])
        self.parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate.')
        self.parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
        self.parser.add_argument('--lrGamma', type=float, default=0.99, help='Multiply LR by Gamma for decaying.')
        self.parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
        self.parser.add_argument('--cuda', action='store_false', help='use cuda')
        self.parser.add_argument('--debug', action='store_true', help='debug mode')
        self.parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
        self.parser.add_argument('--seed', type=int, default=1234, help='Random seed to use.')
        self.parser.add_argument('--logsPath', type=str, default='./logs', help='Path to save runs to.')
        self.parser.add_argument('--runsPath', type=str, default='not defined', help='Path to save runs to.')
        self.parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
        self.parser.add_argument('--evalEvery', type=int, default=1, help='Do a validation set run, and save, every N epochs.')
        self.parser.add_argument('--cacheRefreshEvery', type=int, default=1, help='refresh embedding cache, every N epochs.')
        self.parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping. 0 is off.')
        self.parser.add_argument('--split', type=str, default='val', help='Split to use', choices=['val', 'test'])
        self.parser.add_argument('--lambda_kl', type=float, default=1e-4, help='Lambda for KL loss.')
        self.parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout Rate.')


    def parse(self):
        options = self.parser.parse_args()

        # dataset hard-coded for easy debugging, should be removed for public release
        if options.dataset == 'cub200':
            options.data_path = 'dbs/CUB_200_2011'
        elif options.dataset == 'car196':
            options.data_path = 'dbs/CAR196'
        elif options.dataset == 'pitts':
            options.data_path = 'dbs/pitts'
        elif options.dataset == 'chestx':
            options.data_path = 'dbs/chest_x_det'
        elif options.dataset == 'sop':
            options.data_path = 'dbs/SOP/Stanford_Online_Products'
        else:
            raise NameError('undefined dataset :(')

        # setting
        if options.setting in ['btl']:
            options.loss = 'bayes_triplet'
        elif options.setting in ['dul', 'mcd', 'triplet']:
            options.loss = 'triplet'
        else:
            raise NameError('undefined setting :(')

        return options

    def update_opt_from_json(self, flag_file, options):
        if not exists(flag_file):
            raise ValueError('{} not exist'.format(flag_file))
        # restore_var = ['runsPath', 'net', 'seqLen', 'num_clusters', 'output_dim', 'structDir', 'imgDir', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 'num_clusters', 'optim', 'margin', 'seed', 'patience']
        black_list = ['resume', 'mode', 'phase', 'optim', 'split']
        if os.path.exists(flag_file):
            with open(flag_file, 'r') as f:
                # stored_flags = {'--' + k: str(v) for k, v in json.load(f).items() if k in restore_var}
                stored_flags = {'--' + k: str(v) for k, v in json.load(f).items() if k not in black_list}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in self.parser._actions:
                        if act.dest == flag[2:]:    # stored parser match current parser
                            # store_true / store_false args don't accept arguments, filter these
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                            else:
                                if val == str(act.default):
                                    to_del.append(flag)

                for flag, val in stored_flags.items():
                    missing = True
                    for act in self.parser._actions:
                        if flag[2:] == act.dest:
                            missing = False
                    if missing:
                        to_del.append(flag)

                for flag in to_del:
                    del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                # print('restored flags:', train_flags)
                options = self.parser.parse_args(train_flags, namespace=options)
        return options


class FixRandom:
    def __init__(self, seed) -> None:
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    def seed_worker(self):
        worker_seed = self.seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)
