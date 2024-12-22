#%%
from matplotlib.pyplot import axis
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import wandb
import torch.nn.functional as F
import faiss
import h5py
from ipdb import set_trace
from datetime import datetime
import os
from os.path import exists, join, dirname
import numpy as np
from tqdm import tqdm
import json
import shutil
import importlib
import random
import pickle
import utils.utils as utils

os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# private library
from options import FixRandom
from utils.utils import light_log, cal_recall, schedule_device
from losses.loss import BayesianTripletLoss, TripletLoss
from networks.network import Model
import losses.functional as LF


class Trainer:
    def __init__(self, options) -> None:

        self.opt = options

        # r variables
        self.step = 0
        self.epoch = 0
        self.current_lr = 0
        self.best_recalls = [0, 0, 0]

        # seed
        fix_random = FixRandom(self.opt.seed)
        self.seed_worker = fix_random.seed_worker()

        # id
        self.phase = self.opt.phase.split('_')[0]
        self.time_stamp = datetime.now().strftime('%m%d_%H%M%S')

        # set device
        if self.opt.phase == 'train':
            self.opt.cGPU = schedule_device()
            if self.opt.cuda and not torch.cuda.is_available():
                raise Exception("No GPU found, please run with --nocuda")
        torch.cuda.set_device(self.opt.cGPU)
        self.device = torch.device("cuda")
        print(f"device: {self.device}{torch.cuda.current_device()}")

        # make model
        if self.opt.phase == 'train':
            self.model, self.optimizer, self.scheduler, self.criterion = self.make_model()
        elif self.opt.phase == 'test':
            self.model = self.make_model()

        # make folders
        self.make_folder()

        # make dataset
        self.make_dataset()

        # logs
        if self.opt.phase == 'train':
            wandb.init(project="CIR", config=vars(self.opt), group=f"{self.opt.dataset}", name=f"{self.opt.dataset}_{self.opt.setting}_{self.time_stamp}")

    def make_dataset(self):
        if self.opt.phase == 'train':
            assert os.path.exists('datasets/{}.py'.format(self.opt.dataset)), 'cannot find ' + '{}.py'.format(self.opt.dataset)
            self.dataset = importlib.import_module('datasets.' + self.opt.dataset)
        elif self.opt.phase == 'test':
            self.dataset = importlib.import_module(f"{dirname(self.opt.resume).replace('/', '.')}.models.{self.opt.dataset}")
            # self.dataset = importlib.import_module('tmp.models.{}'.format(self.opt.dataset))

        if self.opt.dataset in ['cub200', 'car196', 'chestx', 'sop']:
            # for emb cache
            self.whole_train_set = self.dataset.Whole('train', data_path=self.opt.data_path, aug=True, debug=self.opt.debug)
            self.whole_training_data_loader = DataLoader(dataset=self.whole_train_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=True, pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
            self.whole_val_set = self.dataset.Whole('val', data_path=self.opt.data_path, aug=False, debug=self.opt.debug)
            self.whole_val_data_loader = DataLoader(dataset=self.whole_val_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
            self.whole_test_set = self.dataset.Whole('test', data_path=self.opt.data_path, aug=False, debug=self.opt.debug)
            self.whole_test_data_loader = DataLoader(dataset=self.whole_test_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
            # for train tuples
            self.train_set = self.dataset.Tuple('train', data_path=self.opt.data_path, margin=self.opt.margin, debug=self.opt.debug)
            self.training_data_loader = DataLoader(dataset=self.train_set, num_workers=8, batch_size=self.opt.batchSize, shuffle=True, collate_fn=self.dataset.collate_fn, worker_init_fn=self.seed_worker)

        elif self.opt.dataset in ['pitts']:
            # for emb cache
            self.whole_train_set = self.dataset.Whole('train', data_path=self.opt.data_path, img_size=(self.opt.height, self.opt.width))
            self.whole_training_data_loader = DataLoader(dataset=self.whole_train_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
            self.whole_val_set = self.dataset.Whole('val', data_path=self.opt.data_path, img_size=(self.opt.height, self.opt.width))
            self.whole_val_data_loader = DataLoader(dataset=self.whole_val_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
            self.whole_test_set = self.dataset.Whole('test', data_path=self.opt.data_path, img_size=(self.opt.height, self.opt.width))
            self.whole_test_data_loader = DataLoader(dataset=self.whole_test_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)

            # for train tuples
            self.train_set = self.dataset.Tuple('train', data_path=self.opt.data_path, margin=self.opt.margin)
            self.training_data_loader = DataLoader(dataset=self.train_set, num_workers=8, batch_size=self.opt.batchSize, shuffle=True, collate_fn=self.dataset.collate_fn, worker_init_fn=self.seed_worker)
            print('{}:{}, {}:{}, {}:{}, {}:{}, {}:{}'.format('dataset', self.opt.dataset, 'database', self.whole_train_set.dbStruct.numDb, 'train_set', self.whole_train_set.dbStruct.numQ, 'val_set', self.whole_val_set.dbStruct.numQ, 'test_set',
                                                             self.whole_test_set.dbStruct.numQ))

        print('{}:{}, {}:{}'.format('cache_bs', self.opt.cacheBatchSize, 'tuple_bs', self.opt.batchSize))

    def make_folder(self):
        ''' create folders to store tensorboard files and a copy of networks files
        '''
        if self.opt.phase == 'train':
            self.opt.runsPath = join(self.opt.logsPath, f"{self.opt.dataset}_{self.opt.setting}_{self.time_stamp}")
            if not os.path.exists(join(self.opt.runsPath, 'models')):
                os.makedirs(join(self.opt.runsPath, 'models'))

            for file in [__file__, f'datasets/{self.opt.dataset}.py', 'networks/network.py']:
                shutil.copyfile(file, os.path.join(self.opt.runsPath, 'models', file.split('/')[-1]))

            with open(join(self.opt.runsPath, 'flags.json'), 'w') as f:
                f.write(json.dumps({k: v for k, v in vars(self.opt).items()}, indent=''))

    def make_model(self):
        # model
        if self.opt.phase == 'train':
            model = Model(setting=self.opt.setting, dropout_rate=self.opt.dropout_rate)
            model = model.to(self.device)
        elif self.opt.phase == 'test':
            assert self.opt.resume != '', 'undefined resume path :('
            network = importlib.import_module(f"{dirname(self.opt.resume).replace('/', '.')}.models.network")
            model = network.Model(setting=self.opt.setting, dropout_rate=self.opt.dropout_rate).to(self.device)
            checkpoint = torch.load(self.opt.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"setting: {self.opt.setting}")

        # optimizer and loss function
        if self.opt.phase == 'train':
            # optimizer
            if self.opt.optim == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), self.opt.lr, weight_decay=self.opt.weightDecay)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.opt.lrGamma, last_epoch=-1, verbose=False)
            elif self.opt.optim == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.opt.lr, momentum=self.opt.momentum, weight_decay=self.opt.weightDecay)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lrStep, gamma=self.opt.lrGamma)
            else:
                raise NameError('undefined optimizer')
            # loss function
            if self.opt.loss == 'triplet':
                criterion = nn.TripletMarginLoss(margin=self.opt.margin, p=2, reduction='sum').to(self.device)
            elif self.opt.loss == 'bayes_triplet':
                criterion = BayesianTripletLoss(margin=0, varPrior=1 / 2047.0).to(self.device)     # 0.00004885
        if self.opt.nGPU > 1:
            model = nn.DataParallel(model)

        if self.opt.phase == 'train':
            return model, optimizer, scheduler, criterion
        elif self.opt.phase == 'test':
            return model

    def build_embedding_cache(self):
        '''build embedding cache, such that we can find the corresponding (p) and (n) with respect to (a) in embedding space
        '''
        if self.opt.dataset in ['cub200', 'car196', 'chestx', 'sop']:
            cache = torch.zeros((len(self.whole_train_set), self.model.mu_dim), device=self.device)                    # ([N, D])
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(tqdm(self.whole_training_data_loader), 1):
                    input = input.to(self.device)                                                                      # torch.Size([B, C, H, W])
                    emb, _ = self.model(input)                                                                         # ([B, D])
                    cache[indices, :] = emb
                    del input, emb
            self.train_set.cache = cache.to(torch.device("cpu"))                                                       # update train tuples set embedding cache
        elif self.opt.dataset in ['pitts']:
            self.train_set.cache = os.path.join(self.opt.runsPath, self.train_set.whichSet + '_feat_cache.hdf5')
            with h5py.File(self.train_set.cache, mode='w') as h5:
                h5feat = h5.create_dataset("features", [len(self.whole_train_set), self.model.mu_dim], dtype=np.float32)
                with torch.no_grad():
                    for iteration, (input, indices) in enumerate(tqdm(self.whole_training_data_loader), 1):
                        input = input.to(self.device)                                                                  # torch.Size([32, 3, 154, 154]) ([32, 5, 3, 200, 200])
                        emb, _ = self.model(input)
                        h5feat[indices.detach().numpy(), :] = emb.detach().cpu().numpy()
                        del input, emb
        else:
            raise NameError('undefined dataset :(')

    def process_batch(self, batch_inputs):
        '''process a batch of input
        '''
        anchor, positives, negatives, neg_counts, indices = batch_inputs
        # in case we get an empty batch
        if anchor is None:
            return None, None
        # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor, where N = batchSize * (nQuery + nPos + n_neg)
        B = anchor.shape[0]                                                                        # ([8, 1, 3, 200, 200])
        n_neg = torch.sum(neg_counts)                                                              # tensor(80) = torch.sum(torch.Size([8]))
        input = torch.cat([anchor, positives, negatives])                                          # ([B, C, H, 200])

        input = input.to(self.device)            # ([96, 1, C, H, W])
        embs, vars = self.model(input)           # ([96, D])

        # track the range of variance
        if self.step % 100 == 0 and self.opt.setting in ['btl', 'dul']:
            if self.opt.setting == 'dul':
                wandb.log({'sigma_sq/avg': torch.mean(vars[1]).item()}, step=self.step)
                wandb.log({'sigma_sq/max': torch.max(vars[1]).item()}, step=self.step)
                wandb.log({'sigma_sq/min': torch.min(vars[1]).item()}, step=self.step)
            else:
                wandb.log({'sigma_sq/avg': torch.mean(vars).item()}, step=self.step)
                wandb.log({'sigma_sq/max': torch.max(vars).item()}, step=self.step)
                wandb.log({'sigma_sq/min': torch.min(vars).item()}, step=self.step)

        tuple_loss = 0
        # Triplet loss
        if self.opt.loss == 'triplet':
            embs_a, embs_p, embs_n = torch.split(embs, [B, B, n_neg])
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    negIx = (torch.sum(neg_counts[:i]) + n).item()
                    tuple_loss += self.criterion(embs_a[i:i + 1], embs_p[i:i + 1], embs_n[negIx:negIx + 1])
            tuple_loss /= n_neg.float().to(self.device)                                            # normalise by actual number of negatives
            if self.opt.net == 'dul':

                def kl_divergence(mu, sigma2):
                    logsigma2 = sigma2.log()
                    kl = -(1 + logsigma2 - mu.pow(2) - sigma2) / 2
                    kl = kl.sum(dim=1).mean()
                    return kl

                mu, sigma2 = vars
                kl_loss = kl_divergence(mu, sigma2)
                tuple_loss = tuple_loss + self.opt.lambda_kl * kl_loss
        # Bayesian triplet loss
        elif self.opt.loss == 'bayes_triplet':
            embs = torch.cat((embs, vars), dim=-1)
            embs_a, embs_p, embs_n = torch.split(embs, [B, B, n_neg])
            for i, neg_count in enumerate(neg_counts):
                emb_a = embs_a[i:i + 1]                                                            # (1, D)
                emb_p = embs_p[i:i + 1]                                                            # (1, D)
                st = torch.sum(neg_counts[:i])
                emb_n = embs_n[st:st + neg_count]                                                  # (neg_count, D)
                x = torch.cat((emb_a, emb_p, emb_n), axis=0).transpose(0, 1)                       # (1+1+neg_count, D)
                label = torch.cat((torch.tensor([-1, 1]), torch.zeros((neg_count, ))))
                tuple_loss += self.criterion(x, label)
            tuple_loss /= B

        del input, embs, embs_a, embs_p, embs_n
        del anchor, positives, negatives

        return tuple_loss, n_neg

    def train(self):
        not_improved = 0
        for epoch in range(self.opt.nEpochs):
            self.epoch = epoch
            self.current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            # build embedding cache
            if self.epoch % self.opt.cacheRefreshEvery == 0:
                self.model.eval()
                self.build_embedding_cache()
                self.model.train()

            # train
            tuple_loss_sum = 0
            n_batches = len(self.training_data_loader)
            for _, batch_inputs in enumerate(tqdm(self.training_data_loader)):
                self.step += 1

                self.optimizer.zero_grad()
                tuple_loss, n_neg = self.process_batch(batch_inputs)
                if tuple_loss is None:
                    continue
                tuple_loss.backward()
                self.optimizer.step()

                tuple_loss_sum += tuple_loss.item()
                if self.step % 10 == 0:
                    wandb.log({'train_tuple_loss': tuple_loss.item()}, step=self.step)
                    wandb.log({'train_batch_num_neg': n_neg}, step=self.step)

            wandb.log({'train_avg_tuple_loss': tuple_loss_sum / n_batches}, step=self.step)

            torch.cuda.empty_cache()

            self.scheduler.step()

            # val every 2 epochs
            if (self.epoch % self.opt.evalEvery) == 0:
                recalls = self.eval()
                if recalls[0] > self.best_recalls[0]:
                    self.best_recalls = recalls
                    not_improved = 0
                else:
                    not_improved += self.opt.evalEvery
                # light log
                light_log(self.opt.runsPath, [
                    f'e={self.epoch:>2d},',
                    f'lr={self.current_lr:>.8f},',
                    f'tl={tuple_loss_sum / n_batches:>.4f},',
                    f'r@1/5/10={recalls[0]:.2f}/{recalls[1]:.2f}/{recalls[2]:.2f}',
                    '\n' if not_improved else ' *\n',
                ])
            else:
                recalls = None
            self.save_model(self.model, is_best=not not_improved)

            # stop when not improving for a period
            if self.opt.phase in ['train'] and self.opt.patience > 0:
                if not_improved > self.opt.patience:
                    print('terminated because performance has not improve for', self.opt.patience, 'epochs')
                    break

        self.save_model(self.model, is_best=False)
        print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(self.best_recalls[0], self.best_recalls[1], self.best_recalls[2]))

        return self.best_recalls

    def eval(self):
        recalls, _ = self.get_recall(self.model, save_embeddings=True if self.opt.phase == 'test' else False)
        if self.opt.phase in ['train']:
            for i, n in enumerate([1, 5, 10]):
                wandb.log({f"{self.opt.split}_r@{n}": recalls[i]}, step=self.step)
        elif self.opt.phase in ['test']:
            print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(recalls[0], recalls[1], recalls[2]))

        return recalls

    def save_model(self, model, is_best=False):
        if is_best:
            torch.save({
                'epoch': self.epoch,
                'step': self.step,
                'state_dict': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, os.path.join(self.opt.runsPath, 'ckpt_best.pth.tar'))

        # torch.save({
        #     'epoch': self.epoch,
        #     'step': self.step,
        #     'state_dict': model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        #     'scheduler': self.scheduler.state_dict(),
        # }, os.path.join(self.opt.runsPath, 'ckpt_e_{}.pth.tar'.format(self.epoch)))

    def get_recall(self, model, save_embeddings=False):
        if self.opt.setting in ['btl', 'dul', 'triplet']:
            model.eval()
        elif self.opt.setting in ['mcd']:
            model.train()
            print('test time dropout enabled')

        if self.opt.split == 'val':
            eval_dataloader = self.whole_val_data_loader
            eval_set = self.whole_val_set
        elif self.opt.split == 'test':
            eval_dataloader = self.whole_test_data_loader
            eval_set = self.whole_test_set
        # print(f"{self.opt.split} set len:{len(eval_set)}")

        whole_mu = torch.zeros((len(eval_set), model.mu_dim), device=self.device)                                                          # (N, D)
        whole_var = torch.zeros((len(eval_set), model.mu_dim if self.opt.setting in ['mcd'] else model.sigma_dim), device=self.device)     # (N, D)
        gt = eval_set.get_positives()                                                                                                      # (N, n_pos)

        with torch.no_grad():
            for iteration, (input, indices) in enumerate(tqdm(eval_dataloader), 1):
                input = input.to(self.device)
                if self.opt.setting in ['btl', 'dul', 'triplet']:
                    mu, var = model(input)                                                         # (B, D)
                    if self.opt.setting in ['dul']:
                        var = var[1]
                    whole_mu[indices, :] = mu
                    whole_var[indices, :] = var
                elif self.opt.setting in ['mcd']:
                    outputs = [model(input) for i in range(40)]                                    # (B, D),  According to Kendall(2016), 40 is enough for converge
                    outputs_mean = [mean for (mean, var) in outputs]
                    outputs_mean = torch.stack(outputs_mean)                                       # ([20, 128, 2048])
                    model_variance = torch.var(outputs_mean, dim=0)                                # ([128, 2048])
                    outputs_mean = torch.mean(outputs_mean, dim=0)                                 # ([128, 2048])
                    whole_mu[indices, :] = outputs_mean
                    whole_var[indices, :] = model_variance
                    # del input, mu, var

        # n_values = [1, 5, 10]
        n_values = [1, 5, 10, 20, 30, 40, 50]
        if self.opt.dataset in ['cub200', 'car196', 'chestx', 'sop']:
            whole_mu = whole_mu.cpu().numpy()
            whole_var = whole_var.cpu().numpy()

            # faiss_index = faiss.IndexFlatL2(whole_mu.shape[1])
            # faiss_index.add(whole_mu)
            # dists, preds = faiss_index.search(whole_mu, max(n_values) + 1)                         # +1 because query itself occupies a position                    # (N, n), (N, n) the results is sorted
            dists, preds = utils.find_nn(whole_mu, whole_mu, max(n_values) + 1)
            dists = dists[:, 1:]                                                                   # -1: to exclude query itself
            preds = preds[:, 1:]                                                                   # -1: to exclude query itself
            mu_q = whole_mu
            mu_db = whole_mu
            sigma_q = whole_var
            sigma_db = whole_var
        elif self.opt.dataset in ['pitts']:
            # whole_var = torch.exp(whole_var)
            whole_mu = whole_mu.cpu().numpy()
            whole_var = whole_var.cpu().numpy()
            mu_q = whole_mu[eval_set.dbStruct.numDb:].astype('float32')
            mu_db = whole_mu[:eval_set.dbStruct.numDb].astype('float32')
            sigma_q = whole_var[eval_set.dbStruct.numDb:].astype('float32')
            sigma_db = whole_var[:eval_set.dbStruct.numDb].astype('float32')
            dists, preds = utils.find_nn(mu_q, mu_db, max(n_values))                               # the results is sorted
            # faiss_index = faiss.IndexFlatL2(mu_q.shape[1])
            # faiss_index.add(mu_db)
            # dists, preds = faiss_index.search(mu_q, max(n_values))                                 # the results is sorted

            # cull queries without any ground truth positives in the database
            val_inds = np.array([True if len(gt[ind]) != 0 else False for ind in range(len(gt))])
            mu_q = mu_q[val_inds]
            sigma_q = sigma_q[val_inds]
            preds = preds[val_inds]
            dists = dists[val_inds]
            gt = gt[val_inds]
        else:
            raise NameError('undefined dataset :(')

        if save_embeddings:
            with open(join(self.opt.runsPath, f"{self.opt.split}_embeddings_{self.opt.resume.split('.')[-3].split('_')[-1]}.pickle"), 'wb') as handle:
                pickle.dump(mu_q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(mu_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(sigma_q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(sigma_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(gt, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('embeddings saved')

        recall_at_k = cal_recall(preds, gt, n_values)

        return recall_at_k, None


if __name__ == '__main__':
    mean_tea = torch.rand([4, 2])
    mean_stu, var_stu = torch.rand([4, 2]), torch.rand([4, 2])
    loss = torch.exp(-var_stu) * torch.square(mean_tea - mean_stu) + var_stu
    loss_sum = torch.sum(loss)
    print(loss_sum)
    print(loss_sum.shape)
    pass