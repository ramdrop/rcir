#%%
import os
import pickle
from os.path import dirname, join, exists
import logging

import lightning.pytorch as pl
import numpy as np
import torch
# from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import  RichProgressBar

from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from PIL import Image
from torch import Tensor, nn, optim, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import wandb
from losses.loss import BayesianTripletLoss, TripletLoss
from networks.network import Model
from utils import utils

plt.style.use('ggplot')
import matin
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
from rich.progress import track
from rich.progress import Progress
from rich.console import Console
from rich import print

# A logger for this file
log = logging.getLogger(__name__)


class TrainExt(pl.Callback):
    def on_validation_start(self, trainer, pl_module):
        if pl_module.embeddings_initialized == False:
            pl_module.mu_all = torch.zeros((len(trainer.datamodule.whole_val_set), pl_module.embedding_dim), device=pl_module.device)          # (N, D)
            pl_module.variance_all = torch.zeros((len(trainer.datamodule.whole_val_set), pl_module.variance_dim), device=pl_module.device)     # (N, D)
            pl_module.gt_all = trainer.datamodule.whole_val_set.get_positives()
            pl_module.embeddings_initialized = True

    def on_test_start(self, trainer, pl_module):
        if pl_module.embeddings_initialized == False:
            pl_module.mu_all = torch.zeros((len(trainer.datamodule.whole_test_set), pl_module.embedding_dim), device=pl_module.device)         # (N, D)
            pl_module.variance_all = torch.zeros((len(trainer.datamodule.whole_test_set), pl_module.variance_dim), device=pl_module.device)    # (N, D)
            pl_module.gt_all = trainer.datamodule.whole_test_set.get_positives()
            pl_module.embeddings_initialized = True

    def on_train_epoch_start(self, trainer, pl_module):
        console = Console()
        cache = torch.zeros((len(trainer.datamodule.whole_train_set), pl_module.embedding_dim), device=pl_module.device)                   # ([N, D])
        with console.status("[bold green]Building embedding cache...") as status:
            with torch.no_grad():
                for input, indices in trainer.datamodule.whole_training_data_loader:
                    input = input.to(pl_module.device)                                                                     # torch.Size([B, C, H, W])
                    emb, _ = pl_module.model(input)                                                                        # ([B, D])
                    cache[indices, :] = emb
                    del input, emb
        trainer.datamodule.train_set.cache = cache.to(torch.device("cpu"))                                             # update train tuples set embedding cache


class MetricLearningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_dim = cfg.train.embedding_dim
        self.variance_dim = cfg.train.get('variance_dim')
        self.model = Model(
            setting=cfg.train.setting,
            mu_dim=self.embedding_dim,
            sigma_dim=self.variance_dim,
            dropout_rate=cfg.train.get('dropout_rate'),
        )
        self.lr = cfg.train.lr
        self.weight_decay = cfg.train.weight_decay
        self.lr_gamma = cfg.train.lr_gamma
        self.lr_setp = cfg.train.lr_step
        self.momentum = cfg.train.momentum
        self.loss = cfg.train.loss
        self.margin = cfg.train.margin
        self.setting = cfg.train.setting
        self.dataset = cfg.dataset.dataset
        self.eval_split = cfg.test.eval_split if cfg.get('test', None) is not None else 'val'

        self.eval_k_list = cfg.train.eval_k_list
        self.optim = cfg.train.optim
        self.batch_size = cfg.dataset.batch_size
        self.val_forward_num = cfg.train.get('val_forward_num', 20)
        if cfg.get('test', None) is None:
            self.output_dir = os.path.relpath(HydraConfig.get().runtime.output_dir)
        else:
            self.output_dir = dirname(cfg.test.ckpt.get(cfg.dataset.dataset, None))

        # for data communication
        self.embeddings_initialized = False
        self.mu_all = None
        self.variance_all = None
        self.gt_all = None

        if self.loss == 'triplet':
            self.criterion = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='sum')
        elif self.loss == 'bayesian_triplet':
            self.criterion = BayesianTripletLoss(margin=0, varPrior=1 / 2047.0)                    # 0.00004885

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        if self.optim == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), self.lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.lr_gamma, last_epoch=-1, verbose=False)
        elif self.optim == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gamma)
        else:
            raise NameError('Undefined optimizer :<')

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        dicts_to_log = {}

        anchor, positives, negatives, neg_counts, indices = batch
        if anchor is None:                                                                         # in case we get an empty batch
            return
        B = anchor.shape[0]                                                                        # ([8, 1, 3, 200, 200]), # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor, where N = batchSize * (nQuery + nPos + n_neg)
        n_neg = torch.sum(neg_counts)                                                              # tensor(80) = torch.sum(torch.Size([8]))
        input = torch.cat([anchor, positives, negatives])                                          # ([B, C, H, 200])  # ([96, 1, C, H, W])

        embs, variances = self.model(input)      # ([96, D])

        # track the range of variance
        if self.global_step % 100 == 0 and self.setting in ['btl', 'dul']:
            if self.setting == 'dul':
                dicts_to_log['sigma_sq/avg'] = torch.mean(variances[1]).item()
                dicts_to_log['sigma_sq/max'] = torch.max(variances[1]).item()
                dicts_to_log['sigma_sq/min'] = torch.min(variances[1]).item()
            else:
                dicts_to_log['sigma_sq/avg'] = torch.mean(variances).item()
                dicts_to_log['sigma_sq/max'] = torch.max(variances).item()
                dicts_to_log['sigma_sq/min'] = torch.min(variances).item()

        tuple_loss = 0
        if self.loss == 'triplet':
            embs_a, embs_p, embs_n = torch.split(embs, [B, B, n_neg])
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    negIx = (torch.sum(neg_counts[:i]) + n).item()
                    tuple_loss += self.criterion(embs_a[i:i + 1], embs_p[i:i + 1], embs_n[negIx:negIx + 1])
            tuple_loss /= n_neg.float()          # normalise by actual number of negatives
            if self.setting == 'dul':
                def kl_divergence(mu, sigma2):
                    logsigma2 = sigma2.log()
                    kl = -(1 + logsigma2 - mu.pow(2) - sigma2) / 2
                    kl = kl.sum(dim=1).mean()
                    return kl

                mu, sigma2 = variances
                kl_loss = kl_divergence(mu, sigma2)
                tuple_loss = tuple_loss + self.lambda_kl * kl_loss
        elif self.loss == 'bayesian_triplet':
            embs = torch.cat((embs, variances), dim=-1)
            embs_a, embs_p, embs_n = torch.split(embs, [B, B, n_neg])
            for i, neg_count in enumerate(neg_counts):
                emb_a = embs_a[i:i + 1]                                                            # (1, D)
                emb_p = embs_p[i:i + 1]                                                            # (1, D)
                st = torch.sum(neg_counts[:i])
                emb_n = embs_n[st:st + neg_count]                                                  # (neg_count, D)
                x = torch.cat((emb_a, emb_p, emb_n), axis=0).transpose(0, 1)                       # (1+1+neg_count, D)
                label = torch.cat((torch.tensor([-1, 1]), torch.zeros((neg_count, ))))
                tuple_loss += self.criterion(x, label)
            tuple_loss /= B                                                                        # NOTE divide by batch size

        self.log_dict(dicts_to_log, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=self.batch_size)
        self.log('train_loss', tuple_loss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        return tuple_loss

    def eval_step_common(self, batch, batch_idx):
        inputs, indices = batch
        if self.setting in ['btl', 'dul', 'triplet']:
            mu, variance = self.model(inputs)                                                      # (B, D)
            if self.setting in ['dul']:
                variance = variance[1]
        elif self.setting in ['mcd']:
            # self.model.train()                   # self.model.eval()
            outputs = [self.model(inputs) for _ in range(self.val_forward_num)]                    # (B, D),  According to Kendall(2016), 40 is enough for converge
            outputs_mean = torch.stack([mean for (mean, variance) in outputs])                     # ([40, 25, 2048])
            variance = torch.var(outputs_mean, dim=0)                                              # ([25, 2048])
            mu = torch.mean(outputs_mean, dim=0)                                                   # ([25, 2048])
            # mu, variance = self.model(inputs)
        self.mu_all[indices] = mu
        self.variance_all[indices] = variance


    def validation_step(self, batch, batch_idx):
        self.eval_step_common(batch, batch_idx)


    def test_step(self, batch, batch_idx):
        self.eval_step_common(batch, batch_idx)


    def compute_metrics(self):
        if self.dataset in ['cub200', 'car196', 'chestx', 'sop']:
            q_mu = self.mu_all.cpu().numpy()
            db_mu = self.mu_all.cpu().numpy()
            q_variance = self.variance_all.cpu().numpy()
            db_variance = self.variance_all.cpu().numpy()
            gt = self.gt_all
            dists, preds = utils.find_nn(q_mu, db_mu, max(self.eval_k_list) + 1)
            dists = dists[:, 1:]                                                                   # -1: to exclude query itself
            preds = preds[:, 1:]                                                                   # -1: to exclude query itself
        elif self.dataset in ['pitts']:
            if self.trainer.state.stage in ['validate', 'sanity_check']:
                numDb = self.trainer.datamodule.whole_val_set.dbStruct.numDb
            elif self.trainer.state.stage == 'test':
                numDb = self.trainer.datamodule.whole_test_set.dbStruct.numDb
            print(self.trainer.state.stage)
            q_mu = self.mu_all[numDb:].cpu().numpy()
            db_mu = self.mu_all[:numDb].cpu().numpy()
            q_variance = self.variance_all[numDb:].cpu().numpy()
            db_variance = self.variance_all[:numDb].cpu().numpy()
            dists, preds = utils.find_nn(q_mu, db_mu, max(self.eval_k_list) + 1)
            gt = self.gt_all
            val_inds = np.array([True if len(gt[ind]) != 0 else False for ind in range(len(gt))])  # cull queries without any ground truth positives in the database
            q_mu = q_mu[val_inds]
            q_variance = q_variance[val_inds]
            preds = preds[val_inds]
            dists = dists[val_inds]
            gt = gt[val_inds]
            dists, preds = utils.find_nn(q_mu, db_mu, max(self.eval_k_list) + 1)

        recall_at_k = utils.cal_recall(preds, gt, self.eval_k_list)
        return recall_at_k, q_mu, db_mu, q_variance, db_variance, preds, dists, gt


    def on_validation_epoch_end(self):
        recall_at_k, q_mu, db_mu, q_variance, db_variance, preds, dists, gt = self.compute_metrics()

        print(f"recall@k:{recall_at_k}")
        self.log_dict({f'val_recall@{k}': v for k, v in zip(self.eval_k_list[:1], recall_at_k[:1])}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({f'val_recall@{k}': v for k, v in zip(self.eval_k_list[1:3], recall_at_k[1:3])}, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if not self.trainer.sanity_checking:
            ece_recall, ece_image = self.compute_ece(preds, q_variance, gt)
            # self.logger.experiment.log({"reliability diagram": wandb.Image(ece_image)})
            self.log_dict({f'ece_recall@{k}': v for k, v in zip(self.eval_k_list[:3], ece_recall[:3])}, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            if self.trainer.checkpoint_callback.best_model_score is not None:
                self.log_dict({'best_score': self.trainer.checkpoint_callback.best_model_score.item()}, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def on_test_epoch_end(self):
        recall_at_k, q_mu, db_mu, q_variance, db_variance, preds, dists, gt = self.compute_metrics()

        print(f"recall@k:{recall_at_k}")
        self.log_dict({f'test_recall@{k}': v for k, v in zip(self.eval_k_list, recall_at_k)}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        ece_recall, _ = self.compute_ece(preds, q_variance, gt, plot=True)
        print(f"ece@recall_k:{ece_recall}")
        with open(join(self.output_dir, f"embeddings_{self.eval_split}.pickle"), 'wb') as handle:
            pickle.dump(q_mu, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(db_mu, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(q_variance, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(db_variance, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(gt, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def compute_ece(self, preds, variances, gt, plot=False):
        assert len(preds) == len(variances) == len(gt), 'length of preds, variances and gt should be the same :('
        num_bins = 11
        n_values = self.eval_k_list[:3]
        if variances.shape[-1] != 1:
            variances = variances.mean(axis=-1)
        variance_reduced = variances
        indices, _ = utils.get_bins(variance_reduced, num_bins)
        # indices, _, k = utils.get_zoomed_bins(variance_reduced, num_bins)
        bins_recall = np.zeros((num_bins - 1, len(n_values)))
        ece_bins_recall = np.zeros((num_bins - 1, len(n_values)))
        for index in range(num_bins - 1):
            if len(indices[index]) == 0:
                continue
            pred_bin = preds[indices[index]]
            gt_bin = [gt[i] for i in indices[index]]
            # calculate r@K
            recall_at_n = utils.cal_recall(pred_bin, gt_bin, n_values)
            bins_recall[index] = recall_at_n
            ece_bins_recall[index] = np.array([len(indices[index]) / variance_reduced.shape[0] * np.abs(recall_at_n[i] / 100.0 - (num_bins - 1 - index) / ((num_bins - 1))) for i in range(len(n_values))])

        ece_recall = ece_bins_recall.sum(axis=0)
        if plot == True:
            n_row, n_col = 1, 2
            fig, axs = plt.subplots(n_row, n_col, figsize=(5 * n_col, 5 * n_row), dpi=100, squeeze=False, tight_layout=True)
            axs = axs.ravel()
            ax = axs[0]
            ax.plot(np.arange(num_bins - 1), bins_recall[:, 0], marker='o', markersize=2)
            ax.plot(np.arange(num_bins - 1), bins_recall[:, 1], marker='o', markersize=2)
            ax.plot(np.arange(num_bins - 1), bins_recall[:, 2], marker='o', markersize=2)
            # ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('$\sigma^2$')
            ax.set_ylabel('recall@k')
            ax.xaxis.set_major_locator(MultipleLocator(1))
            matin.ax_default_style(ax)

            ax = axs[1]
            ax.bar(np.arange(len(indices)), [len(x) / variance_reduced.shape[0] for x in indices])
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_xlabel('$\sigma^2$')
            ax.set_ylabel('num of samples')
            # ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            matin.ax_default_style(ax)
            # fig.canvas.draw()
            # plt.savefig("demo.png")
            # ece_image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
            # ece_image = np.array(fig.canvas.renderer.buffer_rgba())
            if not exists(join(self.output_dir, 'ece')):
                os.makedirs(join(self.output_dir, 'ece'), exist_ok=True)
            plt.savefig(join(self.output_dir, 'ece', f"epoch_{self.current_epoch:03d}.png"))
            plt.close()

        return ece_recall, None


class CustomProgressBar(RichProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(self.trainer, )
        items.pop("v_num", None)
        items["f"] = os.path.relpath(HydraConfig.get().runtime.output_dir)

        return items
