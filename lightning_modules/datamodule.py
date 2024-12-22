import lightning.pytorch as pl
import importlib
from torch.utils.data import DataLoader
from joblib.externals.loky.backend.context import get_context
from hydra.core.hydra_config import HydraConfig


class MetricLearningDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.dataset = cfg.dataset.dataset
        self.data_path = cfg.dataset.data_path
        self.batch_size = cfg.dataset.batch_size
        self.cache_batch_size = cfg.dataset.cache_batch_size
        self.threads = cfg.dataset.threads
        self.margin = cfg.train.margin
        self.multi_run = HydraConfig.get().runtime.choices["hydra/sweeper"] == "optuna"
        self.dataset_cls = importlib.import_module('datasets.' + cfg.dataset.dataset)
        self.eval_split = cfg.test.eval_split if cfg.get('test', None) is not None else 'val'

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        #
        self.whole_train_set = self.dataset_cls.Whole('train', data_path=self.data_path, aug=True)

        self.whole_training_data_loader = DataLoader(
            dataset=self.whole_train_set,
            num_workers=self.threads,
            batch_size=self.cache_batch_size,
            shuffle=True,
            pin_memory=True,
            multiprocessing_context=get_context('loky') if self.multi_run else None,
        )

        pass

    def train_dataloader(self):
        # train_split = Dataset(...)
        # return DataLoader(train_split)
        self.train_set = self.dataset_cls.Tuple('train', data_path=self.data_path, margin=self.margin)
        self.training_data_loader = DataLoader(
            dataset=self.train_set,
            num_workers=self.threads,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.dataset_cls.collate_fn,
            multiprocessing_context=get_context('loky') if self.multi_run else None,
        )

        return self.training_data_loader

    def val_dataloader(self):
        # val_split = Dataset(...)
        # return DataLoader(val_split)
        self.whole_val_set = self.dataset_cls.Whole('val', data_path=self.data_path, aug=False)
        self.whole_val_data_loader = DataLoader(
            dataset=self.whole_val_set,
            num_workers=self.threads,
            batch_size=self.cache_batch_size,
            shuffle=False,
            pin_memory=True,
            multiprocessing_context=get_context('loky') if self.multi_run else None,
        )

        return self.whole_val_data_loader

    def test_dataloader(self):
        # test_split = Dataset(...)
        # return DataLoader(test_split)
        self.whole_test_set = self.dataset_cls.Whole(self.eval_split, data_path=self.data_path, aug=False)
        self.whole_test_data_loader = DataLoader(
            dataset=self.whole_test_set,
            num_workers=self.threads,
            batch_size=self.cache_batch_size,
            shuffle=False,
            pin_memory=True,
            multiprocessing_context=get_context('loky') if self.multi_run else None,
        )

        return self.whole_test_data_loader
