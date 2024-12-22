#%%
import os
from datetime import datetime
from os.path import basename, dirname, exists
import logging

import hydra
import lightning.pytorch as pl
import torch
from hydra import utils
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from lightning_modules.datamodule import MetricLearningDataModule
from lightning_modules.lightning import MetricLearningModule, TrainExt, CustomProgressBar
from utils import utils

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(cfg.train.seed, workers=True)

    loggers = []
    if cfg.get('test', None) is None:
        output_dir = os.path.relpath(HydraConfig.get().runtime.output_dir)
        if cfg.train.logger_wandb:
            prefix = 'op_' if HydraConfig.get().runtime.choices["hydra/sweeper"] == "optuna" else ''
            logger_wandb = WandbLogger(project='RCIR', name=f"{prefix}{basename(output_dir)}", save_dir=output_dir, log_model=False, offline=False)
            logger_wandb.log_hyperparams(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
            loggers.append(logger_wandb)
        if cfg.train.logger_csv:
            logger_csv = CSVLogger(save_dir=output_dir, name="train_csv", flush_logs_every_n_steps=1, version=datetime.now().strftime('%m%d_%H%M%S'))
            logger_csv.log_hyperparams(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
            loggers.append(logger_csv)
    else:
        assert cfg.test.ckpt.get(cfg.dataset.dataset, None) is not None, f"ckpt: {cfg.test.ckpt.get(cfg.dataset.dataset, None)} :("
        print(f"ckpt: {cfg.test.ckpt.get(cfg.dataset.dataset, None)}")
        output_dir = dirname(cfg.test.ckpt.get(cfg.dataset.dataset, None))
        logger_csv = CSVLogger(save_dir=output_dir, name="test_csv", flush_logs_every_n_steps=1, version=datetime.now().strftime('%m%d_%H%M%S'))
        logger_csv.log_hyperparams(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        loggers.append(logger_csv)

    if cfg.train.devices == 'schedule':
        cfg.train.devices = [utils.schedule_device()]

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        devices=cfg.train.devices,
        max_epochs=cfg.train.n_epoch,                                                                                                                # cfg.train.n_epoch
        logger=loggers,
        callbacks=[
            ModelCheckpoint(monitor='val_recall@1', mode='max', save_last=cfg.dataset.get('save_last', False), filename='best'),
            EarlyStopping(monitor='val_recall@1', mode='max', min_delta=0.0005, patience=cfg.train.patience, strict=True, log_rank_zero_only=True),
            LearningRateMonitor(logging_interval='step'),
            RichProgressBar(),
            TrainExt()
        ],
        num_sanity_val_steps=cfg.get('num_sanity_val_steps', 10),
        limit_train_batches=cfg.get('limit_train_batches', None),
        deterministic=True,
    )                                                                                                                                                # limit_train_batches=30,

    ml = MetricLearningModule(cfg)
    ml_data = MetricLearningDataModule(cfg)
    if cfg.get('test', None) is None:
        trainer.fit(model=ml, datamodule=ml_data, ckpt_path=cfg.train.get('resume_ckpt', None))
        # trainer.test(model=ml, datamodule=ml_data, ckpt_path=trainer.checkpoint_callback.best_model_path)
        return trainer.checkpoint_callback.best_model_score
    else:
        trainer.test(model=ml, datamodule=ml_data, ckpt_path=cfg.test.ckpt.get(cfg.dataset.dataset, None))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_fake(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # generate a random value
    # print(HydraConfig.get().runtime.choices["hydra/sweeper"])
    loss = torch.rand(1).item()
    log.info("Info level message")
    log.debug("Debug level message")
    return loss


if __name__ == '__main__':
    # run_fake()
    run()
