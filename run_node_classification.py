import os
import torch
import torch_geometric

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.transforms import NormalizeFeatures

# Local imports
from source.pl_modules import NodeClassificationModule
from source.models import NodeClassificationModel
from source.utils import CustomTensorBoardLogger, register_resolvers, reduce_precision, find_devices
from source.data import NodeClassDataModule

register_resolvers()
reduce_precision()


@hydra.main(version_base=None, config_path="config", config_name="run_node_classification")
def run(cfg : DictConfig) -> float:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ### 🌱 Seed everything
    if 'seed' in cfg.dataset.hparams:
        print(f"Setting seed to {cfg.dataset.hparams.seed}")
        torch_geometric.seed.seed_everything(cfg.dataset.hparams.seed)

    ### 📊 Load data
    trans = NormalizeFeatures() if cfg.dataset.hparams.norm_feats else None
    data_module = NodeClassDataModule(cfg.dataset.name, transform=trans,
                                           **cfg.dataset.hparams)
        
    ### 🧠 Load the model
    if cfg.architecture.model == 'bottleneck':
        torch_model = NodeClassificationModel(
            in_channels=data_module.torch_dataset.num_features, 
            out_channels=data_module.torch_dataset.num_classes, 
            hidden_channels=cfg.architecture.hparams.hidden_channels, 
            num_mp_layers=cfg.architecture.hparams.num_mp_layers, 
            activation=cfg.architecture.hparams.activation, 
            pooler=cfg.pooler.name, 
            pool_kwargs=cfg.pooler.hparams, 
            pooled_nodes=int(
                data_module.torch_dataset[0].x.size(0)*cfg.architecture.hparams.pool_ratio),
            aux_net_kwargs=getattr(cfg.pooler, 'aux_net', None),
            use_gine_enc=cfg.architecture.hparams.use_gine_enc,
            use_gine_bottleneck=cfg.architecture.hparams.use_gine_bottleneck,
            lift_zero_pad=cfg.architecture.hparams.lift_zero_pad,
            res_connect=cfg.architecture.hparams.res_connect,
            dropout=cfg.architecture.hparams.dropout,
            )
    else:
        raise NotImplementedError(f"Model {cfg.architecture.model} not implemented")

    ### 📈 Optimizer scheduler
    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    ### ⚡ Lightning module
    lightning_model = NodeClassificationModule(
        model=torch_model,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        log_lr=cfg.log_lr,
        log_grad_norm=cfg.log_grad_norm,
        plot_dict=dict(cfg.plot_preds_at_epoch),
        fold=data_module.fold,
        )

    ### 🪵 Logger 
    if cfg.get('logger').get('backend') is None:
        logger = None
    elif cfg.logger.backend == 'tensorboard':
        logger = CustomTensorBoardLogger(save_dir=cfg.logger.logdir, name=None, version='')
        logger.cfg = cfg
    else:
        raise NotImplementedError("Logger backend not supported.")
    
    ### 📞 Callbacks
    early_stop_callback = EarlyStopping(
        monitor=cfg.callbacks.monitor,
        patience=cfg.callbacks.patience,
        mode=cfg.callbacks.mode,
    )
    cb = [early_stop_callback]
    
    if cfg.callbacks.checkpoints:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor=cfg.callbacks.monitor,
            mode=cfg.callbacks.mode,
            dirpath=cfg.logger.logdir+"/checkpoints/", 
            filename=cfg.architecture.name + "_" + cfg.pooler.name + "___{epoch:03d}-{val_acc:e}",
        )
        cb.append(checkpoint_callback)


    ### 🚀 Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=cb, 
        devices=find_devices(1), # Num of GPUs available
        max_epochs=cfg.epochs, 
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        gradient_clip_val=cfg.clip_val,
        accelerator='gpu', # mettere anche caso per non gpu
        overfit_batches=0.0, # >0 for debugging
        )
    
    trainer.fit(lightning_model, data_module.dataloader(), data_module.dataloader())
    val_loss = trainer.callback_metrics["val_loss"].item() # Used by the sweeper to optimize the hyperparameters


    if cfg.callbacks.checkpoints:
        trainer.test(lightning_model, data_module.dataloader(), ckpt_path='best')
    else:
        trainer.test(lightning_model, data_module.dataloader())
    
    logger.finalize('success')

    return val_loss

if __name__ == "__main__":
    run()