import os
import torch
import torch_geometric
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf

# Local imports
from source.data import (
    EXPWL1DataModule, 
    BenchHardDataModule, 
    MultipartiteDataModule, 
    TUDataModule)
from source.pl_modules import GraphClassificationModule
from source.models import GraphClassificationModel
from source.utils import CustomTensorBoardLogger, register_resolvers, reduce_precision, find_devices

register_resolvers()
reduce_precision()


@hydra.main(version_base=None, config_path="config", config_name="run_graph_classification")
def run(cfg : DictConfig) -> float:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ### 🌱 Seed everything
    if 'seed' in cfg.dataset.hparams:
        print(f"Setting seed to {cfg.dataset.hparams.seed}")
        torch_geometric.seed.seed_everything(cfg.dataset.hparams.seed)

    ### 📊 Load data
    if cfg.dataset.name in ['EXPWL1']:
        data_module = EXPWL1DataModule(cfg.dataset.hparams)
    elif cfg.dataset.name in ['BenchHard']:
        data_module = BenchHardDataModule(cfg.dataset.hparams)
    elif cfg.dataset.name in ['Multipartite']:
        data_module = MultipartiteDataModule(cfg.dataset.hparams, cfg.dataset.params)
    elif cfg.dataset.name in ['TUDataset']:
        data_module = TUDataModule(cfg.dataset.hparams)
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not recognized")
        
    ### 🧠 Load the model
    torch_model = GraphClassificationModel(
        in_channels=data_module.num_features,                                   # Size of node features
        out_channels=data_module.num_classes,                                   # Number of classes
        edge_channels=data_module.num_edge_features,                            # Size of edge features
        num_layers_pre=cfg.architecture.hparams.num_layers_pre,                 # Number of GIN layers before pooling
        num_layers_post=cfg.architecture.hparams.num_layers_post,               # Number of GIN layers after pooling
        hidden_channels=cfg.architecture.hparams.hidden_channels,               # Dimensionality of node embeddings
        activation=cfg.architecture.hparams.activation,                         # Activation of the MLP in GIN 
        pooler=cfg.pooler.name,                                                 # Pooling method
        pool_kwargs=cfg.pooler.hparams,                                         # Pooling method kwargs
        pooled_nodes=int(
            data_module.avg_nodes*cfg.architecture.hparams.pool_ratio),         # Number of nodes after pooling 
        aux_net_kwargs=getattr(cfg.pooler, 'aux_net', None),                    # Auxiliar network kwargs
        use_gine=True if data_module.num_edge_features is not None else False,  # Use GINE instead of GIN
        )

    ### 📈 Optimizer scheduler
    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    ### ⚡ Lightning module
    lightning_model = GraphClassificationModule(
        model=torch_model,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        log_lr=cfg.log_lr,
        log_grad_norm=cfg.log_grad_norm,
        plot_dict=dict(cfg.plot_preds_at_epoch)
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
        mode=cfg.callbacks.mode
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
    
    trainer.fit(lightning_model, data_module.train_dataloader(), data_module.val_dataloader())
    val_loss = trainer.callback_metrics["val_loss"].item() # Used by the sweeper to optimize the hyperparameters

    if cfg.callbacks.checkpoints:
        trainer.test(lightning_model, data_module.test_dataloader(), ckpt_path='best')
    else:
        trainer.test(lightning_model, data_module.test_dataloader())
    
    logger.finalize('success')

    return val_loss

if __name__ == "__main__":
    run()