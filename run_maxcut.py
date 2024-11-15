import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.profile import count_parameters

# Local imports
from source.data import PyGSPDataset, GsetDataset
from source.pl_modules import CutModule
from source.models import CutModel
from source.utils import CustomTensorBoardLogger, register_resolvers, reduce_precision, find_devices

register_resolvers()
reduce_precision()


@hydra.main(version_base=None, config_path="config", config_name="run_maxcut")
def run(cfg: DictConfig) -> float:
    """Main training function for MaxCut experiments.
    
    Args:
        cfg (DictConfig): Hydra configuration object containing all experiment parameters
        
    Returns:
        float: Final performance metric
    """

    print(OmegaConf.to_yaml(cfg, resolve=True))


    ### 📊 Load data 
    if cfg.dataset.name=='PyGSPDataset':
        torch_dataset = PyGSPDataset(root='data/PyGSP', name=cfg.dataset.hparams.dataset, 
                                     kwargs=cfg.dataset.params, force_reload=cfg.dataset.hparams.reload)
    elif cfg.dataset.name=='GsetDataset':
        torch_dataset = GsetDataset(root='data/Gset', name=cfg.dataset.hparams.dataset, 
                                    directed=cfg.dataset.hparams.directed, force_reload=cfg.dataset.hparams.reload)
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not recognized")
      
    data_loader = DataLoader(torch_dataset, batch_size=cfg.batch_size, shuffle=False)


    ### 🧠 Load the model


    torch_model = CutModel(
        in_channels=torch_dataset.num_features,                        # Size of node features
        hidden_channels=cfg.architecture.hparams.hidden_channels,              # Dimensionality of node embeddings
        num_layers_pre=cfg.architecture.hparams.num_layers_pre,                # Number of GIN layers before pooling
        pool_kwargs=cfg.pooler.hparams,                                        # Pooling method kwargs
        aux_net_kwargs=getattr(cfg.pooler, 'aux_net', None)            # Auxiliar network kwargs
        )

    num_parameters = count_parameters(torch_model)
    print(f"Number of parameters: {num_parameters}")
    

    ### 📈 Optimizer scheduler
    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None


    ### ⚡ Lightning module
    lightning_model = CutModule(
        model=torch_model,
        optim_class=getattr(torch.optim, 'Adam'),
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
            filename=cfg.architecture.name + "_" + cfg.pooler.name + "___{epoch:03d}-{NMI:e}",
        )
        cb.append(checkpoint_callback)


    ### 🚀 Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=cb, 
        devices=find_devices(1), # Num of GPUs available
        max_epochs=cfg.epochs, 
        gradient_clip_val=cfg.clip_val,
        accelerator='gpu',
        overfit_batches=0.0, # >0 for debugging
        )

    trainer.fit(lightning_model, data_loader)

    if cfg.callbacks.checkpoints:
        trainer.test(lightning_model, data_loader, ckpt_path='best')
    else:
        trainer.test(lightning_model, data_loader)

    logger.finalize('success')

    return trainer.callback_metrics["test_loss"].item()

if __name__ == "__main__":
    run()