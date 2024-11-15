from typing import Optional, Mapping, Type
import pytorch_lightning as pl
from lightning.pytorch.utilities import grad_norm
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
import torch

class BaseModule(pl.LightningModule):
    """
    🧱 Base Lightning Module class, it contains the logging logic
    """
    def maybe_log_maxcutpool(self, batch, kept_nodes, score, mappa, batch_idx, pooled_adj, pooled_batch, plot_type=[], istest=False):

        if self.plot_preds_at_epoch is not None:
            b_idx = self.plot_preds_at_epoch.get('batch', 0)
            
            s_idx = self.plot_preds_at_epoch.get('samples', 1)
            every = self.plot_preds_at_epoch.get('every', 1)            

            if batch_idx == b_idx and (self.current_epoch%every==0 or istest):
                
                mask = batch.batch[batch.edge_index[0]] == s_idx
                sample_edge_index = batch.edge_index[:, mask]
                sample_edge_index = sample_edge_index - batch.ptr[s_idx]
                sample_num_nodes = batch.ptr[s_idx+1] - batch.ptr[s_idx]

                # mask = (batch.edge_index[0] < batch.ptr[s_idx+1]) & (batch.edge_index[1] < batch.ptr[s_idx+1]) & (batch.edge_index[0] >= batch.ptr[s_idx]) & (batch.edge_index[1] >= batch.ptr[s_idx])
                # sample_edge_index = batch.edge_index[:, mask] - batch.ptr[s_idx]
                if batch.x.shape[-1] == 2:
                    pos=batch[s_idx].x.detach().cpu().numpy()
                else:
                    pos=None

                # Log nodes selected by MaxCutPool
                if 'selected_nodes' in plot_type:
                    title = f'visuals/selected_nodes' #@ep{self.current_epoch}
                    if istest:
                        title += '_test'
                    selected_nodes = np.zeros(batch.x.shape[0])
                    selected_nodes[kept_nodes.cpu().numpy()] = 1
                    sample_selected = selected_nodes[batch.batch.cpu()==s_idx]
                    self.logger.log_nx_graph_plot(to_scipy_sparse_matrix(sample_edge_index, num_nodes=sample_num_nodes), 
                                                signal=sample_selected, 
                                                pos=pos,
                                                name=title,
                                                global_step=self.global_step
                                                )
                
                if 'score' in plot_type:
                    title = f'visuals/score'
                    if istest:
                        title += '_test'
                    
                    sample_score = score[batch.batch.cpu()==s_idx].detach().cpu().numpy()

                    self.logger.log_nx_graph_plot(to_scipy_sparse_matrix(sample_edge_index, num_nodes=sample_num_nodes), 
                                                        signal=sample_score, 
                                                        pos=pos,
                                                        name=title,
                                                        global_step=self.global_step)

                if 'absolute_score' in plot_type:
                    title = f'visuals/absolute_score'
                    if istest:
                        title += '_test'
                    
                    sample_score = score[batch.batch.cpu()==s_idx].detach().cpu().numpy()
                    #commentare riga sotto
                    sample_score = np.abs(sample_score)

                    self.logger.log_nx_graph_plot(to_scipy_sparse_matrix(sample_edge_index, num_nodes=sample_num_nodes), 
                                                        signal=sample_score, 
                                                        cmap='seismic',
                                                        vmin=-1,
                                                        vmax=1.,
                                                        pos=pos,
                                                        name=title,
                                                        global_step=self.global_step)

                if 'rounded_score' in plot_type:
                    title = f'visuals/rounded_score'
                    if istest:
                        title += '_test'
                    
                    sample_score = score[batch.batch.cpu()==s_idx].detach().cpu().numpy()
                    sample_score = np.round(sample_score)

                    self.logger.log_nx_graph_plot(to_scipy_sparse_matrix(sample_edge_index, num_nodes=sample_num_nodes), 
                                                        signal=sample_score, 
                                                        vmin=-1,
                                                        vmax=1.,
                                                        pos=pos,
                                                        name=title,
                                                        global_step=self.global_step)

                if 'pooled_graph' in plot_type:
                    title = f'visuals/pooled_graph'
                    if istest:
                        title += '_test'

                    pooled_mask = (pooled_batch[pooled_adj][0] == s_idx)
                    pooled_edge_index = pooled_adj[:,pooled_mask]
                    pooled_num_nodes = (pooled_batch == s_idx).sum().item()

                    if pooled_edge_index.numel() > 0:
                        pooled_edge_index = pooled_edge_index - torch.min(pooled_edge_index)
                    else:
                        pooled_edge_index = torch.tensor([[],[]], device=pooled_adj.device)
                    self.logger.log_nx_graph_plot(to_scipy_sparse_matrix(pooled_edge_index, num_nodes=pooled_num_nodes), 
                            pos=None,
                            with_labels=True,
                            name=title,
                            global_step=self.global_step)
                
                # Log cluster assignments
                if 'assignments' in plot_type:
                    title = f'visuals/assignments'
                    if istest:
                        title += '_test'
                    selected_nodes = np.zeros(batch.x.shape[0])
                    selected_nodes[kept_nodes.cpu().numpy()] = 1
                    sample_selected = selected_nodes[batch.batch.cpu()==s_idx]

                    label_0 = mappa[0, batch.batch.cpu()==s_idx] - batch.ptr[s_idx]
                    label_1 = mappa[1, batch.batch.cpu()==s_idx] - torch.min(mappa[1, batch.batch.cpu()==s_idx])

                    labels = torch.stack([label_0, label_1], dim=0).detach().cpu().numpy()

                    labels = {labels[0, i]: labels[1, i] for i in range(labels.shape[1])}
                    self.logger.log_nx_graph_plot(to_scipy_sparse_matrix(sample_edge_index, num_nodes=sample_num_nodes), 
                                                    signal=sample_selected, 
                                                    pos=pos,
                                                    node_size=25,
                                                    font_size=8,
                                                    cmap='Pastel2',
                                                    with_labels=True,
                                                    labels=labels,
                                                    name=title,
                                                    global_step=self.global_step)

    def __init__(self,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 log_lr: bool = True,
                 log_grad_norm: bool = False,
                 sync_dist: bool = False,   # if ``True``, reduces the metric across devices. Causes overhead. Use only for multi-gpu train
                 ):
        super().__init__()
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or dict()
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or dict()
        self.log_lr = log_lr
        self.log_grad_norm = log_grad_norm
        self.sync_dist = sync_dist

        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            dict: Configuration containing:
                - optimizer: The optimizer instance
                - lr_scheduler: The learning rate scheduler (if configured)
                - monitor: Metric to monitor (if configured)
        """
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg['optimizer'] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            scheduler = self.scheduler_class(optimizer,
                                             **self.scheduler_kwargs)
            cfg['lr_scheduler'] = scheduler
            if metric is not None:
                cfg['monitor'] = metric
        return cfg


    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimizer step if configured.
        
        Args:
            optimizer: The optimizer about to perform a step
        """
        if self.log_grad_norm:
            self.log_dict(grad_norm(self, norm_type=2))


    def on_train_epoch_start(self) -> None:
        """Log learning rate at the start of each training epoch."""
        if self.log_lr:
            optimizers = self.optimizers()
            if isinstance(optimizers, list):
                for i, optimizer in enumerate(optimizers):
                    lr = optimizer.optimizer.param_groups[0]['lr']
                    self.log(f'lr_{i}', lr, on_step=False, on_epoch=True,
                             logger=True, prog_bar=False, batch_size=1, sync_dist=self.sync_dist)
            else:
                lr = optimizers.optimizer.param_groups[0]['lr']
                self.log(f'lr', lr, on_step=False, on_epoch=True,
                         logger=True, prog_bar=False, batch_size=1, sync_dist=self.sync_dist)