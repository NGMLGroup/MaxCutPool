from typing import Optional, Mapping, Type
import torch
import torchmetrics
from torchmetrics.clustering import NormalizedMutualInfoScore, HomogeneityScore, CompletenessScore
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

from source.layers.ndp import eval_cut
from .base_module import BaseModule


class CutModule(BaseModule):
    """
    Lightning module to perform MaxCut with MaxCutPool 🎱
    """
    def __init__(self,
                 model: Optional[torch.nn.Module] = None,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 log_lr: bool = True,
                 log_grad_norm: bool = False,
                 sync_dist: bool = False,   # if ``True``, reduces the metric across devices. Causes overhead. Use only for multi-gpu train
                 plot_dict: Optional[Mapping] = None
                 ):
        super().__init__(optim_class, optim_kwargs, scheduler_class, scheduler_kwargs,
                         log_lr, log_grad_norm, sync_dist)

        self.model = model 
        self.sync_dist = sync_dist
        self.plot_preds_at_epoch = plot_dict


    def forward(self, data):
        """
        ⏩ 
        """
        x, kept_nodes, score, aux_loss = self.model(data)

        return x, kept_nodes, score, aux_loss

    def training_step(self, batch, batch_idx):
        """
        🐾
        """
        y = batch.y
        _, kept_nodes, score, loss = self.forward(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, 
                 prog_bar=True, sync_dist=self.sync_dist, batch_size=1)

        # Log images (maxcutpool only)
        num_nodes = batch.x.shape[0]
        if 'train' in self.plot_preds_at_epoch['set'] and num_nodes <= 1000:
            if self.logger is not None:
                if self.logger.cfg.logger.backend=='tensorboard' and self.logger.cfg.pooler.name=='maxcutpool':
                    self.maybe_log_maxcutpool(batch, kept_nodes, score, None, batch_idx, None, None, ['selected_nodes', 'score'])

        return {'loss':loss}
    

    def test_step(self, batch, batch_idx):
        """Perform evaluation step and logs metrics.
        
        Computes:
        - Cut size and number of cut edges
        - Loss value
        - Visualization of cuts if configured
        
        Args:
            batch (Data): Input graph batch
            batch_idx (int): Index of current batch
        """
        y = batch.y
        _, kept_nodes, score, loss = self.forward(batch)

        rounded_score = torch.where(score > 0, torch.tensor(1.), torch.tensor(-1.)).detach().cpu().numpy()
        edge_index_L, edge_weight_L = get_laplacian(batch.edge_index, batch.edge_weight, normalization=None)
        num_edges = batch.edge_index.shape[1]
        num_nodes = batch.x.shape[0]
        L = to_scipy_sparse_matrix(edge_index_L, edge_weight_L, num_nodes).tocsr()

        cut_size = eval_cut(num_edges, L, rounded_score)
        cut_edges = int(cut_size * num_edges)

        self.log('test_loss', loss, sync_dist=self.sync_dist, batch_size=1)
        self.log('cut_size', cut_size, sync_dist=self.sync_dist, batch_size=1)
        self.log('cut_edges', cut_edges, sync_dist=self.sync_dist, batch_size=1)


        # Log images
        if 'test' in self.plot_preds_at_epoch['set'] and num_nodes <= 1000:
            if self.logger is not None:
                if self.logger.cfg.logger.backend=='tensorboard' and self.logger.cfg.pooler.name=='maxcutpool':
                    self.maybe_log_maxcutpool(batch, kept_nodes, score, None, batch_idx, None, None, ['score', 'rounded_score'], istest=True)
