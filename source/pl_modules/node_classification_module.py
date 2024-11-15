from typing import Optional, Mapping, Type
import torch
import torchmetrics
from torchmetrics.classification import Accuracy, AUROC

from .base_module import BaseModule


class NodeClassificationModule(BaseModule):
    """
    Lightning module to perform node classification with graph pooling 🎱
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
                 plot_dict: Optional[Mapping] = None,
                 fold: Optional[int] = None,
                 ):
        super().__init__(optim_class, optim_kwargs, scheduler_class, scheduler_kwargs,
                         log_lr, log_grad_norm, sync_dist)

        self.model = model 
        self.fold = fold
        self.sync_dist = sync_dist
        self.plot_preds_at_epoch = plot_dict
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_metrics = torchmetrics.MetricCollection({
            'train_acc': Accuracy(task='multiclass', num_classes=model.num_classes),
            })
        self.train_auroc = AUROC(task='multiclass', num_classes=model.num_classes)
        self.val_metrics = torchmetrics.MetricCollection({
            'val_acc': Accuracy(task='multiclass', num_classes=model.num_classes),
            })
        self.val_auroc = AUROC(task='multiclass', num_classes=model.num_classes)
        self.test_metrics = torchmetrics.MetricCollection({
            'test_acc': Accuracy(task='multiclass', num_classes=model.num_classes),
            })
        self.test_auroc = AUROC(task='multiclass', num_classes=model.num_classes)


    def forward(self, data):
        """
        ⏩ 
        """
        logits, pooled_adj, edge_weight, pooled_batch, kept_nodes, score, aux_loss, s, maps_list = self.model(data)

        return logits, pooled_adj, pooled_batch, kept_nodes, score, aux_loss, s, maps_list

    def training_step(self, batch, batch_idx):
        """
        🐾
        """

        logits, pooled_adj, pooled_batch, kept_nodes, score, aux_loss, s, _ = self.forward(batch)
        mask = batch.train_mask if self.fold is None else batch.train_mask[:, self.fold]
        clf_loss = self.loss(logits[mask], batch.y[mask])
        loss = clf_loss + aux_loss

        # Log losses and metrics
        self.train_metrics.update(logits[mask].argmax(1).detach().int(), 
                                  batch.y[mask].int())
        self.train_auroc.update(logits[mask], batch.y[mask])
        self.log('train_clf_loss', clf_loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log('train_aux_loss', aux_loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log('train_loss', loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=self.sync_dist)
        self.log('train_acc', self.train_metrics['train_acc'], batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log('train_auroc', self.train_auroc, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)

        # Log images (maxcutpool only)
        if 'train' in self.plot_preds_at_epoch['set']:
            if self.logger is not None:
                if self.logger.cfg.logger.backend=='tensorboard' and self.logger.cfg.pooler.name=='maxcutpool':
                    self.maybe_log_maxcutpool(batch, kept_nodes, score, 
                                           s, batch_idx, 
                                           pooled_adj, pooled_batch, 
                                           ['selected_nodes', 'score'])

        return {'loss':loss}
    

    def validation_step(self, batch, batch_idx):
        """
        🐾
        """
        logits, pooled_adj, pooled_batch, kept_nodes, score, aux_loss, s, _ = self.forward(batch)
        mask = batch.val_mask if self.fold is None else batch.train_mask[:, self.fold]
        clf_loss = self.loss(logits[mask], batch.y[mask])
        loss = clf_loss + aux_loss

        # Log losses and metrics
        self.val_metrics.update(logits[mask].argmax(1).detach().int(),
                                batch.y[mask].int())
        self.val_auroc.update(logits[mask], batch.y[mask])
        self.log('val_clf_loss', clf_loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log('val_aux_loss', aux_loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log('val_loss', loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=self.sync_dist)
        self.log('val_acc', self.val_metrics['val_acc'], batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log('val_auroc', self.val_auroc, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        
        # Log images
        if 'val' in self.plot_preds_at_epoch['set']:
            if self.logger is not None:
                if self.logger.cfg.logger.backend=='tensorboard' and self.logger.cfg.pooler.name=='maxcutpool':
                    self.maybe_log_maxcutpool(batch, kept_nodes, score, 
                                           s, batch_idx, 
                                           pooled_adj, pooled_batch, 
                                           ['selected_nodes', 'score', 'assignments', 'absolute_score', 'pooled_graph'])

        return {'val_loss':loss}
    

    def test_step(self, batch, batch_idx):
        """
        🧪
        """
        logits, pooled_adj, pooled_batch, kept_nodes, score, aux_loss, s, _ = self.forward(batch)
        mask = batch.test_mask if self.fold is None else batch.train_mask[:, self.fold]
        clf_loss = self.loss(logits[mask], batch.y[mask])
        loss = clf_loss + aux_loss

        # Log losses and metrics
        self.test_metrics.update(logits[mask].argmax(1).detach().int(),
                                 batch.y[mask].int())
        self.test_auroc.update(logits[mask], batch.y[mask])
        self.log('test_clf_loss', clf_loss, batch_size=batch.y.size(0), sync_dist=self.sync_dist)
        self.log('test_aux_loss', aux_loss, batch_size=batch.y.size(0), sync_dist=self.sync_dist)
        self.log('test_loss', loss, batch_size=batch.y.size(0), sync_dist=self.sync_dist)
        self.log('test_acc', self.test_metrics['test_acc'], batch_size=batch.y.size(0), sync_dist=self.sync_dist)
        self.log('test_auroc', self.test_auroc, batch_size=batch.y.size(0), sync_dist=self.sync_dist)
        
        # Log images
        if 'test' in self.plot_preds_at_epoch['set']:
            if self.logger is not None:
                if self.logger.cfg.logger.backend=='tensorboard' and self.logger.cfg.pooler.name=='maxcutpool':
                    self.maybe_log_maxcutpool(batch, kept_nodes, score, 
                                           s, batch_idx, 
                                           pooled_adj, pooled_batch, 
                                           ['selected_nodes', 'score', 'assignments', 'absolute_score', 'pooled_graph'], 
                                           istest=True)

        return {'test_loss':loss}