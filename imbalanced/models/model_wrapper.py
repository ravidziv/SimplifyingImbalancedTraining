import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import numpy as np
import os
def update_ens(all_preds, sgd_ens_preds, n_ensembled):
    if sgd_ens_preds is None:
        sgd_ens_preds = all_preds.copy()
    else:
        # TODO: rewrite in a numerically stable way
        sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                n_ensembled + 1
        ) + all_preds / (n_ensembled + 1)
    n_ensembled += 1
    return sgd_ens_preds, n_ensembled


class ModelWrapper(pl.LightningModule):
    def __init__(self, base_model, lr=1e-3, momentum=0.9, wd=1e-4, c_loss=F.cross_entropy, epochs=200,
                 start_samples=150):
        super().__init__()
        self.lr = lr
        self.base_model = base_model
        self.momentum = momentum
        self.wd = wd
        self.c_loss = c_loss
        self.epochs = epochs
        self.start_samples = start_samples
        self.sgd_ens_preds = None
        self.n_ensembled = 0
        self.save_hyperparameters()

    def forward(self, x):
        preds = self.base_model(x)
        return preds

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        preds = self(x)
        loss = self.c_loss(preds, y)
        acc = accuracy(preds, y)
        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, y, y_hat = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        metrics['val_pred'] = y_hat
        metrics['val_labels'] = y
        return metrics

    def validation_epoch_end(self, outs):
        all_preds = torch.stack([out_i['val_pred'] for out_i in outs]).reshape((-1, 10))
        all_labels = torch.stack([out_i['val_labels'] for out_i in outs]).reshape(-1)
        epoch = self.trainer.current_epoch
        if epoch + 1 > self.start_samples:
            if self.sgd_ens_preds is None:
                self.sgd_ens_preds = all_preds
            self.sgd_ens_preds, self.n_ensembled = update_ens(
                all_preds=all_preds, sgd_ens_preds=self.sgd_ens_preds, n_ensembled=self.n_ensembled)
            loss = self.c_loss(self.sgd_ens_preds, all_labels)
            acc = accuracy(self.sgd_ens_preds, all_labels)
            metrics = {"val_ens_accc": acc, "val_ens_loss": loss}
            self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
            dir =self.logger.save_dir
            np.savez(
                os.path.join(dir, f"sgd_ens_preds.npz"),
                predictions=self.sgd_ens_preds.cpu(),
                targets=all_labels.cpu(),
            )

    def test_step(self, batch, batch_idx):
        loss, acc, y, y_hat = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.base_model(x)
        return y_hat

    def _shared_eval_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.c_loss(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc, y, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]
