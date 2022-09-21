import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import numpy as np
import os
from imbalanced.optimizers.sam import SAM


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
                 start_samples=150, recalibrated=False, calibrated_factor=None, args=None):
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
        # self.save_hyperparameters()
        self.recalibrated = recalibrated
        self.calibrated_factor = calibrated_factor
        self.args = args

    def forward(self, x):
        preds = self.base_model(x)
        return preds

    def training_step(self, batch, batch_idx):
        metrics, y, pred = self.shared_step(batch, name= 'train')
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)
        return metrics['train_loss']

    def validation_step(self, batch, batch_idx):

        metrics, y, pred = self._shared_eval_step(batch, batch_idx, name='val')
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        metrics['val_pred'] = pred
        metrics['val_labels'] = y
        return metrics

    def validation_epoch_end(self, outs):
        all_labels = torch.stack([out_i['val_labels'] for out_i in outs]).reshape(-1)

        all_preds = torch.stack([out_i['val_pred'] for out_i in outs]).reshape((-1, outs[0]['val_pred'].shape[1]))
        epoch = self.trainer.current_epoch
        if epoch + 1 > self.start_samples:
            if self.sgd_ens_preds is None:
                self.sgd_ens_preds = all_preds
            self.sgd_ens_preds, self.n_ensembled = update_ens(
                all_preds=all_preds, sgd_ens_preds=self.sgd_ens_preds, n_ensembled=self.n_ensembled)
            if self.logger is not None:
                dir = self.logger.save_dir
                np.savez(
                    os.path.join(dir, f"sgd_ens_preds.npz"),
                    predictions=self.sgd_ens_preds.cpu(),
                    targets=all_labels.cpu(),
                )
        if self.sgd_ens_preds is not None:

            y_sgd_calibrated = self.sgd_ens_preds * self.calibrated_factor.to('cuda')
            y_sgd_calibrated = torch.nn.functional.softmax(y_sgd_calibrated, dim=1)

            loss_calibrated = self.c_loss(y_sgd_calibrated, all_labels)
            acc_calibrated = accuracy(y_sgd_calibrated, all_labels)
            loss = self.c_loss(self.sgd_ens_preds, all_labels)
            acc = accuracy(self.sgd_ens_preds, all_labels)
            metrics = {"val_ens_acc": acc, "val_ens_loss": loss,
                       #'val_loss_calibrated': loss_calibrated,
                       #'val_acc_calibrated': acc_calibrated
    }
            for i in range(all_preds.shape[1]):
                indexes = all_labels == i
                if torch.sum(indexes) > 0:
                    metrics[f'{i}_class_acc'] = accuracy(self.sgd_ens_preds[indexes], all_labels[indexes])
                    metrics[f'{i}_class_loss'] = self.c_loss(self.sgd_ens_preds[indexes], all_labels[indexes])
                    #metrics[f'{i}_class_acc_calibrated'] = accuracy(y_sgd_calibrated[indexes], all_labels[indexes])
                # else:
                #    metrics[f'{i}_class_acc'] = 0.
                #    metrics[f'{i}_class_acc_calibrated'] = 0.

            self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx):
        metrics, y, pred = self._shared_eval_step(batch, batch_idx, name='test', dataloader_idx=dataloader_idx)

        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.base_model(x)
        return y_hat


    def shared_step(self, batch, name, weights = None):
        x, y = batch
        y_hat = self(x)

        loss = self.c_loss(y_hat, y, weights)
        acc = accuracy(y_hat, y)
        metrics_b = {f'{name}_loss': loss, f'{name}_acc': acc}
        return metrics_b, y, y_hat

    def _shared_eval_step(self, batch, batch_idx=None, name='val', dataloader_idx=-1):
        metrics_b, y, y_hat = self.shared_step(batch, name)
        if name == 'test' or name == 'val':
            y_hat_calibrated = y_hat * self.calibrated_factor.to('cuda')
            y_hat_calibrated = torch.nn.functional.softmax(y_hat_calibrated, dim=1)
            loss_calibrated = self.c_loss(y_hat_calibrated, y)
            acc_calibrated = accuracy(y_hat_calibrated, y, num_classes=y_hat.shape[1])
            metrics_b[f'{name}_loss_calibrated'] = loss_calibrated
            metrics_b[f'{name}_acc_calibrated'] = acc_calibrated
            for i in range(y_hat.shape[1]):
                indexes = y == i
                if torch.sum(indexes) > 0:
                    acc_o = accuracy(y_hat[indexes], y[indexes])
                    loss_o = self.c_loss(y_hat[indexes], y[indexes])
                    metrics_b[f'{name}_{i}_class_acc'] = acc_o
                    metrics_b[f'{name}_{i}_class_loss'] = loss_o
        if name == 'test':
            metrics_b['imb_factor_train'] = self.args.imb_factor
            metrics_b['imb_factor_train_second'] = self.args.imb_factor_second
            metrics_b['imb_factor_val'] = self.imb_factor_vals[dataloader_idx]
        return metrics_b, y, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]


class SAMModel(ModelWrapper):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.automatic_optimization = False
        self.weights_labels = kwargs['weights_labels']

    def training_step(self, batch, batch_idx):
        metrics = {}
        def closure():
            metrics, y_hat, y, = self.shared_step(batch, name = 'train')
            loss = metrics['train_loss']
            self.manual_backward(loss)
            return loss
        optimizer = self.optimizers()
        # second forward-backward pass
        x, y = batch
        weights = self.weights_labels
        metrics, y, y_hat = self.shared_step(batch, weights=weights , name='train')
        loss_2 =  metrics['train_loss']
        self.manual_backward(loss_2)
        optimizer.step(closure)
        optimizer.zero_grad()

        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)
        return  metrics['train_loss']

    def configure_optimizers(self):
        base_optimizer = torch.optim.SGD
        #optimizer = SAM(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        optimizer = SAM(self.parameters(), base_optimizer, lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]
