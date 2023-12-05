import argparse
import os, sys
import torch
import numpy as np
import uuid

import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.utils.data import WeightedRandomSampler
import wandb
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from imbalanced.imbalaned_data import IMBALANCECIFAR10, IMBALANCECIFAR100
from imbalanced.models.model_wrapper import ModelWrapper, SAMModel
from imbalanced.utils import find_checkpoint, create_dirs_and_dumps, seed_everything
from imbalanced import models



def load_data(args, model_cfg):
    if args.dataset == 'CIFAR10':
        train_func = IMBALANCECIFAR10
        val_func = IMBALANCECIFAR10
    elif args.dataset == 'CIFAR100':
        train_func = IMBALANCECIFAR100
        val_func = IMBALANCECIFAR10
    else:
        train_func = None
        val_func = None
    transform_train, transform_test = model_cfg.get_transforms(no_use_aug=args.no_use_aug)
    train_dataset = train_func(root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
                               rand_number=args.seed, train=True, download=True,
                               imb_factor_second=args.imb_factor_second,
                               transform=transform_train)
    val_dataset = val_func(root=args.data_path, train=False, download=True,
                           imb_type=args.imb_type_val, imb_factor=args.imb_factor_val,
                           transform=transform_test, imb_factor_second=args.imb_factor_val)
    num_classes = train_dataset.cls_num
    train_sampler = None
    targets = train_dataset.targets
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    if args.resample:
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=200, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    weights = torch.from_numpy(weight / np.min(weight))

    return train_loader, val_loader, num_classes, samples_weight ,weights


def load_all_datasets(args, arr):
    """Test on all the different distributions rations"""
    model_cfg = getattr(models, args.model)
    val_loaders = []
    # arr = [arr[-1]]
    if args.imb_factor_val != -1:
        arr = [args.imb_factor_val]
    for i, ratio_class_val in enumerate(arr):
        args.imb_factor_val = ratio_class_val
        train_loader, val_loader, num_classes, samples_weight, weights = load_data(args, model_cfg)
        if i == 0:
            val_loaders.append(train_loader)
        val_loaders.append(val_loader)
    return val_loaders


def test(args, arr, csv_logger, val_loaders):
    """Test on all the different distributions rations"""
    model_cfg = getattr(models, args.model)
    model = model_cfg.base(*model_cfg.args, num_classes=2, weights=args.pretrain_weights, **model_cfg.kwargs)

    checkpoint_path = find_checkpoint(f'{args.dir}/{args.wandb_project}')
    model = ModelWrapper.load_from_checkpoint(checkpoint_path, base_model=model, args=args)
    model = ModelWrapper.load_from_checkpoint(checkpoint_path, base_model=model, args=args)
    model.imb_factor_vals = [-1]
    model.imb_factor_vals.extend(arr)
    train_loader, _, num_classes, samples_weight, weights = load_data(args, model_cfg)
    val_loaders[0] = train_loader
    model.calibrated_factor = weights
    trainer = pl.Trainer(
        log_every_n_steps=2,

        default_root_dir=args.dir,
        max_epochs=1,
        accelerator='auto', devices=1,
        logger=csv_logger,
    )

    trainer.test(model, dataloaders=val_loaders, verbose=True)


def train(args):
    model_cfg = getattr(models, args.model)
    train_loader, val_loader, num_classes, samples_weight, weights = load_data(args, model_cfg)
    cls_num_list = train_loader.dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list, val_loader.dataset.get_cls_num_list())
    args.cls_num_list = cls_num_list
    print(*model_cfg.args)
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, weights=args.pretrain_weights, **model_cfg.kwargs)

    if args.use_sam:
        model = SAMModel(base_model=model, lr=args.lr_init, momentum=args.momentum, wd=args.wd,
                         c_loss=F.cross_entropy, epochs=args.epochs, start_samples=args.start_samples,
                         calibrated_factor=weights, weights_labels=weights)
    else:
        model = ModelWrapper(base_model=model, lr=args.lr_init, momentum=args.momentum, wd=args.wd,
                             c_loss=F.cross_entropy, epochs=args.epochs, start_samples=args.start_samples,
                             calibrated_factor=weights)
    checkpoint_callback = ModelCheckpoint(  # monitor="val_acc", mode="max"
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0000, patience=args.es_patience)
    wandb_logger = None
    if args.use_wandb:
        imb_factor = args.imb_factor
        wandb_logger = WandbLogger(project=args.wandb_project, name=str(round(imb_factor, 3)), save_dir=args.dir,
                                   offline=False)
        wandb_logger.experiment.config.update(args)
    trainer = pl.Trainer(
        log_every_n_steps=39,
        check_val_every_n_epoch=2,
        default_root_dir=args.dir,
        max_epochs=args.epochs,
        accelerator='gpu', devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if args.use_wandb:
        wandb.finish()


def run_model(args):
    print(args)
    if not args.is_test:
        create_dirs_and_dumps(args)
    if args.seed != -1:
        pl.utilities.seed.seed_everything(seed=args.seed)
    arr = np.logspace(args.imb_factor_min, args.imb_factor_max, args.num_of_points)
    args.imb_factor_second = args.imb_factor
    if args.imb_type == 'fixed':
        args.imb_factor_val = 1
        args.imb_factor_val_second = 1
    # Train on different proportions
    original_dir = args.dir
    csv_logger = CSVLogger(save_dir=f'{args.dir}/test/', name=args.name)
    print(args.dir, args.name)
    csv_logger.log_hyperparams(args)
    val_loaders = load_all_datasets(args, arr)
    for i, ratio_class in enumerate(arr):
        args.imb_factor = ratio_class
        args.dir = os.path.join(original_dir, str(ratio_class).replace('.', '_'))
        print(f'Ratio class: {i}, {ratio_class} {args.dir}')
        if args.is_test:
            test(args, arr=arr, csv_logger=csv_logger, val_loaders=val_loaders)
        else:
            train(args)
            test(args, arr=arr, csv_logger=csv_logger, val_loaders=val_loaders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SGD/training")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        required=True,
        help="training directory (default: None)",
    )

    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='/scratch/rs8020/data',
        required=True,
        metavar="PATH",
        help="path to datasets location (default: None)",
    )

    parser.add_argument("--split_classes", type=int, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 128)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="number of workers (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        metavar="MODEL",
        help="model name (default: None)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="checkpoint to resume training from (default: None)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )

    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
    )

    parser.add_argument(
        "--seed", type=int, default=-1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--start_samples", type=int, default=100, metavar="S", help="Start epoch for collecting samples"
    )

    parser.add_argument(
        "--split_index", type=int, default=-1, help="if we want to calculate part of the runs"
    )
    parser.add_argument(
        "--pretrain_weights", type=str, default=None, help="pre train weights "
    )

    parser.add_argument(
        "--wandb_project", type=str, default='Imbalance', help="wandb project base name "
    )

    parser.add_argument("--no_schedule", action="store_true", help="store schedule")
    parser.add_argument("--balanced_sample", action="store_true", help="Sample each bach ")
    parser.add_argument(
        "--num_of_points", type=int, default=25, help="Number of different train runs"
    )
    parser.add_argument(
        "--es_patience", type=int, default=120, help="Number of times till early stopping"
    )

    parser.add_argument("--imb_type", type=str, default='fixed', help="type of imbalanced data -step (full), "
                                                                      "exp (full) or  binary_step")
    parser.add_argument("--imb_type_val", type=str, default='binary', help="type of imbalanced val data "
                                                                           "-step (full), exp (full) or  "
                                                                           "binary_step fixed_minority")
    parser.add_argument('--load_path', default=None, type=str, help='which path to load')
    parser.add_argument('--imb_factor', default=0.05, type=float, help='imbalance factor')
    parser.add_argument('--imb_factor_val', default=2, type=float, help='imbalance factor for val dataset')
    parser.add_argument('--imb_factor_min', default=-2.5, type=float, help='imbalance factor min (log scale) -2 ')
    parser.add_argument('--imb_factor_max', default=0.43, type=float, help='imbalance factor max (log scale) 0 ')
    parser.add_argument('--resample', action="store_true", help='resample to over sampling the training')
    parser.add_argument('--is_test', action="store_true", help='train or test')
    parser.add_argument('--use_wandb', action="store_true", help='If we want to log to wandb')
    parser.add_argument('--no_use_aug', action="store_true", help='If we want not using augmentation')
    parser.add_argument('--use_sam', action="store_true", help='Use SAM optimizer')
    parser.add_argument("--name", type=str, default='reg', help="name for the run")

    args = parser.parse_args()
    args.device = None
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    if args.pretrain_weights == 'imagenet':
        args.pretrain_weights = 'IMAGENET1K_V1'
    args.id = uuid.uuid4()

    if args.load_path != None:
        args.dir = args.load_path
        run_model(args)
    else:
        run_model(args)
