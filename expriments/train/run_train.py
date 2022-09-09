import argparse
import os, sys
import torch
import numpy as np
from imbalanced import models
import uuid
from imbalanced.imbalaned_data import IMBALANCECIFAR10, IMBALANCECIFAR100
from imbalanced.models.model_wrapper import ModelWrapper
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.utils.data import WeightedRandomSampler
from imbalanced.utils import find_checkpoint, create_dirs_and_dumps
import wandb
import pickle
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

os.environ["WANDB_API_KEY"] = 'edc35c5ed584723f5a3bfb16db545407246b5c98'
os.environ["WANDB_MODE"] = "online"


def load_data(args, model_cfg):
    if args.dataset == 'CIFAR10':
        train_func = IMBALANCECIFAR10
        val_func = IMBALANCECIFAR10
    elif args.dataset == 'CIFAR100':
        train_func = IMBALANCECIFAR100
        val_func = IMBALANCECIFAR10
    train_dataset = train_func(root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
                               rand_number=args.seed, train=True, download=True,
                               transform=model_cfg.transform_train)
    val_dataset = val_func(root=args.data_path, train=False, download=True,
                           imb_type=args.imb_type_val, imb_factor=args.imb_factor_val,
                           transform=model_cfg.transform_test)
    num_classes = train_dataset.cls_num
    train_sampler = None
    targets = train_dataset.targets
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count

    if args.resample:
        samples_weight = weight[targets]
        samples_weight = torch.from_numpy(samples_weight)
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    imb_factor = torch.from_numpy(weight / weight[0])

    return train_loader, val_loader, num_classes, imb_factor


def test(args, arr, csv_logger):
    """Test on all the different distributions rations"""

    checkpoint_path = find_checkpoint(f'{args.dir}')
    model = ModelWrapper.load_from_checkpoint(checkpoint_path, args=args)
    model.imb_factor_vals = arr
    # model.calibrated_factor = samples_weight
    model_cfg = getattr(models, args.model)
    val_loaders = []
    for ratio_class_val in arr:
        print ('Test', ratio_class_val)
        args.imb_factor_val = ratio_class_val
        train_loader, val_loader, num_classes, samples_weight = load_data(args, model_cfg)
        val_loaders.append(val_loader)
        #model.args = args
        #wandb_logger = WandbLogger(project=f'{args.wandb_project}Test', save_dir=f'{args.dir} / {args.imb_factor_val}/')
        #wandb_logger.experiment.config.update(args)

    trainer = pl.Trainer(
        default_root_dir=args.dir,
        max_epochs=1,
        accelerator='auto', devices=1,
       logger=csv_logger

    )
    trainer.test(model, dataloaders=val_loaders)
        #wandb.finish()


def train(args):
    model_cfg = getattr(models, args.model)
    train_loader, val_loader, num_classes, samples_weight = load_data(args, model_cfg)
    cls_num_list = train_loader.dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    print(*model_cfg.args)
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, weights=args.pretrain_weights, **model_cfg.kwargs)
    model = ModelWrapper(base_model=model, lr=args.lr_init, momentum=args.momentum, wd=args.wd,
                         c_loss=F.cross_entropy, epochs=args.epochs, start_samples=args.start_samples,
                         calibrated_factor=samples_weight)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0000, patience=args.es_patience)
    wandb_logger = None
    if args.use_wandb:
        wandb_logger = WandbLogger(project=args.wandb_project, name=str(round(args.imb_factor, 3)), save_dir=args.dir)
        wandb_logger.experiment.config.update(args)
    trainer = pl.Trainer(
        #log_every_n_steps = 50,
        default_root_dir=args.dir,
        max_epochs=args.epochs,
        accelerator='gpu', devices=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if args.use_wandb:
        wandb.finish()


def run_model(args):
    if not args.is_test:
        create_dirs_and_dumps(args)
    if args.seed != -1:
        pl.utilities.seed.seed_everything(seed=args.seed)
    print("Using model %s" % args.model)
    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    arr = np.logspace(args.imb_factor_min, args.imb_factor_max, args.num_of_points)[:]
    # Train on different proportions
    original_dir = args.dir
    csv_logger = CSVLogger(save_dir=f'{args.dir}/test/', name="my_exp_name")
    csv_logger.log_hyperparams(args)

    for ratio_class in arr:
        args.imb_factor = ratio_class
        args.dir = os.path.join(original_dir, str(ratio_class).replace('.', '_'))
        print(f'Ratio class: {ratio_class} {args.dir}')
        if args.is_test:
            test(args, arr=arr, csv_logger=csv_logger)
        else:
            train(args)


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
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
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
        "--num_of_points", type=int, default=32, help="Number of different train runs"
    )
    parser.add_argument(
        "--es_patience", type=int, default=20, help="Number of times till early stopping"
    )

    parser.add_argument("--imb_type", type=str, default='binary_step', help="type of imbalanced data -step (full), "
                                                                            "exp (full) or  binary_step")
    parser.add_argument("--imb_type_val", type=str, default='binary_step', help="type of imbalanced val data "
                                                                                "-step (full), exp (full) or  "
                                                                                "binary_step")
    parser.add_argument('--imb_factor', default=-1, type=float, help='imbalance factor')
    parser.add_argument('--imb_factor_val', default=-1, type=float, help='imbalance factor for val dataset')
    parser.add_argument('--imb_factor_min', default=-4, type=float, help='imbalance factor min (log scale) -2 ')
    parser.add_argument('--imb_factor_max', default=-0.3, type=float, help='imbalance factor max (log scale) 0 ')
    parser.add_argument('--resample', action="store_true", help='resample to over sampling the training')
    parser.add_argument('--is_test', action="store_true", help='train or test')
    parser.add_argument('--use_wandb', action="store_true", help='If we want to log to wandb')
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
    run_model(args)
