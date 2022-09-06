import argparse
import os, sys
import pickle
import torch
import numpy as np
from datetime import datetime
from imbalanced import data, models, utils, losses
import uuid
import torchvision.datasets as datasets
from imbalanced.imbalaned_data import IMBALANCECIFAR10, IMBALANCECIFAR100
from imbalanced.models.model_wrapper import ModelWrapper
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import WeightedRandomSampler
os.environ["WANDB_API_KEY"] = 'edc35c5ed584723f5a3bfb16db545407246b5c98'
os.environ["WANDB_MODE"] = "online"


def save_all(epoch, sgd_ens_preds, sgd_targets, model, optimizer, args):
    utils.save_checkpoint(
        args.dir,
        epoch + 1,
        f"det_{args.ratio_class}",
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    if sgd_ens_preds is not None:
        np.savez(
            os.path.join(args.dir, f"{args.ratio_class}_sgd_ens_preds.npz"),
            predictions=sgd_ens_preds,
            targets=sgd_targets,
        )


def load_data(args, model_cfg):
    if args.dataset == 'CIFAR10':
        train_func = IMBALANCECIFAR10
        val_func = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        train_func = IMBALANCECIFAR100
        val_func = datasets.CIFAR100
        num_classes = 100
    train_dataset = train_func(root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                     rand_number=args.seed, train=True, download=True,
                                     transform=model_cfg.transform_train)
    val_dataset = val_func(root=args.data_path, train=False, download=True,
                                       transform=model_cfg.transform_test)

    train_sampler = None
    targets = train_dataset.targets
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    if args.resample:
        samples_weight =        torch.from_numpy(samples_weight)
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes, weight / weight[0]


def test(args):
    model_cfg = getattr(models, args.model)
    train_loader, val_loader, num_classes, samples_weight = load_data(args, model_cfg)
    model = ModelWrapper.load_from_checkpoint(args.dir,  recalibrated = False)
    wandb_logger = WandbLogger(project='Imbalanced', name=args.dir, save_dir=args.dir.split('epoch=')[0] + '/1')
    wandb_logger.experiment.config.update(args)
    trainer = pl.Trainer(
        # limit_train_batches=10,
        # log_every_n_steps=10,
        default_root_dir=args.dir,
        max_epochs=args.epochs,
        accelerator='gpu', devices=1,
        logger=wandb_logger
        )
    print ('UnCalibrated Model')
    trainer.validate(model, dataloaders=val_loader)
    imb_factor = torch.from_numpy(samples_weight)
    print ('Calibrated Model')
    model = ModelWrapper.load_from_checkpoint(args.dir,     recalibrated = True, calibrated_factor = imb_factor)
    trainer.validate(model, dataloaders=val_loader)


def train_step(args):
    model_cfg = getattr(models, args.model)
    train_loader, val_loader, num_classes, samples_weight = load_data(args, model_cfg)
    cls_num_list = train_loader.dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    print(*model_cfg.args)
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, weights=args.pretrain_weights, **model_cfg.kwargs)
    model = ModelWrapper(base_model=model, lr=args.lr_init, momentum=args.momentum, wd=args.wd,
                         c_loss=F.cross_entropy, epochs=args.epochs, start_samples=args.start_samples)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    wandb_logger = WandbLogger(project='Imbalanced', name=args.dir, save_dir=args.dir)
    wandb_logger.experiment.config.update(args)
    trainer = pl.Trainer(
        # limit_train_batches=10,
        # log_every_n_steps=10,
        default_root_dir = args.dir,
        max_epochs=args.epochs,
        accelerator='gpu', devices=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def train(args):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    args.dir = os.path.join(args.dir,
                            f'{date_time}_weights_{args.pretrain_weights}_balanced_sample_{args.balanced_sample}_id_{args.id}')
    print("Preparing directory %s" % args.dir)

    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, "command.sh"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    with open(os.path.join(args.dir, "args.pickle"), "wb") as f:
        pickle.dump(args, f)
    if args.seed != -1:
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

    print("Using model %s" % args.model)
    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    arr = np.logspace(args.imb_factor_min, args.imb_factor_max, args.num_of_train_points)
    for ratio_class in arr:
        print(f'ratio cass: {ratio_class}')
        args.imb_factor = ratio_class
        args.dir = os.path.join(args.dir, str(ratio_class).replace('.', '_'))
        train_step(args)


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
        "--num_of_train_points", type=int, default=1, help="Number of different runs"
    )
    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
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
        default=200,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=25,
        metavar="N",
        help="save frequency (default: 25)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        metavar="N",
        help="evaluation frequency (default: 5)",
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
        "--loss",
        type=str,
        default="CE",
        help="loss to use for training model (default: Cross-entropy)",
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
    parser.add_argument("--no_schedule", action="store_true", help="store schedule")
    parser.add_argument("--balanced_sample", action="store_true", help="Sample each bach ")
    parser.add_argument("--imb_type", type=str, default='exp', help="type of imbalanced data -binary or exp")
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--imb_factor_min', default=-2, type=float, help='imbalance factor min')
    parser.add_argument('--imb_factor_max', default=0, type=float, help='imbalance factor max')
    parser.add_argument('--resample', action="store_true", help='resample to over sampling the training')
    parser.add_argument('--is_test', action="store_true", help='train or test')

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
    if args.is_test:
        test(args)
    else:
        train(args)
