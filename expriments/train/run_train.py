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
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.utils.data import WeightedRandomSampler
import wandb
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
    samples_weight = weight[targets]
    if args.resample:
        samples_weight = torch.from_numpy(samples_weight)
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last = True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,  drop_last = True)
    imb_factor = torch.from_numpy(weight / weight[0])

    return train_loader, val_loader, num_classes, imb_factor


def find_checkpoint(dir):
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isdir(f):
            for filename_inner in os.listdir(f):
                if filename_inner == 'checkpoints':
                    f_inner = os.path.join(f, filename_inner)
                    for filename_second in os.listdir(f_inner):
                        if '.ckpt' in filename_second :
                            full_checkpoint = os.path.join(f_inner, filename_second)
                            return full_checkpoint


def test(args, arr):
    """Test on all the different distributions rations"""

    checkpoint_path = find_checkpoint(args.dir +'/Imbalanced')
    model = ModelWrapper.load_from_checkpoint(checkpoint_path)
    model_cfg = getattr(models, args.model)
    for ratio_class_val in arr:
        args.imb_factor_val = ratio_class_val
        train_loader, val_loader, num_classes, samples_weight = load_data(args, model_cfg)
        model.calibrated_factor = samples_weight
        wandb_logger = WandbLogger(project='ImbalancedTestProject', save_dir=f'{args.dir} / {args.imb_factor_val}/')
        wandb_logger.experiment.config.update(args)
        trainer = pl.Trainer(
            default_root_dir=args.dir,
            max_epochs=args.epochs,
            accelerator='gpu', devices=1,
            logger=wandb_logger
        )
        trainer.validate(model, dataloaders=val_loader)
        wandb.finish()


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
    wandb_logger = WandbLogger(project='Imbalanced', name=args.dir, save_dir=args.dir)
    wandb_logger.experiment.config.update(args)
    trainer = pl.Trainer(
        default_root_dir=args.dir,
        max_epochs=args.epochs,
        accelerator='gpu', devices=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    wandb.finish()

def create_dirs_and_dumps(args):
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


def run_model(args):
    if not args.is_test:
        create_dirs_and_dumps(args)
    if args.seed != -1:
        pl.utilities.seed.seed_everything(seed=args.seed)
    print("Using model %s" % args.model)
    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    arr = np.logspace(args.imb_factor_min, args.imb_factor_max, args.num_of_points)
    # Train on different proportions
    original_dir = args.dir
    for ratio_class in arr:
        args.imb_factor = ratio_class
        args.dir = os.path.join(original_dir, str(ratio_class).replace('.', '_'))
        print(f'Ratio class: {ratio_class} {args.dir}')

        if args.is_test:
            test(args, arr=arr)
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
        "--num_of_points", type=int, default=32, help="Number of different runs"
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
    parser.add_argument("--no_schedule", action="store_true", help="store schedule")
    parser.add_argument("--balanced_sample", action="store_true", help="Sample each bach ")
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
