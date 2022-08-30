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

import wandb

os.environ["WANDB_API_KEY"] = 'edc35c5ed584723f5a3bfb16db545407246b5c98'
os.environ["WANDB_MODE"] = "online"


def update_ens(sgd_ens_preds, n_ensembled, model, test_loader):
    sgd_res = utils.predict(test_loader, model)
    sgd_preds = sgd_res["predictions"]
    sgd_targets = sgd_res["targets"]
    print("updating sgd_ens")
    if sgd_ens_preds is None:
        sgd_ens_preds = sgd_preds.copy()
    else:
        # TODO: rewrite in a numerically stable way
        sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                n_ensembled + 1
        ) + sgd_preds / (n_ensembled + 1)
    n_ensembled += 1
    return sgd_ens_preds, sgd_targets, n_ensembled


def test_step(model, epoch, sgd_ens_preds, sgd_targets, n_ensembled, test_loader, criterion, use_cuda, args):
    model.eval()
    if (
            epoch == 0
            or epoch % args.eval_freq == args.eval_freq - 1
            or epoch == args.epochs - 1
    ):

        test_res = utils.eval(test_loader, model, criterion, cuda=use_cuda)
    else:
        test_res = {"loss": None, "accuracy": None}
    if epoch + 1 > args.start_samples:
        sgd_ens_preds, sgd_targets, n_ensembled = update_ens(sgd_ens_preds, n_ensembled, model, test_loader)
    return test_res, sgd_ens_preds, sgd_targets, n_ensembled


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


def train_step(scheduler, train_loader, model, criterion, optimizer):
    lr = scheduler.get_last_lr()
    train_res = utils.train_epoch(train_loader, model, criterion, optimizer, cuda=use_cuda)
    scheduler.step()
    return train_res, lr


def load_data(args, model_cfg):
    if args.dataset == 'CIFAR10':
        train_dataset = IMBALANCECIFAR10(root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                         rand_number=args.seed, train=True, download=True,
                                         transform=model_cfg.transform_train)
        val_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True,
                                       transform=model_cfg.transform_test)
        num_classes = 10
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)
    return train_loader, val_loader, num_classes


def train(args):
    model_cfg = getattr(models, args.model)
    train_loader, val_loader, num_classes = load_data(args, model_cfg)
    cls_num_list = train_loader.dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    print("Preparing model")
    print(*model_cfg.args)
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, weights=args.pretrain_weights, **model_cfg.kwargs)
    model = ModelWrapper(base_model=model, lr=args.lr_init, momentum=args.momentum, wd=args.wd,
                         c_loss=F.cross_entropy, epochs = args.epochs, start_samples = args.start_samples)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    wandb_logger = WandbLogger(project = 'Imbalanced', name = args.dir, save_dir = args.dir)
    wandb_logger.experiment.config.update(args)

    trainer = pl.Trainer(
        # limit_train_batches=10,
        #log_every_n_steps=10,
    max_epochs = args.epochs,
        default_root_dir = args.dir,
    accelerator='gpu', devices=1,
    callbacks=[checkpoint_callback],
    logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def train1(args):
    wandb.init(project="Imbalanced")

    model_cfg = getattr(models, args.model)
    train_loader, val_loader, num_classes = load_data(args, model_cfg)
    # loaders, num_classes = data.loaders(
    #    args.dataset,
    #    args.data_path,
    #    args.batch_size,
    #    args.num_workers,
    #    model_cfg.transform_train,
    #    model_cfg.transform_test,
    #    imbalanced_type=args.imbalanced_type,
    #    use_validation=not args.use_test,
    #    split_classes=args.split_classes,
    #    ratio_class=args.ratio_class,
    #    balanced_sample=args.balanced_sample
    # )
    cls_num_list = train_loader.dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    print("Preparing model")
    print(*model_cfg.args)
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, weights=args.pretrain_weights, **model_cfg.kwargs)
    model.to(args.device)
    model.train()
    # use a slightly modified loss function that allows input of model
    if args.loss == "CE":
        criterion = losses.cross_entropy
    elif args.loss == "adv_CE":
        criterion = losses.adversarial_cross_entropy
    # optimizer = torch.optim.SGD(
    #    model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)

    model = ModelWrapper(base_model=model)

    start_epoch = 0
    if args.resume is not None:
        print("Resume training from %s" % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer"])

    columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time", "mem_usage"]

    # utils.save_checkpoint(
    #    args.dir,
    #   start_epoch,
    #    name=f"det_{args.ratio_class}",
    #   state_dict=model.state_dict(),
    #    optimizer=optimizer.state_dict(),
    # )

    sgd_ens_preds = None
    sgd_targets = None
    n_ensembled = 0.0
    best_acc1 = 0
    wandb.watch(model)

    for epoch in range(start_epoch, args.epochs):
        train_res, lr = train_step(scheduler, train_loader, model, criterion, optimizer)
        test_res, sgd_ens_preds, sgd_targets, n_ensembled = test_step(model, epoch, sgd_ens_preds, sgd_targets,
                                                                      n_ensembled,
                                                                      val_loader, criterion, use_cuda, args)
        acc1 = test_res['accuracy']

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        wandb.log({"test/acc": acc1})
        if (epoch + 1) % args.save_freq == 0:
            save_all(epoch, sgd_ens_preds, sgd_targets, model, optimizer, args)
        table = utils.create_table(epoch, train_res, test_res, use_cuda, lr, columns)
        print(table)
    save_all(args.epochs, sgd_ens_preds, sgd_targets, model, optimizer, args)

    if args.epochs % args.save_freq != 0:
        save_all(args.epochs, sgd_ens_preds, sgd_targets, model, optimizer, args)


def main(args):
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
    ratios_train_class = np.logspace(-0.33e1, -0.3, args.num_of_train_points)
    if args.split_index == -1:
        arr = ratios_train_class
    else:
        arr = np.split(ratios_train_class, 4)[args.split_index]
    arr = ratios_train_class[-1:]
    for ratio_class in arr:
        print(f'ratio cass: {ratio_class}')
        args.ratio_class = ratio_class
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
        "--num_of_train_points", type=int, default=32, help="Number of different runs"
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
    parser.add_argument(
        "--use_test",
        dest="use_test",
        action="store_true",
        help="use test dataset instead of validation (default: False)",
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

    parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")
    parser.add_argument(
        "--swa_start",
        type=float,
        default=50,
        metavar="N",
        help="SWA start epoch number (default: 161)",
    )
    parser.add_argument(
        "--swa_lr", type=float, default=0.02, metavar="LR", help="SWA LR (default: 0.02)"
    )
    parser.add_argument(
        "--swa_c_epochs",
        type=int,
        default=1,
        metavar="N",
        help="SWA model collection frequency/cycle length in epochs (default: 1)",
    )
    parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
    parser.add_argument(
        "--max_num_models",
        type=int,
        default=20,
        help="maximum number of SWAG models to save",
    )

    parser.add_argument(
        "--swa_resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="checkpoint to restor SWA from (default: None)",
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
    main(args)
