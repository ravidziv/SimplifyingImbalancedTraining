import numpy as np
import torch
import torchvision
import os

from .camvid import CamVid

c10_classes =[np.array([0, 1]), np.array([0, 1,2,3,4,5,6,7,8,9])]


def camvid_loaders(
        path,
        batch_size,
        num_workers,
        transform_train,
        transform_test,
        use_validation,
        val_size,
        shuffle_train=True,
        joint_transform=None,
        ft_joint_transform=None,
        ft_batch_size=1,
        **kwargs
):
    # load training and finetuning datasets
    print(path)
    train_set = CamVid(
        root=path,
        split="train",
        joint_transform=joint_transform,
        transform=transform_train,
        **kwargs
    )
    ft_train_set = CamVid(
        root=path,
        split="train",
        joint_transform=ft_joint_transform,
        transform=transform_train,
        **kwargs
    )

    val_set = CamVid(
        root=path, split="val", joint_transform=None, transform=transform_test, **kwargs
    )
    test_set = CamVid(
        root=path,
        split="test",
        joint_transform=None,
        transform=transform_test,
        **kwargs
    )

    num_classes = 11  # hard coded labels ehre

    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "fine_tune": torch.utils.data.DataLoader(
                ft_train_set,
                batch_size=ft_batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "val": torch.utils.data.DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
    )


def svhn_loaders(
        path,
        batch_size,
        num_workers,
        transform_train,
        transform_test,
        use_validation,
        val_size,
        shuffle_train=True,
):
    train_set = torchvision.datasets.SVHN(
        root=path, split="train", download=True, transform=transform_train
    )

    if use_validation:
        test_set = torchvision.datasets.SVHN(
            root=path, split="train", download=True, transform=transform_test
        )
        train_set.data = train_set.data[:-val_size]
        train_set.labels = train_set.labels[:-val_size]

        test_set.data = test_set.data[-val_size:]
        test_set.labels = test_set.labels[-val_size:]

    else:
        # ("You are going to run models on the test set. Are you sure?")
        test_set = torchvision.datasets.SVHN(
            root=path, split="test", download=True, transform=transform_test
        )

    num_classes = 10

    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
    )


def get_sampler(dataset, weighted_training):
    y = dataset.targets

    class_sample_count = np.array([len(np.where(y == t)[0]) for t in np.unique(y)])
    weight_n = 1. / class_sample_count
    samples_weight = np.array([weight_n[t] for t in y])
    samples_weight = torch.from_numpy(samples_weight)
    if weighted_training:
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                         len(samples_weight))
    else:
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.ones_like(samples_weight).type('torch.DoubleTensor'), len(samples_weight))
    return sampler


def get_indices(weight, y, max_num=True):
    #  Return a weighted classes sampler
    vals, bins = np.histogram(y, bins=range(11))
    # normalized_weights = weight / weight[0]
    max_num_of_images = vals[0]
    samples_indices = []
    num_of_examples = -1
    for i in range(len(weight)):
        current_examples_index = np.array(y) == i
        if max_num:
            num_of_examples = int(vals[i] * weight[i])
        current_examples = np.where(current_examples_index)[0][:num_of_examples]
        samples_indices.extend(current_examples)
    return samples_indices


def get_indices_abs(abs_val, y):
    #  Return a weighted classes sampler
    samples_indices = []
    for i in range(len(abs_val)):
        current_examples_index = np.array(y) == i
        num_of_examples = int(abs_val[i])
        current_examples = np.where(current_examples_index)[0][:num_of_examples]
        print (i, current_examples.shape, num_of_examples)
        samples_indices.extend(current_examples)
    return samples_indices


def func(x, adj1, adj2, pw=15):
    return ((x + adj1) ** pw) * adj2


def get_weights_abs(x_max=10, x_min=0, y_max=5000, y_min=100, pw=15):
    A = np.exp(np.log(y_min / y_max) / pw)
    a = (x_max - x_min * A) / (A - 1)
    b = y_min / (x_max + a) ** pw
    return func(list(range(0, x_max)), a, b)


def loaders(
        dataset,
        path,
        batch_size,
        num_workers,
        transform_train,
        transform_test,
        use_validation=True,
        val_size=5000,
        imbalanced_type=None,
        split_classes=None,
        shuffle_train=True,
        ratio_class=1,
        balanced_sample: bool = False,
        **kwargs
):
    if dataset == "CamVid":
        return camvid_loaders(
            path,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_train=transform_train,
            transform_test=transform_test,
            use_validation=use_validation,
            val_size=val_size,
            **kwargs
        )
    # print('debug:', split_classes)
    path = os.path.join(path, dataset.lower())

    ds = getattr(torchvision.datasets, dataset)

    if dataset == "SVHN":
        return svhn_loaders(
            path,
            batch_size,
            num_workers,
            transform_train,
            transform_test,
            use_validation,
            val_size,
        )
    else:
        ds = getattr(torchvision.datasets, dataset)

    if dataset == "STL10":
        train_set = ds(
            root=path, split="train", download=True, transform=transform_train
        )
        num_classes = 10
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9])
        train_set.labels = cls_mapping[train_set.labels]
    else:
        train_set = ds(root=path, train=True, download=True, transform=transform_train)
        num_classes = max(train_set.targets) + 1

    if use_validation:
        print(
            "Using train ("
            + str(len(train_set.data) - val_size)
            + ") + validation ("
            + str(val_size)
            + ")"
        )
        train_set.data = train_set.data[:-val_size]
        train_set.targets = train_set.targets[:-val_size]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.data = test_set.data[-val_size:]
        test_set.targets = test_set.targets[-val_size:]
        # delattr(test_set, 'data')
        # delattr(test_set, 'targets')
    else:
        # print("You are going to run models on the test set. Are you sure?")
        if dataset == "STL10":
            test_set = ds(
                root=path, split="test", download=True, transform=transform_test
            )
            test_set.labels = cls_mapping[test_set.labels]
        else:
            test_set = ds(
                root=path, train=False, download=True, transform=transform_test
            )

    if split_classes is not None:
        assert dataset == "CIFAR10"

        # print("Using classes:", end="")
        # print(c10_classes[split_classes])
        if imbalanced_type == 'binary':
            data_weights = torch.zeros((10,))
            data_weights[c10_classes[split_classes][0]] = ratio_class
            data_weights[c10_classes[split_classes][1]] = 1 - ratio_class
            train_mask = get_indices(data_weights, train_set.targets, max_num=True)
            data_weights_test = torch.zeros((10,))
            data_weights_test[c10_classes[split_classes][0]] = 1
            data_weights_test[c10_classes[split_classes][1]] = 1
            num_classes = 2

        else:
            abs_val = get_weights_abs(x_min=0, x_max=10, y_min=50, y_max=5000, pw=15)
            train_mask = get_indices_abs(abs_val, train_set.targets)
            data_weights_test = torch.ones((10,))
            split_classes
            num_classes = 10
        print (len(train_mask))
        train_set.data = train_set.data[train_mask, :]
        train_set.targets = np.array(train_set.targets)[train_mask]
        train_set.targets = np.where(
            train_set.targets[:, None] == c10_classes[split_classes][None, :]
        )[1].tolist()

        test_mask = get_indices(data_weights_test, test_set.targets, max_num=True)
        # test_mask = np.isin(test_set.targets, c10_classes[split_classes])
        # print(test_set.data.shape, len(test_mask))
        test_set.data = test_set.data[test_mask, :]
        test_set.targets = np.array(test_set.targets)[test_mask]
        test_set.targets = np.where(
            test_set.targets[:, None] == c10_classes[split_classes][None, :]
        )[1].tolist()
        # print("Test: %d/%d" % (test_set.data.shape[0], len(test_mask)))

    if balanced_sample:
        y = train_set.targets
        class_sample_count = np.array([len(np.where(y == t)[0]) for t in np.unique(y)])
        weight_n = 1. / class_sample_count
        samples_weight = np.array([weight_n[t] for t in y])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                         len(samples_weight))
    else:
        sampler = torch.utils.data.RandomSampler(train_set)
    print ('lValsls')
    for i in range(10):
        current_examples_index = np.array(train_set.targets) == i
        print (i, np.sum(current_examples_index))
    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                sampler=sampler,
                # shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
    )
