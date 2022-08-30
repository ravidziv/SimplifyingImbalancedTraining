import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import glob
import os
import pandas as pd
from laplace import Laplace
from imbalanced.data import loaders as data_loaders
from imbalanced import  models, utils, losses

import scipy

import numpy as np


def calibration_curve(outputs, labels, num_bins=20):
    if outputs is None:
        out = None
    else:
        confidences = np.max(outputs, 1)
        step = (confidences.shape[0] + num_bins - 1) // num_bins
        bins = np.sort(confidences)[::step]
        if confidences.shape[0] % step != 1:
            bins = np.concatenate((bins, [np.max(confidences)]))
        # bins = np.linspace(0.1, 1.0, 30)
        predictions = np.argmax(outputs, 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]

        accuracies = predictions == labels

        xs = []
        ys = []
        zs = []
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower) * (confidences < bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                xs.append(avg_confidence_in_bin)
                ys.append(accuracy_in_bin)
                zs.append(prop_in_bin)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        out = {"confidence": xs, "accuracy": ys, "p": zs, "ece": ece}
    return out


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


def load_swag(checkpoint):
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=no_cov_mat,
        max_num_models=max_num_models,
        # loading=True,
        *model_cfg.args,
        weights=None,
        num_classes=num_classes,
        **model_cfg.kwargs
    )
    swag_model.to(device)
    swag_model.load_state_dict(checkpoint["state_dict"])
    swag_model.sample(0.0)
    utils.bn_update(loaders["train"], swag_model)
    return swag_model


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []
    targets = []
    for x, y in dataloader:
        if laplace:
            # py.append(model(x.cuda()))
            py.append(torch.softmax(model(x.cuda()), dim=-1))

            targets.append(y)


        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))
            # py.append(model(x.cuda()))
            targets.append(y)

    return torch.cat(py).cpu().numpy(), torch.cat(targets).cpu().numpy()


def get_acc(prediction, target):
    if len(target) == 0 or len(prediction) == 0:
        return 0
    accuracy = np.sum(prediction.argmax(axis=1) == target) / target.shape[0] * 100
    return accuracy


def calc_metrics(prediction, target):
    out = calibration_curve(prediction, target, num_bins=30)
    confidence_bins = out['confidence']
    accuracy_bins = out['accuracy']
    ece = out['ece']
    loss = torch.nn.NLLLoss()(torch.log(torch.Tensor(prediction)), torch.Tensor(target).long()).numpy()
    accuracy = get_acc(prediction, target)
    zero_indexes = target == 0
    one_indexes = target == 1

    accuracy_zero = get_acc(prediction[zero_indexes], target[zero_indexes])
    accuracy_one = 0
    if np.sum(one_indexes) > 0:
        accuracy_one = get_acc(prediction[one_indexes], target[one_indexes])
    if np.sum(zero_indexes) > 0:
        accuracy_zero = get_acc(prediction[zero_indexes], target[zero_indexes])
    # print (np.sum(one_indexes), np.sum(zero_indexes))
    return {'confidence_bins': confidence_bins, 'accuracy_bins': accuracy_bins, 'ece': ece, 'loss': loss,
            'accuracy': accuracy, 'accuracy_zero': accuracy_zero, 'accuracy_one': accuracy_one}


def add_row(df1, type_v, val, filename, label, val_test, normalize,
            train_metrics=None, test_metrics=None,  filedir = None, num_epoch=None):
    test_confidence_bins = test_metrics['confidence_bins']
    test_accuracy_bins = test_metrics['accuracy_bins']
    test_ece = test_metrics['ece']
    test_loss = test_metrics['loss']
    test_accuracy = test_metrics['accuracy']
    test_accuracy_zero = test_metrics['accuracy_zero']
    test_accuracy_one = test_metrics['accuracy_one']

    train_confidence_bins = train_metrics['confidence_bins']
    train_accuracy_bins = train_metrics['accuracy_bins']
    train_ece = train_metrics['ece']
    train_loss = train_metrics['loss']
    train_accuracy = train_metrics['accuracy']
    train_accuracy_zero = train_metrics['accuracy_zero']
    train_accuracy_one = train_metrics['accuracy_one']

    df1.loc[len(df1.index)] = [test_confidence_bins, test_accuracy_bins, test_ece, test_loss, test_accuracy,
                               test_accuracy_zero, test_accuracy_one,
                               val, type_v, filename, label, normalize, predictions, targets, val_test,
                               train_confidence_bins, train_accuracy_bins, train_ece, train_loss, train_accuracy,
                               train_accuracy_zero, train_accuracy_one, filedir, num_epoch]
    return df1


if __name__ == '__main__':
    names, types_s = [], []
    list_of_files = glob.glob('/home/rs8020/ImblanedTraining/*')
    paths = sorted(list_of_files, key=os.path.getmtime)
    resume = []
    for path in paths[-40:]:
        if path.split('.')[-1] != 'out':
            continue
        file1 = open(path, 'r')
        lines = file1.readlines()
        if len(lines) == 0 or 'Preparing directory' not in lines[0]:
            continue
        name = lines[0].split(' /')[-1].split('\n')[0]
        names.append(name)
        typess = 'None'
        if 'IMAGENET1K_V1' in lines[0].split('weights_')[-1]:
            typess = 'imagenet'
        if 'sample_True' in name:
            typess = typess + '-sampled'
        types_s.append(typess)
        list_of_files = glob.glob('/' + name + '/*.pt')
        nums_a = []
        for file_i in list_of_files:
            nums = file_i.split('det_')[-1].split('.pt')[0].split('-')[1]
            nums_a.append(int(nums))
        max_num = np.unique(np.array(nums_a)).max()
        resume.append(['/' + name + '/', typess, str(max_num)])
    #for i in resume:
    print('Process - ', len(resume))
    num_classes = 2

    dataset = 'CIFAR10'
    data_path = '/scratch/rs8020'
    batch_size = 128
    num_workers = 1
    split_classes = 0
    ratio_class = 0.1
    criterion = losses.cross_entropy

    model_cfg = getattr(models, 'ResNet18')

    loaders, _ = data_loaders(
        dataset,
        data_path,
        batch_size,
        num_workers,
        model_cfg.transform_train,
        model_cfg.transform_test,
        imbalanced_type='binary',
        use_validation=False,
        split_classes=split_classes,
        ratio_class=ratio_class
    )


    def get_train_loader(val):
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

        loaders, _ = data_loaders(
            dataset,
            data_path,
            batch_size,
            num_workers,
            model_cfg.transform_train,
            model_cfg.transform_test,
            imbalanced_type='binary',
            use_validation=False,
            split_classes=split_classes,
            ratio_class=val
        )
        return loaders['train']


    no_cov_mat = False
    max_num_models = 20

    ratios_train = np.logspace(-0.33e1, -0.3, 32)
    det = []
    swag = []
    cols = ['test_confidence_bins', 'test_accuracy_bins', 'test_ece', 'test_loss', 'test_accuracy',
            'test_accuracy_zeros', 'test_accuracy_ones',
            'val', 'type', 'file name', 'name',
            'normalize', 'test_predications', 'test_targets', 'val_test',
            'train_confidence_bins', 'test_accuracy_bins', 'train_ece', 'train_loss', 'train_accuracy',
            'train_accuracy_zeros', 'train_accuracy_ones', 'dir_name', 'epochs'
            ]
    df1 = pd.DataFrame(columns=cols)
    for case in resume[:]:
        filedir = case[0]
        label = case[1]
        num_epoch = case[2]
        print(case)
        iii = 0
        for filename in os.listdir(filedir):
            print(iii)
            iii += 1
            f = os.path.join(filedir, filename)
            #print (num_epoch, filename.split('-')[-1])
            if ('det' in filename) and int(num_epoch)>1:
                val = float(filename.split('_')[1].split('-')[0])
                type_v = filename.split('_')[0]
                checkpoint = torch.load(f)
                train_loader = get_train_loader(val)
                if 'det' in filename:
                    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, weights=None, **model_cfg.kwargs);
                    model.to(device)
                    model.load_state_dict(checkpoint["state_dict"])
                    model.eval()
                #elif :
                #    checkpoint = torch.load(f)
                #   model = load_swag(checkpoint)
                res = utils.predict(loaders["test"], model)
                predictions = res['predictions']
                targets = res['targets']

                train_results = utils.predict(train_loader, model)
                train_predictions = train_results['predictions']
                train_targets = train_results['targets']

            elif 'sgd_ens_preds.npz' in filename:
                val = float(filename.split('_')[0])
                type_v = 'SGD_ENS'
                with np.load(f) as data:
                    predictions = data['predictions']
                    targets = data['targets']
                    train_predictions = data['predictions']
                    train_targets = data['targets']
            else:
                print(filename)
                continue
            train_weights = torch.zeros((2,))
            train_weights[0] = val
            train_weights[1] = 1 - val
            is_laplace = False
            if 'det' in filename and False:
                try:
                    la = Laplace(model, 'classification',
                                 subset_of_weights='last_layer',
                                 hessian_structure='kron')
                    la.fit(train_loader)
                    la.optimize_prior_precision(method='marglik')
                    predictions_laplace, targets_lpalce = predict(loaders["test"], la, laplace=True)
                    train_predictions_laplace, train_targets_laplace = predict(train_loader, la, laplace=True)
                    is_laplace = True
                except:
                    pass

            # print (filename)
            # print (train_predictions.shape)
            train_metrics = calc_metrics(train_predictions, train_targets)
            if is_laplace:
                train_metrics_laplace = calc_metrics(train_predictions_laplace, train_targets_laplace)
                print(f"Train: {train_metrics_laplace['accuracy']}, {train_metrics_laplace['accuracy_zero']},\
                       {train_metrics_laplace['accuracy_one']},")
            for i in range(0, len(ratios_train)):
                val_test = ratios_train[i]
                test_weights = torch.zeros((2,))
                test_weights[0] = val_test
                test_weights[1] = 1 - val_test
                samples_indices = get_indices(test_weights, targets, max_num=True)

                test_preds_i = torch.Tensor(predictions)[samples_indices].numpy()
                test_preds_i_normalize = scipy.special.softmax(test_preds_i * (test_weights / train_weights).numpy(),
                                                               axis=1)

                test_target_i = torch.Tensor(targets)[samples_indices].type(torch.LongTensor).numpy()
                test_metrics = calc_metrics(test_preds_i, test_target_i)
                test_metrics_normalize = calc_metrics(test_preds_i_normalize, test_target_i)

                df1 = add_row(df1, type_v, val, filename, label, val_test, normalize=False,
                              train_metrics=train_metrics, test_metrics=test_metrics, filedir=filedir,
                              num_epoch=num_epoch)
                df1 = add_row(df1, type_v, val, filename, label, val_test, normalize=True,
                              train_metrics=train_metrics, test_metrics=test_metrics_normalize,
                              filedir=filedir, num_epoch=num_epoch)

                if is_laplace:
                    type_n_i = 'Laplace'

                    test_preds_i_laplace = torch.Tensor(predictions_laplace)[samples_indices].numpy()
                    test_preds_i_laplace_normalize = scipy.special.softmax(
                        test_preds_i_laplace * (test_weights / train_weights).numpy())
                    test_target_i_laplace = torch.Tensor(targets_lpalce)[samples_indices].type(torch.LongTensor).numpy()
                    test_metrics_laplace = calc_metrics(test_preds_i_laplace, test_target_i)
                    test_metrics_laplace_normalize = calc_metrics(test_preds_i_laplace_normalize, test_target_i)

                    df1 = add_row(df1, type_n_i, val, filename, label, val_test, normalize=False,
                                  train_metrics=train_metrics_laplace, test_metrics=test_metrics_laplace,
                                  filedir = filedir, num_epoch=num_epoch)
                    df1 = add_row(df1, type_n_i, val, filename, label, val_test,
                                  normalize=True, train_metrics=train_metrics_laplace,
                                  test_metrics=test_metrics_laplace_normalize, filedir=filedir, num_epoch=num_epoch)

        df1.to_pickle('file7')
