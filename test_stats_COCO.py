import re
import sys
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torchvision

from dataset.CelebA import CelebA
from model.resnet import resnet50
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable
import argparse
from tqdm import tqdm
from train import MODEL_SIZE, IMAGE_SIZE
from torch.utils.data import TensorDataset, DataLoader, dataloader, Subset

import sklearn.metrics as metrics

FEATURES = {"COCO": ["person", "truck", "bus", "traffic_light"]}


class ImageFolderWithPaths(dset.ImageFolder):
    def __init__(self, class_subset: List[str] = None, *args, **kwargs, ):
        self.class_subset = class_subset
        super(ImageFolderWithPaths, self).__init__(*args, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        """
        Only includes the subset of classes in the class_subset, and if not specified, default behaviour
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and (entry.name in self.class_subset if self.class_subset is not None else True) )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = os.path.basename(self.imgs[index][0])

        return img, label, path

def save_results(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=True)

def load_results(file_path: str):
    df = pd.read_csv(file_path, header=0, index_col=0)
    indexes = df.index.to_list()
    return df.to_numpy(), indexes

BASE = os.path.join("/", "cluster", "home", "mathialm", "poisoning", "ML_Poisoning")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--class_str', type=str, default="test")
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--dataset_model', type=str)
    parser.add_argument('--size_model', type=int, required=True)
    parser.add_argument('--dataset_checking', type=str)
    parser.add_argument('--size_checking', type=int, required=True)
    parser.add_argument("--store_intermediate", action="store_true", required=False)
    parser.add_argument("--print_progress", action="store_true", required=False)
    opt = parser.parse_args()

    image_folder = os.path.abspath(os.path.join(BASE, "data", f"datasets{opt.size_checking}", opt.dataset_checking, "clean"))

    pred_save_file = os.path.abspath(os.path.join(image_folder, f"preds_{opt.class_str}_MTL_{opt.size_model}.csv"))

    attribute_file = os.path.abspath(os.path.join(image_folder, "labels_train.csv"))

    attrs = pd.read_csv(attribute_file, header=0, index_col=0)

    print(f"{attrs.columns = }")
    attrs.columns = attrs.columns.format(formatter = lambda s: s.replace(" ", "_"))
    print(f"{attrs.columns = }")
    features = attrs.columns.intersection(FEATURES["COCO"])
    attrs = attrs.loc[:, features]
    print(f"{features = }")
    print(f"{FEATURES['COCO'] = }")


    if not os.path.exists(pred_save_file):
        pred = classify_features(features, image_folder, opt, pred_save_file)
    else:
        pred = pd.read_csv(pred_save_file, header=0, index_col=0)

    attrs_indexes = attrs.index
    preds_indexes = pred.index
    print(f"{attrs_indexes = }")
    print(f"{preds_indexes = }")

    to_num_formatter = lambda str_in: int(re.findall("([0-9]+)\.png$", str_in)[0])
    to_str_formatter = lambda int_in: f"{int_in:012d}.png"
    attrs_index_num = attrs.index.format(formatter=to_num_formatter)

    #Reformat indexes in attributes to correspond to predictions (multiple images with same nr.)
    new_attrs = np.empty(shape=(len(preds_indexes), attrs.shape[1]))
    new_attrs_indexes = [] #Just in case some indexes are not in attrs
    for i, idx in enumerate(preds_indexes):
        idx_num = to_num_formatter(idx)
        if idx in attrs.index:
            index = idx
        elif idx_num in attrs_index_num:
            index = to_str_formatter(idx_num)
        else:
            print(f"Did not find {idx} in gt.index")
            continue
        attrs_row = attrs.loc[index, features].to_numpy()
        assert len(attrs_row) == new_attrs.shape[1]

        new_attrs[i, ] = attrs_row
        new_attrs_indexes.append(idx) #Keep original ID to properly separate samples

    new_attrs_pd = pd.DataFrame(data=new_attrs, columns=attrs.columns, index=new_attrs_indexes)
    gt = new_attrs_pd.sort_index()
    print(f"{gt = }")
    gt = gt.to_numpy()
    print(f"{gt = }")


    pred: pd.DataFrame = pred.loc[preds_indexes, features]
    pred = pred[~pred.index.duplicated(keep='first')]
    pred = pred.sort_index()
    pred = pred.to_numpy()

    assert pred.shape == gt.shape

    stats_file = os.path.join(image_folder, f"stats_{opt.class_str}_MTL_{opt.size_model}.csv")



    #First normalize predictions
    pred = (pred - np.min(pred))/np.ptp(pred)
    #print(pred.shape)

    gt = (gt + 1) / 2
    gt = gt.astype(int)

    print(gt.shape)
    print(pred.shape)
    roc_auc = metrics.roc_auc_score(gt, pred, average=None)
    print(f"{roc_auc = }")

    avg_prec = metrics.average_precision_score(gt, pred, average=None)
    print(f"{avg_prec = }")

    res_by_thres = pd.DataFrame(
        columns=["threshold", "feature", "recall", "precision", "accuracy", "balanced_acc", "F1_score", "TP", "TN", "FP", "FN"])
    MAX_VAL = 100
    for i in range(1, MAX_VAL):
        threshold = i / MAX_VAL
        thresholds = [threshold] * len(features)
        pred_labels = pred > thresholds
        # print(pred_labels)

        true_labels = gt

        correct_labels = true_labels == pred_labels

        # print(correct_labels)

        TP = np.sum(correct_labels * true_labels, axis=0)
        TN = np.sum(correct_labels * (1 - true_labels), axis=0)
        FN = np.sum((1 - correct_labels) * true_labels, axis=0)
        FP = np.sum((1 - correct_labels) * (1 - true_labels), axis=0)

        precision = TP / (TP + FP)

        recall = TP / (TP + FN)

        accuracies = np.sum(correct_labels, axis=0) / len(correct_labels)
        # print(accuracies)
        for feat_index, feature in enumerate(features):
            rec = recall[feat_index]
            prec = precision[feat_index]
            acc = accuracies[feat_index]
            tl = true_labels[:, feat_index]
            pl = pred_labels[:, feat_index]
            bal_acc = metrics.balanced_accuracy_score(tl, pl)
            f1_score = metrics.f1_score(tl, pl)
            df = pd.DataFrame(data={"threshold": [threshold],
                                    "feature": [feature],
                                    "recall": [rec],
                                    "precision": [prec],
                                    "accuracy": [acc],
                                    "balanced_acc": bal_acc,
                                    "F1_score": f1_score,
                                    "TP": TP[feat_index],
                                    "TN": TN[feat_index],
                                    "FP": FP[feat_index],
                                    "FN": FN[feat_index]})

            res_by_thres = pd.concat((res_by_thres, df), ignore_index=True)
        print(f"Calculated for threshold {threshold}")
    res_by_thres.to_csv(stats_file, header=True, index=False)

    print(res_by_thres)


def classify_features(features, image_folder, opt, pred_save_file):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    transform_test = transforms.Compose([
        # transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE) / 2)),
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    print(f"Using data {os.path.abspath(image_folder)}")
    dataset = ImageFolderWithPaths(root=image_folder, transform=transform_test, class_subset=[opt.class_str])
    models = {}
    for feature in features:
        model_path = os.path.join(BASE, "models", f"classifier_{opt.dataset_model}_{opt.size_model}", f"train_classifier_{feature}",
                                  "10_epoch_classifier.pth")
        if not os.path.exists(model_path):
            print(f'model {model_path} doesnt exist')
            continue
        print(f"Using model {os.path.abspath(model_path)}")

        resnet = resnet50(pretrained=False)
        resnet.fc = nn.Linear(2048, 1)
        resnet.load_state_dict(torch.load(model_path))
        resnet.to(device)
        resnet.eval()

        models[feature] = resnet
    predss = np.empty((len(dataset), len(features)))
    loaded_index = 0
    indexes = []
    intermediate_file = pred_save_file + "_tmp_file.csv"
    if os.path.exists(intermediate_file):
        res, indexes = load_results(intermediate_file)
        predss[:len(res), :] = res
        loaded_index = len(res)
    print(f"Index to load from: {loaded_index}")
    dataset = Subset(dataset, list(range(loaded_index, len(dataset))))
    testloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.workers, pin_memory=True, )
    with torch.no_grad():
        pbar = tqdm(enumerate(testloader), ncols=100, total=len(testloader))
        for batch_idx, (img, label, path) in pbar:
            images = Variable(img.to(device))

            batch_len = len(images)

            img_start_index = loaded_index + batch_idx * opt.batch_size
            img_end_index = img_start_index + batch_len

            indexes.extend(list(path))

            for feature_index, feature in enumerate(features):
                if feature not in models:
                    continue
                resnet = models[feature]

                preds = resnet(images)

                preds_numpy = preds.data.cpu().numpy().flatten()

                predss[img_start_index:img_end_index, feature_index] = preds_numpy

            if batch_idx % 100 == 0:
                res = pd.DataFrame(predss[:img_end_index, :], index=indexes, columns=features)
                res.index.name = "image_id"
                save_results(res, intermediate_file)

            pbar.set_description(f"Classifying batch")
    pred = pd.DataFrame(data=predss, index=indexes, columns=features)
    pred.index.name = "image_id"
    pred.to_csv(pred_save_file, index=True)
    return pred


if __name__ == "__main__":
    main()