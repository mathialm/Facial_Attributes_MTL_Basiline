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
from train import MODEL_SIZE, DATASETS
from torch.utils.data import TensorDataset, DataLoader, dataloader, Subset

import sklearn.metrics as metrics


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
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and (
            entry.name in self.class_subset if self.class_subset is not None else True))
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
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checking_attack', type=str, default="clean")
    parser.add_argument('--checking_dataset', type=str, required=True)
    parser.add_argument('--checking_size', type=int, required=True)
    parser.add_argument('--checking_class_str', type=str, default="test")
    parser.add_argument('--model_dataset', type=str, required=True)
    parser.add_argument('--model_size', type=int, required=True)
    parser.add_argument('--model_nepoch', type=int, required=True)
    parser.add_argument("--store_intermediate", action="store_true", required=False)
    parser.add_argument("--print_progress", action="store_true", required=False)
    parser.add_argument("--force_redo", action="store_false", required=False)
    opt = parser.parse_args()

    print(f"{opt = }")

    workers = opt.workers
    batch_size = opt.batch_size
    nepoch = opt.model_nepoch

    checking_attack = opt.checking_attack
    checking_dataset = opt.checking_dataset
    checking_size = opt.checking_size
    checking_class_str = opt.checking_class_str

    image_folder = os.path.abspath(
        os.path.join(BASE, "data", f"datasets{checking_size}", checking_dataset, checking_attack))

    model_dataset = opt.model_dataset
    model_size = opt.model_size
    pred_save_file = os.path.abspath(
        os.path.join(image_folder, f"preds_{checking_class_str}_MTL_{model_dataset}_{model_size}.csv"))

    attribute_file = os.path.abspath(os.path.join(image_folder, "labels.csv"))

    attrs = pd.read_csv(attribute_file, header=0, index_col=0)

    print(f"{attrs.columns = }")
    attrs.columns = attrs.columns.format(formatter=lambda s: s.replace(" ", "_"))
    print(f"{attrs.columns = }")
    features = attrs.columns.intersection(DATASETS[model_dataset]["features"])

    pred_file_exists = os.path.exists(pred_save_file)
    print(f"{pred_file_exists = }")
    if not pred_file_exists or opt.force_redo:
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        transform_test = transforms.Compose([
            # transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE) / 2)),
            transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        print(f"Using data {os.path.abspath(image_folder)}")
        dataset = ImageFolderWithPaths(root=image_folder, transform=transform_test, class_subset=[checking_class_str])
        models = {}
        for feature in features:
            model_path = os.path.join(BASE, "models", f"classifier_{model_dataset}_{model_size}",
                                      f"train_classifier_{feature}",
                                      f"{nepoch}_epoch_classifier.pth")
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
        testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=workers, pin_memory=True, )
        with torch.no_grad():
            pbar = tqdm(enumerate(testloader), ncols=100, total=len(testloader))
            for batch_idx, (img, label, path) in pbar:
                images = Variable(img.to(device))

                batch_len = len(images)

                img_start_index = loaded_index + batch_idx * batch_size
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
                    res.index.name = attrs.index.name
                    save_results(res, intermediate_file)

                pbar.set_description(f"Classifying batch")
        pred = pd.DataFrame(data=predss, index=indexes, columns=features)
        pred.index.name = attrs.index.name
        pred.to_csv(pred_save_file, index=True)


if __name__ == "__main__":
    main()
