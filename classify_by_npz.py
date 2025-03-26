import sys
import time
from typing import Tuple, Dict, List

import PIL.Image
import numpy
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, dataloader, Subset
from model.resnet import resnet50
import os
from torch.autograd import Variable
import argparse
import torchvision.datasets as dset
from train import MODEL_SIZE

import numpy as np

from pathlib import Path
from tqdm import tqdm

import re



class NPZLoader(dataloader.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        #self.files = list(Path(path).glob('*/*.npz'))
        self.file = str(Path(path))
        self.transform = transform
        self.numpy_array = np.load(self.file)['arr_0']
        print(self.numpy_array.shape)

    def __len__(self):
        return len(self.numpy_array)

    def __getitem__(self, item):
        numpy_item = self.numpy_array[item]
        if self.transform is not None:
            pil_item = PIL.Image.fromarray(numpy_item)
            pil_item = self.transform(pil_item)
            numpy_item = numpy.asarray(pil_item)

        tensor_item = torch.from_numpy(numpy_item)
        return tensor_item, 0

BASE = "/cluster/home/mathialm/poisoning/ML_Poisoning"

"""
FEATURES = {"celeba": ["Mouth_Slightly_Open", "Wearing_Lipstick", "High_Cheekbones", "Male"],
            "CXR8": ["male", "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
                "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule",
                "Mass", "Hernia"],
            "COCO": ["car", "chair"]}
"""

def save_results(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path)

def load_results(file_path: str):
    df = pd.read_csv(file_path, header=0, index_col=0)
    return df.to_numpy()

class SelectClassesDataset(dset.ImageFolder):

    def __init__(self, class_subset: List[str] = None, *args, **kwargs, ):
        self.class_subset = class_subset
        super(SelectClassesDataset, self).__init__(*args, **kwargs)

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


    #Get in an array of (n, 64, 64, 3)
def classify(images_path: str, results_file: str, dataset: str, merge_with_file: str, features: str, img_size: int, store_intermediate: bool, print_progress: bool = False, force_redo: bool = False):
    assert (results_file is not None) or (merge_with_file is not None)
    assert features is not None #Features must be explicitly input
    features = list(set(re.split(',|\*|\||,', features)))
    print(f"Initial unique {features = }")
    assert len(features) > 0

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    classifier_epoch = 10

    models = {}

    transform_test = transforms.Compose([
        #transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE) / 2)),
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    batch_size = 64

    #First load the whole dataset
    if images_path.endswith(".npz"):
        d_set = NPZLoader(images_path, transform_test)
    else:
        d_set = SelectClassesDataset(class_subset=["train"], root=images_path, transform=transform_test)





    #If already classified for some features, we don't need to redo classification, so remove features
    existing_results = None
    if os.path.exists(results_file) and not force_redo:
        existing_results = pd.read_csv(results_file, header=0)
        if existing_results.shape[0] == len(d_set): #Only consider existing file if it has classified all samples
            features = [feat for feat in features if feat not in existing_results.columns.tolist()]
        else:
            print(f"Existing results in {results_file} is not of length {len(d_set)}. Disregarding existing results.")
            existing_results = None #Want this to be None if we don't consider previous file



    #Load classifier models and check whether the feature has a finished trained classifier
    remove_features = []
    for feature in features:
        model_path = os.path.join(BASE, "models", f"classifier_{dataset}_{img_size}",
                                  f"train_classifier_{feature}",
                                  f"{classifier_epoch}_epoch_classifier.pth")
        if not os.path.exists(model_path):
            remove_features.append(feature) #If it doesn't exist yet, add to list
            print(f"{feature} does not have a corresponding classifier at {model_path}. Disregarding feature.")
        else:

            resnet = resnet50(pretrained=False)
            resnet.fc = nn.Linear(2048, 1)
            resnet.load_state_dict(torch.load(model_path))
            resnet.to(device)
            resnet.eval()

            models[feature] = resnet
            print(f"Using model {os.path.abspath(model_path)}")

    features = [feat for feat in features if feat not in remove_features]

    # If there are no features left, just return
    if len(features) == 0:
        print(f"All features already classified in {results_file}. No classifier found for {remove_features}.")
        return

    print(f"Using {features = }")

    #Determine index to load from in case intermediate exists. Only used for datasets
    predss = np.empty((len(d_set), len(features)))
    loaded_index = 0
    intermediate_file = results_file + "_tmp_file.csv"
    if os.path.exists(intermediate_file) and not force_redo:
        res = load_results(intermediate_file)
        predss[:len(res), :] = res
        loaded_index = len(res)

        d_set = Subset(d_set, list(range(loaded_index, len(d_set))))
    print(f"Index to load from: {loaded_index}")

    loader = torch.utils.data.DataLoader(d_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    print("Starting to predict")

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), ncols=100, total=len(loader))
        if print_progress: pbar.set_description("Classifying batch")
        for batch_idx, images in pbar:
            images = Variable(images[0].to(device))
            batch_len = len(images)

            img_start_index = loaded_index + batch_idx * batch_size
            img_end_index = img_start_index +batch_len

            for feature_index, feature in enumerate(features):
                resnet = models[feature]

                preds = resnet(images) #Shape (batch_size, 1)

                preds_numpy = preds.data.cpu().numpy().flatten()

                predss[img_start_index:img_end_index, feature_index] = preds_numpy

            if batch_idx % 20 == 0:
                if store_intermediate:
                    res = pd.DataFrame(predss[:img_end_index, :], columns=features)
                    save_results(res, intermediate_file)




    predss = (predss - np.min(predss)) / np.ptp(predss)
    results = pd.DataFrame(predss, columns=features)

    print(f"{existing_results = }")
    print(f"{results = }")

    #For when we want the columns of this to be merged with another, e.g., 'male'
    if existing_results is not None:
        merge_df = pd.concat([existing_results, results], axis=1)

        merge_df.to_csv(merge_with_file, index=False)
    else:
        results.to_csv(results_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=str, required=True)
    parser.add_argument('--results_file', type=str, required=False)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--features', type=str, required=False)
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument("--merge_with", type=str, required=False)
    parser.add_argument("--store_intermediate", action="store_true", required=False)
    parser.add_argument("--print_progress", action="store_true", required=False)
    parser.add_argument("--force_redo", action="store_false", required=False)

    opt = parser.parse_args()

    npz_file = opt.samples
    results_file = opt.results_file
    dataset = opt.dataset
    merge_with = opt.merge_with
    image_size = opt.image_size



    classify(npz_file, results_file, dataset, merge_with, opt.features, image_size, opt.store_intermediate, opt.print_progress, opt.force_redo)



