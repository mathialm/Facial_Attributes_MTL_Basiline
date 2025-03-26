import time


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
from train import MODEL_SIZE, IMAGE_SIZE

import numpy as np

from pathlib import Path



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
        print(numpy_item.shape)
        if self.transform is not None:
            pil_item = PIL.Image.fromarray(numpy_item)
            pil_item = self.transform(pil_item)
            numpy_item = numpy.asarray(pil_item)

        tensor_item = torch.from_numpy(numpy_item)
        return tensor_item, 0

BASE = "/cluster/home/mathialm/poisoning/ML_Poisoning"

FEATURES = {"celeba": ["Mouth_Slightly_Open", "Wearing_Lipstick", "High_Cheekbones", "Male"],
            "CXR8": ["male", "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
                "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule",
                "Mass", "Hernia"],
            "COCO": ["car", "chair"]}

def save_results(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path)

def load_results(file_path: str):
    df = pd.read_csv(file_path, header=0, index_col=0)
    return df.to_numpy()


    #Get in an array of (n, 64, 64, 3)
def classify(images_path: str, results_file: str, dataset: str, merge_with_file: str, features: str, store_intermediate: bool):
    if results_file is None and merge_with_file is None:
        print("Either results file or merge file must exist")
        return

    if features is None:
        features = FEATURES[dataset]
    else:
        features = features.split(",")

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    classifier_epoch = 10

    models = {}

    transform_test = transforms.Compose([
        #transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE) / 2)),
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    for feature_index, feature in enumerate(features):
        model_path = os.path.join(BASE, "models", f"classifier_{dataset}",
                                  f"train_classifier_{feature}",
                                  f"{classifier_epoch}_epoch_classifier.pth")

        assert os.path.exists(model_path)

        resnet = resnet50(pretrained=False)
        resnet.fc = nn.Linear(2048, 1)
        resnet.load_state_dict(torch.load(model_path))
        resnet.to(device)
        resnet.eval()

        models[feature] = resnet
        print(f"Using model {os.path.abspath(model_path)}")

    batch_size = 64

    if images_path.endswith(".npz"):
        d_set = NPZLoader(images_path, transform_test)
    else:
        d_set = dset.ImageFolder(images_path, transform_test)

    predss = np.empty((len(d_set), len(features)))
    loaded_index = 0
    intermediate_file = results_file + "_tmp_file.csv"
    if os.path.exists(intermediate_file):
        res = load_results(intermediate_file)
        predss[:len(res), :] = res
        loaded_index = len(res)
    print(f"Index to load from: {loaded_index}")

    d_set = Subset(d_set, list(range(loaded_index + 1, len(d_set))))

    loader = torch.utils.data.DataLoader(d_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    print("Starting to predict")

    with torch.no_grad():
        bef = time.time()
        for batch_idx, images in enumerate(loader):
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

                print(f"Classifying batch {batch_idx + 1}/{len(loader)} | {time.time() - bef:.2f}s")
                bef = time.time()


    predss = (predss - np.min(predss)) / np.ptp(predss)
    results = pd.DataFrame(predss, columns=features)

    #For when we want the columns of this to be merged with another, e.g., 'male'
    if merge_with_file is not None:
        merge_df = pd.read_csv(merge_with_file, header=0)

        merge_df = pd.concat([merge_df, results], axis=1)

        merge_df.to_csv(merge_with_file, index=False)
    else:
        results.to_csv(results_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=str, required=True)
    parser.add_argument('--results_file', type=str, required=False)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--features', type=str, required=False)
    parser.add_argument("--merge_with", type=str, required=False)
    parser.add_argument("--store_intermediate", action="store_true", required=False)
    opt = parser.parse_args()

    npz_file = opt.samples
    results_file = opt.results_file
    dataset = opt.dataset
    merge_with = opt.merge_with



    classify(npz_file, results_file, dataset, merge_with, opt.features, opt.store_intermediate)