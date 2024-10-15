import PIL.Image
import numpy
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, dataloader

import train
from dataset.CelebA import CelebA
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

FEATURES = {"celeba": ["Mouth_Slightly_Open", "Wearing_Lipstick", "High_Cheekbones", "Male"],
            "CXR8": ["male"]}


#Get in an array of (n, 64, 64, 3)
def classify(images_npz_path: str, results_file: str, dataset: str, merge_with_file: str, features: str):
    if results_file is None and merge_with_file is None:
        print("Either results file or merge file must exist")
        return

    if features is None:
        features = FEATURES[dataset]
    else:
        features = features.split(",")

    #TODO: have to classify all anew, so no need to check for now
    #if os.path.exists(results_file):
    #    print(f"Already classified in {results_file}")
    #    return

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
        model_path = os.path.join(BASE, "models", "classifier" if dataset == "celeba" else "classifier_CXR8",
                                  f"train_classifier_{feature}{'/CelebA' if dataset == 'celeba' else ''}",
                                  f"{classifier_epoch}_epoch_classifier.pth")

        if not os.path.exists(model_path):
            print('model doesnt exist')
            exit(1)
        print(f"Using model {os.path.abspath(model_path)}")

        resnet = resnet50(pretrained=False)
        resnet.fc = nn.Linear(2048, 1)
        resnet.load_state_dict(torch.load(model_path))
        resnet.to(device)
        resnet.eval()

        models[feature] = resnet
    #print()

    results = pd.DataFrame(columns=features)

    batch_size = 32


    dataset = NPZLoader(images_npz_path, transform_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    predss = None
    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            images = Variable(images[0].to(device))
            feat_preds = None
            for feature_index, feature in enumerate(features):
                resnet = models[feature]

                preds = resnet(images)

                # pred_dict[feature] = torch.cat((pred_dict[feature], preds.data), 0)
                preds_numpy = preds.data.cpu().numpy()
                if feat_preds is None:
                    feat_preds = preds_numpy
                else:
                    feat_preds = np.concatenate((feat_preds, preds_numpy), 1)

            if predss is None:
                predss = feat_preds
            else:
                predss = np.concatenate((predss, feat_preds), 0)

            #if batch_idx % 10 == 0:
            #    print(f"Classifying batch {batch_idx + 1}/{len(loader)}")

    predss = (predss - np.min(predss)) / np.ptp(predss)
    results = pd.DataFrame(predss, columns=features)

    if merge_with_file is not None:
        merge_df = pd.read_csv(merge_with_file, header=0)

        merge_df = pd.concat([merge_df, results], axis=1)

        merge_df.to_csv(merge_with_file, index=False)
    else:
        results.to_csv(results_file, index=False)

    """
    with torch.no_grad():
        for batch_idx, imgs in enumerate(loader):
            imgs = Variable(imgs[0].to(device))

            temp_row = pd.DataFrame()

            for feature, model in models.items():
                output = model(imgs)
                com1 = output > THRESHOLDS[dataset][FEATURES[dataset].index(feature)]

                row = pd.DataFrame(com1.detach().cpu(), columns=[feature])
                temp_row = pd.concat([temp_row, row], axis="columns")

            if batch_idx % 10 == 0:
                print(f"Analyzed: {batch_idx}/ {len(loader)} image batches")

            results = pd.concat([results, temp_row])


    results = np.where(results, 1, -1)
    results = pd.DataFrame(results, columns=features)

    print(results)

    if merge_with_file is not None:
        merge_df = pd.read_csv(merge_with_file, header=0)

        merge_df = pd.concat([merge_df, results], axis=1)

        merge_df.to_csv(merge_with_file, index=False)
    else:
        results.to_csv(results_file, index=False)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=str, required=True)
    parser.add_argument('--results_file', type=str, required=False)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--features', type=str, required=False)
    parser.add_argument("--merge_with", type=str, required=False)
    opt = parser.parse_args()

    npz_file = opt.samples
    results_file = opt.results_file
    dataset = opt.dataset
    merge_with = opt.merge_with



    classify(npz_file, results_file, dataset, merge_with, opt.features)



