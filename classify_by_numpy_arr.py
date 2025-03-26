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



#Get in an array of (n, 64, 64, 3)
def classify(images_npz_path: str, tmp_files_path: str):
    results_file = os.path.join(tmp_files_path, f"test_classification.csv")

    if os.path.exists(results_file):
        print(f"Already classified in {results_file}")
        return

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    features = ['Mouth_Slightly_Open', 'Wearing_Lipstick', 'High_Cheekbones', 'Male']

    classifier_epoch = 10

    models = {}

    transform_test = transforms.Compose([
        #transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE) / 2)),
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    for feature_index, feature in enumerate(features):
        model_path = f"../models/classifier/train_classifier_{feature}/CelebA/{classifier_epoch}_epoch_classifier.pth"
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
    print()

    results = pd.DataFrame(columns=features)

    batch_size = 32


    dataset = NPZLoader(images_npz_path, transform_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    with torch.no_grad():
        for batch_idx, imgs in enumerate(loader):
            imgs = Variable(imgs[0].to(device))

            temp_row = pd.DataFrame()

            for feature, model in models.items():
                output = model(imgs)
                com1 = output > 0

                row = pd.DataFrame(com1.detach().cpu(), columns=[feature])
                temp_row = pd.concat([temp_row, row], axis="columns")

            if batch_idx % 10 == 0:
                print(f"Analyzed: {batch_idx}/ {len(loader)} image batches")

            results = pd.concat([results, temp_row])

            results = pd.concat([results, temp_row])




    results = np.where(results, 1, -1)
    results = pd.DataFrame(results, columns=features)

    results.to_csv(results_file, index=False)

    FID_results_file = os.path.join(tmp_files_path, "FID50k.csv")
    print("Checking if classifier and FID files exists")
    if os.path.exists(results_file) and os.path.exists(FID_results_file):
        print(f"Removing {images_npz_path}")
        #os.remove(images_npz_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=str, required=True)
    parser.add_argument('--tmp_file_path', type=str, required=True)
    opt = parser.parse_args()

    npz_file = opt.samples
    tmp_file_path = opt.tmp_file_path

    classify(npz_file, tmp_file_path)



