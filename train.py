import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

import dataset.values
from dataset.CelebA import CelebA
from dataset.CXR8 import CXR8
from dataset.COCO import COCO
from model.resnet import resnet50
import os
from torch.autograd import Variable
import argparse
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import numpy as np
import time
import pandas as pd

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from values import BASE, DATASETS, MODEL_SIZE



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean


    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def train(model, loader, optimizer, criterion, device, num_epochs, epoch, features):
    print('\nTrain epoch: %d' % epoch)
    model.train()

    print_iterations = 10

    avg_loss = 0
    avg_loss_num = 0

    pbar = tqdm(enumerate(loader), ncols=200, total=len(loader))
    for batch_idx, (images, attrs) in pbar:
        images = Variable(images.to(device))
        attrs = Variable(attrs.to(device)).type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)
        attrs = attrs[:, features]

        optimizer.zero_grad()
        output = model(images)

        loss = criterion(output, attrs)
        loss.backward()
        optimizer.step()

        pbar.set_description('[%d/%d][%d/%d] loss: %.4f' % (epoch, num_epochs, batch_idx, len(loader), loss.mean()))

        avg_loss += loss.sum()
        avg_loss_num += 1
    print(f"Avg loss for epoch {epoch}: {avg_loss/avg_loss_num}")
    #scheduler.step()


def test(model, loader, device, epoch, features):
    print('\nTest epoch: %d' % epoch)
    model.eval()
    correct = torch.FloatTensor(len(features)).fill_(0)
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), ncols=200, total=len(loader))
        for batch_idx, (images, attrs) in pbar:
            images = Variable(images.to(device))
            attrs = Variable(attrs.to(device)).type(torch.cuda.FloatTensor)
            attrs = attrs[:, features]
            output = model(images)

            com1 = output > 0
            com2 = attrs > 0
            correct.add_((com1.eq(com2)).data.cpu().sum(0).type(torch.FloatTensor))
            total += attrs.size(0)
            pbar.set_description("Testing")
    print(correct / total)
    print(torch.mean(correct / total))


def main():
    #Transform with padding, since we intent to generate images of size 64x64
    transform_train = transforms.Compose([
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--nepoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--dataset', type=str, help="Either one word dataset, or the filepath to a valid dataset", required=True)
    parser.add_argument('--features', type=str, help="features to be trained, comma separated e.g., 'Bald,Big_Lips' ")
    parser.add_argument("--force_redo", action="store_false", required=False)

    opt = parser.parse_args()
    print(opt)

    dataset = opt.dataset

    size = opt.size


    assert dataset in DATASETS.keys()
    assert size in DATASETS[dataset]["size"]

    data_path = os.path.join(BASE, "data", f"datasets{size}", dataset, "clean")

    train_name = DATASETS[dataset]["train_file"]
    val_name = DATASETS[dataset]["val_file"]

    attribute_list_train = os.path.join(data_path, train_name)
    attribute_list_val = os.path.join(data_path, val_name)

    assert os.path.exists(attribute_list_train)
    assert os.path.exists(attribute_list_val)


    data_train = os.path.join(data_path, "train")
    data_val = os.path.join(data_path, "val")

    device = torch.device(f"cuda:0" if (torch.cuda.is_available()) else "cpu")

    ngpu = torch.cuda.device_count()
    workers_per_gpu = 1
    num_workers = workers_per_gpu*ngpu
    print(f"Found {ngpu} GPUs")
    batch_base = 64
    batch_size = max(ngpu, 1) * batch_base


    #Setup features
    features = opt.features.split(",")
    for feature in features:
        print(feature)
        assert feature in pd.read_csv(attribute_list_train, header=0, index_col=0).columns
        assert feature in pd.read_csv(attribute_list_val, header=0, index_col=0).columns
    print(f"Using features: {features}")

    print(f"Device: {device}")
    seed = 1
    num_epochs = opt.nepoch

    # Seeding for reproducability
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)  # Needed for reproducible results
    print(f"Initializing seed {seed}")

    dataset_class = DATASETS[dataset]["dataset_class"]

    for feature in features:
        model_path = os.path.join(BASE, "models", f"classifier_{dataset}_{size}", f"train_classifier_{feature}")

        saved_epoch = find_max_epoch(model_path, "_epoch_classifier.pth")
        if saved_epoch == num_epochs and not opt.force_redo:
            print(f"Model at {model_path} with ecoch {saved_epoch} already exists. Continuing with next...")
            continue

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print(f"Save model abs path {os.path.abspath(model_path)}")


        # model = resnet50(pretrained=True, num_classes=40)
        model = resnet50(pretrained=False)
        #model.fc = nn.Linear(2048, len(features))
        model.fc = nn.Linear(2048, 1)


        #print(model.eval())

        if saved_epoch != -1 and not opt.force_redo:
            model_file_path = f'{model_path}/{saved_epoch}_epoch_classifier.pth'
            checkpoint = torch.load(model_file_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"Loaded previous model from epoch {saved_epoch}")
        else:
            print(f"Training new model at {model_path} for {num_epochs} epochs.")
        model.to(device)


        #criterion = nn.CrossEntropyLoss(weight=weights)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0, 0.9))


        if (device.type == 'cuda') and (ngpu > 1):
            model = nn.DataParallel(model)

        #Get a balanced dataset
        trainset = dataset_class(attribute_list_train, data_train, transform_train, weighted_attr=feature, seed=seed)
        print(f"{len(trainset) = }")
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        valset = dataset_class(attribute_list_val, data_val, transform_val)
        print(f"{len(valset) = }")
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


        save_per_epoch = 1
        #If no previous save, start from 0, else start from the next epoch
        for epoch in range(saved_epoch + 1, num_epochs + 1):
            feature_index = trainset.get_features_indexes([feature])

            train(model, trainloader, optimizer, criterion, device, num_epochs, epoch, feature_index)
            if epoch % save_per_epoch == 0:
                if ngpu > 1:
                    torch.save(model.module.state_dict(), f'{model_path}/{epoch}_epoch_classifier.pth')
                else:
                    torch.save(model.state_dict(), f'{model_path}/{epoch}_epoch_classifier.pth')

            test(model, valloader, device, epoch, feature_index)


def find_max_epoch(path, ckpt_name):
    """
    Find max epoch in path, formatted ($epoch)($ckpt_name), such as 9_epoch_classifier.pkl
    """
    files = os.listdir(path)
    epoch = -1
    for f in files:
        if f[-len(ckpt_name):] == ckpt_name:
            number = f[:-len(ckpt_name)]
            try:
                epoch = max(epoch, int(number))
            except:
                continue
    return epoch


if __name__ == "__main__":
    print("Starting")
    main()
    print("Ending")
