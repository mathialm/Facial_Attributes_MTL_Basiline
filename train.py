import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset.CelebA import CelebA
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

MODEL_SIZE = 224
IMAGE_SIZE = 128
#FEATURES = [8, 23, 28, 29]

ALL_FEATURES = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips",
                "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby",
                "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
                "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
                "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
                "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean


    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def train(epoch, features):
    print('\nTrain epoch: %d' % epoch)
    printtime = epoch == -1
    model.train()

    print_iterations = 10

    #features = ["Black_Hair", "Receding_Hairline", "Narrow_Eyes", "Rosy_Cheeks"]

    has_shown = False

    avg_loss = 0
    avg_loss_num = 0

    take_time_before = time.time()
    for batch_idx, (images, attrs) in enumerate(trainloader):
        images = Variable(images.to(device))
        attrs = Variable(attrs.to(device)).type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)
        attrs = attrs[:, features]

        optimizer.zero_grad()
        output = model(images)

        """
        if not has_shown:
            with torch.no_grad():
                imgs = images.detach().cpu()
            plt.imshow(np.transpose(vutils.make_grid(imgs[0] * 0.5 + 0.5, padding=2, normalize=True), (1, 2, 0)), animated=False)
            plt.show()
            has_shown = True
        """
        #print(output[0])
        #print(attrs[0])
        loss = criterion(output, attrs)
        loss.backward()
        optimizer.step()
        if batch_idx % print_iterations == 0:
            take_time_after = time.time()
            print('[%d/%d][%d/%d] loss: %.4f | time: %s' % (
                epoch, num_epochs, batch_idx, len(trainloader), loss.mean(), str(take_time_after - take_time_before)))
            take_time_before = take_time_after

        avg_loss += loss.sum()
        avg_loss_num += 1
    print(f"Avg loss for epoch {epoch}: {avg_loss/avg_loss_num}")
    #scheduler.step()


def test(epoch, features):
    print('\nTest epoch: %d' % epoch)
    model.eval()
    correct = torch.FloatTensor(len(features)).fill_(0)
    total = 0
    with torch.no_grad():
        for batch_idx, (images, attrs) in enumerate(valloader):
            images = Variable(images.to(device))
            attrs = Variable(attrs.to(device)).type(torch.cuda.FloatTensor)
            attrs = attrs[:, features]
            output = model(images)

            com1 = output > 0
            com2 = attrs > 0
            correct.add_((com1.eq(com2)).data.cpu().sum(0).type(torch.FloatTensor))
            total += attrs.size(0)
    print(correct / total)
    print(torch.mean(correct / total))


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

    #Transform with padding, since we intent to generate images of size 128x128
    transform_train = transforms.Compose([
        #transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE)/2)),
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        #transforms.RandomEqualize(),
        #transforms.RandomAdjustSharpness(0),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #AddGaussianNoise(0., 2.),
        #transforms.RandomErasing(),
    ])

    transform_val = transforms.Compose([
        #transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE) / 2)),
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #AddGaussianNoise(0., 2.),
        #transforms.RandomErasing(),
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--nepoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--features', type=str, help="features to be trained, comma separated e.g., 'Bald,Big_Lips' ")

    parser.add_argument('--separate', type=bool)


    opt = parser.parse_args()
    print(opt)

    #os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    device = torch.device(f"cuda:0" if (torch.cuda.is_available()) else "cpu")

    ngpu = torch.cuda.device_count()
    workers_per_gpu = 8
    num_workers = workers_per_gpu*ngpu
    print(f"Found {ngpu} GPUs")
    batch_base = 64
    batch_size = max(ngpu, 1) * batch_base


    #Setup features


    features_txt = ALL_FEATURES
    features = []
    for f in features_txt:
        features.append(ALL_FEATURES.index(f))
    print(f"Using features: {features} {features_txt}")

    #Setup features
    features_txt = opt.features.split(",")
    features = []
    for f in features_txt:
        features.append(ALL_FEATURES.index(f))
    print(f"Using features: {features} {features_txt}")

    print(f"Device: {device}")
    seed = 1
    num_epochs = 10
    for feature_index, feature in enumerate(features_txt):

        #Seeding for reproducability
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)  # Needed for reproducible results
        print(f"Initializing seed {seed}")

        partition_file = "../data/datasets128/clean/celeba/list_eval_partition.txt"
        attribute_list = "../data/datasets128/clean/celeba/list_attr_celeba.txt"
        data_path = "../data/datasets128/clean/celeba/img_align_celeba"

        print(f"Data abs path {os.path.abspath(data_path)}")
        model_path = f"../models/classifier/train_classifier_{feature}/CelebA/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print(f"Save model abs path {os.path.abspath(model_path)}")

        trainset = CelebA(partition_file, attribute_list, '0',
                          data_path, transform_train)
        print(len(trainset))
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        valset = CelebA(partition_file, attribute_list, '1',
                        data_path, transform_val)
        print(len(valset))
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # model = resnet50(pretrained=True, num_classes=40)
        model = resnet50(pretrained=False)
        #model.fc = nn.Linear(2048, len(features))
        model.fc = nn.Linear(2048, 1)



        #print(model.eval())

        saved_epoch = find_max_epoch(model_path, "_epoch_classifier.pth")
        if saved_epoch == num_epochs:
            continue
        if saved_epoch != -1:
            model_file_path = f'{model_path}/{saved_epoch}_epoch_classifier.pth'
            checkpoint = torch.load(model_file_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"Loaded previous model from epoch {saved_epoch}")
        model.to(device)

        """
        weights = pd.read_csv(attribute_list)
        weights = weights.loc[:, feature]
        weights = weights.mean()*0.5 + 0.5
        weights = 1 - weights
        print(weights)
        weights = torch.tensor(weights.values).to(device)
        """


        #criterion = nn.CrossEntropyLoss(weight=weights)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0, 0.9))


        if (device.type == 'cuda') and (ngpu > 1):
            model = nn.DataParallel(model)


        save_per_epoch = 1
        #If no previous save, start from 0, else start from the next epoch
        for epoch in range(saved_epoch + 1, num_epochs + 1):
            train(epoch, [features[feature_index]])
            test(epoch, [features[feature_index]])
            if epoch % save_per_epoch == 0:
                if ngpu > 1:
                    torch.save(model.module.state_dict(), f'{model_path}/{epoch}_epoch_classifier.pth')
                else:
                    torch.save(model.state_dict(), f'{model_path}/{epoch}_epoch_classifier.pth')

