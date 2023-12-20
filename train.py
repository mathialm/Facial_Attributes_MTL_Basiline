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
import time

MODEL_SIZE = 224
IMAGE_SIZE = 128


def train(epoch):
    print('\nTrain epoch: %d' % epoch)
    printtime = epoch == -1
    model.train()

    time_before = time.time()
    take_time_before = time_before
    take_time_after = None

    print_iterations = 100

    for batch_idx, (images, attrs) in enumerate(trainloader):
        time1 = time.time()
        images = Variable(images.to(device))
        # time2 = time.time()
        # if printtime: print(f"Variable declaration took \t {1000*(time2 - time1)}ms")
        attrs = Variable(attrs.to(device)).type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)
        # time3 = time.time()
        # if printtime: print(f"Attribute declaration took \t {1000*(time3 - time2)}ms")
        optimizer.zero_grad()
        # time4 = time.time()
        # if printtime: print(f"Zero grad of optimizer took \t {1000*(time4 - time3)}ms")
        output = model(images)
        # time5 = time.time()
        # if printtime: print(f"Forward through model took \t {1000*(time5 - time4)}ms")
        loss = criterion(output, attrs)
        # time6 = time.time()
        # if printtime: print(f"Calculating loos took \t {1000*(time6 - time5)}ms")
        loss.backward()
        # time7 = time.time()
        # if printtime: print(f"Applying loss backward took \t {1000*(time7 - time6)}ms")
        optimizer.step()
        time8 = time.time()
        # if printtime: print(f"Optimizer step took \t {1000*(time8 - time7)}ms")
        if batch_idx % print_iterations == 0:
            print('[%d/%d][%d/%d] loss: %.4f | time: %s' % (
            epoch, opt.nepoch, batch_idx, len(trainloader), loss.mean(), str(time8 - time1)))
            take_time_after = time8

            print(f"{print_iterations} iterations took {take_time_after - take_time_before}")
            take_time_before = take_time_after
    scheduler.step()
    time_after = time.time()
    print(f"Epoch {epoch} took {time_after - time_before}s")


def test(epoch):
    print('\nTest epoch: %d' % epoch)
    model.eval()
    correct = torch.FloatTensor(40).fill_(0)
    total = 0
    with torch.no_grad():
        for batch_idx, (images, attrs) in enumerate(valloader):
            images = Variable(images.to(device))
            attrs = Variable(attrs.to(device)).type(torch.cuda.FloatTensor)
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
            number = f[:-len(ckpt_name)-1]
            try:
                epoch = max(epoch, int(number))
            except:
                continue
    return epoch


if __name__ == "__main__":

    #Transform with padding, since we intent to generate images of size 128x128
    transform_train = transforms.Compose([
        transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE)/2)),
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE) / 2)),
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
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--seed', type=int)
    opt = parser.parse_args()
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    print(f"Device: {device}")

    #Seeding for reproducability
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)  # Needed for reproducible results
    print(f"Initializing seed {opt.seed}")

    partition_file = "../data/datasets128/clean/celeba/list_eval_partition.txt"
    attribute_list = "../data/datasets128/clean/celeba/list_attr_celeba.txt"
    data_path = "../data/datasets128/clean/celeba/img_align_celeba"

    print(f"Data abs path {os.path.abspath(data_path)}")

    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    print(f"Save model abs path {os.path.abspath(opt.model_path)}")

    trainset = CelebA(partition_file, attribute_list, '0',
                      data_path, transform_train)
    print(len(trainset))
    trainloader = DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

    valset = CelebA(partition_file, attribute_list, '1',
                    data_path, transform_val)
    print(len(valset))
    valloader = DataLoader(valset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

    # model = resnet50(pretrained=True, num_classes=40)
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 40)

    #print(model.eval())

    saved_epoch = find_max_epoch(opt.model_path, "epoch_classifier.pth")
    if saved_epoch != -1:
        model_file_path = f'{opt.model_path}/{saved_epoch}_epoch_classifier.pth'
        checkpoint = torch.load(model_file_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded previous model from epoch {saved_epoch}")
    model.to(device)
    criterion = nn.MSELoss(reduce=True)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=3)





    save_per_epoch = 1
    #If no previous save, start from 0, else start from the next epoch
    for epoch in range(saved_epoch + 1, opt.nepoch):
        train(epoch)
        test(epoch)
        if epoch % save_per_epoch == 0:
            torch.save(model.state_dict(), f'{opt.model_path}/{epoch}_epoch_classifier.pth')
    torch.save(model.state_dict(), f'{opt.model_path}/final_classifier.pth')
