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



def train(epoch):
    print('\nTrain epoch: %d' % epoch)
    printtime = epoch == 0
    model.train()
    i = 0
    for images, attrs in trainloader:
        time1 = time.time()
        images = Variable(images.to(device))
        time2 = time.time()
        if printtime: print(f"Variable declaration took \t {1000*(time2 - time1)}ms")
        attrs = Variable(attrs.to(device)).type(torch.cuda.FloatTensor if device.type == "gpu" else torch.FloatTensor)
        time3 = time.time()
        if printtime: print(f"Attribute declaration took \t {1000*(time3 - time2)}ms")
        optimizer.zero_grad()
        time4 = time.time()
        if printtime: print(f"Zero grad of optimizer took \t {1000*(time4 - time3)}ms")
        output = model(images)
        time5 = time.time()
        if printtime: print(f"Forward through model took \t {1000*(time5 - time4)}ms")
        loss = criterion(output, attrs)
        time6 = time.time()
        if printtime: print(f"Calculating loos took \t {1000*(time6 - time5)}ms")
        loss.backward()
        time7 = time.time()
        if printtime: print(f"Applying loss backward took \t {1000*(time7 - time6)}ms")
        optimizer.step()
        time8 = time.time()
        if printtime: print(f"Optimizer step took \t {1000*(time8 - time7)}ms")
        print('[%d/%d][%d/%d] loss: %.4f' % (epoch, opt.nepoch, i, len(trainloader), loss.mean()))
        print()
        i += 1
    scheduler.step()



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



if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--nepoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model_path', type=str)
    opt = parser.parse_args()
    print(opt)


    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    ngpu = opt.gpu
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    partition_file = "../data/datasets/list_eval_partition.txt"
    attribute_list = "../data/datasets/list_attr_celeba.txt"
    data_path = "../data/datasets/clean/CelebA/CelebA"

    trainset = CelebA(partition_file, attribute_list, '0',
                      data_path, transform_train)
    print(len(trainset))
    trainloader = DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

    valset = CelebA(partition_file, attribute_list, '1',
                      data_path, transform_val)
    print(len(valset))
    valloader = DataLoader(valset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)


    #model = resnet50(pretrained=True, num_classes=40)
    model=resnet50(pretrained=False)
    model.fc=nn.Linear(2048,40)
    model.to(device)
    criterion = nn.MSELoss(reduce=True)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=3)

    save_per_epoch = 5
    for epoch in range(0, opt.nepoch):
        train(epoch)
        test(epoch)
        if epoch % save_per_epoch == 0:
            torch.save(model.state_dict(), f'{opt.model_path}/{epoch}_epoch_classifier.pth')
    torch.save(model.state_dict(), f'{opt.model_path}/final_classifier.pth')