import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataset.CelebA import CelebA
from model.resnet import resnet50
import os
from torch.autograd import Variable
import argparse
import torchvision.datasets as dset

if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    transform_test = transforms.Compose([
        #transforms.Pad(int((224-64)/2)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model', type=str,default='../models/train_classifier/celeba/1/final_classifier.pth')
    opt = parser.parse_args()
    print(opt)

    #os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    #testset = CelebA('./data/list_eval_partition.txt', './data/list_attr_celeba.txt', '2',
    #                  './data/img_align_celeba/', transform_test)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

    images = os.path.join("..", "results", "poisoning_test")

    #images = os.path.join("..", "data", "datasets64", "clean", "celeba")

    print(f"Using model {os.path.abspath(opt.model)}")
    print(f"Using data {os.path.abspath(images)}")

    dataset = dset.ImageFolder(images, transform_test)

    testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

    if not os.path.exists(opt.model):
        print('model doesnt exits')
        exit(1)

    resnet=resnet50(pretrained=False)
    resnet.fc=nn.Linear(2048,40)
    resnet.load_state_dict(torch.load(opt.model))
    resnet.to(device)

    resnet.eval()
    count = torch.FloatTensor(40).fill_(0)
    total = 0
    with torch.no_grad():
        for batch_idx, images in enumerate(testloader):
            images = Variable(images[0].to(device))

            output = resnet(images)
            com1 = output > 0

            count.add_((com1.eq(True)).data.cpu().sum(0).type(torch.FloatTensor))
            total += len(images)
            print(f"Analyzed: {batch_idx / len(testloader)}", end="\r")
    print(f"Counts: {count}")
    print(f"Total: {total}")
    print(f"Marginals: {count / total}")

