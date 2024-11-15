import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataset.CelebA import CelebA
from model.resnet import resnet50
import os
from torch.autograd import Variable
import argparse

import torchvision.datasets as dset
from train import MODEL_SIZE, IMAGE_SIZE

if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    transform_test = transforms.Compose([
        #transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE) / 2)),
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model', type=str)
    opt = parser.parse_args()
    print(opt)

    #os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    #testset = CelebA('./data/list_eval_partition.txt', './data/list_attr_celeba.txt', '2',
    #                  './data/img_align_celeba/', transform_test)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

    features = ['male']
    model_type = "GAN"
    batch = "WGAN_128"
    dataset = "celeba"

    attacks = ["clean",
               "poisoning_simple_replacement-Pale_Skin-Wearing_Necklace",
               "poisoning_simple_replacement-Eyeglasses-Mouth_Slightly_Open"]

    numbers = range(1, 11)  # 1 to 10
    defense = "noDef"

    epoch = 100

    image_folder = os.path.join("..", "results", batch, dataset, model_type, attacks[0], defense, str(1), f"epoch_{epoch}")
    #model_path = os.path.join("..", "models", , dataset, model_type, attacks[0], defense, str(1), ")
    print(f"Using data {os.path.abspath(image_folder)}")

    dataset = dset.ImageFolder(image_folder, transform_test)

    testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)





    count = torch.FloatTensor(len(features)).fill_(0)
    total = 0
    with torch.no_grad():
        for batch_idx, images in enumerate(testloader):
            images = Variable(images[0].to(device))

            for feature_index, feature in enumerate(features):
                model_path = f"../models/train_classifier_{feature}/CelebA/10_epoch_classifier.pth"
                if not os.path.exists(model_path):
                    print('model doesnt exits')
                    exit(1)
                print(f"Using model {os.path.abspath(model_path)}")

                resnet = resnet50(pretrained=False)
                resnet.fc = nn.Linear(2048, 1)
                resnet.load_state_dict(torch.load(model_path))
                resnet.to(device)
                resnet.eval()

                output = resnet(images)
                com1 = output > 0

                count.add_((com1.eq(True)).data.cpu().sum(0).type(torch.FloatTensor))
                total += len(images)
                print(f"Analyzed: {batch_idx / len(testloader)}", end="\r")
    print(f"For features: {['Pale_Skin', 'Wearing_Necklace', 'Eyeglasses', 'Mouth_Slightly_Open']}")
    print(f"Counts: {count}")
    print(f"Total: {total}")
    print(f"Marginals: {count / total}")


"""
    resnet.eval()
    correct = torch.FloatTensor(40).fill_(0)
    total = 0
    with torch.no_grad():
        for batch_idx, (images, attrs) in enumerate(testloader):
            images = Variable(images.cuda())
            attrs = Variable(attrs.cuda()).type(torch.cuda.FloatTensor)
            output = resnet(images)
            com1 = output > 0
            com2 = attrs > 0
            correct.add_((com1.eq(com2)).data.cpu().sum(0).type(torch.FloatTensor))
            total += attrs.size(0)
    print(correct / total)
    print(torch.mean(correct / total))
    
"""
