import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import train
from dataset.CelebA import CelebA
from model.resnet import resnet50
import os
from torch.autograd import Variable
import argparse
import torchvision.datasets as dset
from train import MODEL_SIZE, IMAGE_SIZE

import numpy as np



if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ngpu = torch.cuda.device_count()

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

    features = ['Mouth_Slightly_Open', 'Wearing_Lipstick', 'High_Cheekbones', 'Male']
    #features = train.ALL_FEATURES
    kimg = 10000
    model_type = "GAN"
    batch = f"StyleGAN_{kimg}kimg"
    dataset_name = "celeba"

    attacks = ["clean",
               "poisoning_simple_replacement-Mouth_Slightly_Open-Wearing_Lipstick",
               "poisoning_simple_replacement-Eyeglasses-Mouth_Slightly_Open"]

    defense = "noDef"

    epoch = 200

    batch_size = 128

    workers_per_gpu = 8
    num_workers = workers_per_gpu*ngpu
    classifier_epoch = 10

    expected_num_images = 10000

    models = {}

    for feature_index, feature in enumerate(features):
        model_path = f"../models/classifier/train_classifier_{feature}/CelebA/{classifier_epoch}_epoch_classifier.pth"
        if not os.path.exists(model_path):
            print('model doesnt exits')
            exit(1)
        print(f"Using model {os.path.abspath(model_path)}")

        resnet = resnet50(pretrained=False)
        resnet.fc = nn.Linear(2048, 1)
        resnet.load_state_dict(torch.load(model_path))
        resnet.to(device)
        resnet.eval()

        models[feature] = resnet
    print()


    for attack in attacks:
        #model_path = os.path.join("..", "models", , dataset, model_type, attacks[0], defense, str(1), ")

        for i in range(1, 11):

            results = pd.DataFrame(columns=features)
            #image_folder = os.path.join("..", "results", "diff_stylegan_best")
            image_folder = os.path.join("..", "results", batch, dataset_name, model_type, attack, defense, str(i),)
            print(f"Using data {os.path.abspath(image_folder)}")

            results_file = os.path.join(image_folder, f"classification.csv")

            if os.path.exists(results_file):
                print(f"Already classified {attack} {i}")
                continue

            dataset = dset.ImageFolder(image_folder, transform_test)

            if len(dataset) != expected_num_images:
                print(f"{attack} {i} only have {len(dataset)} / {expected_num_images} images")
                continue

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=opt.workers)

            count = torch.FloatTensor(len(features)).fill_(0)
            total = 0
            shown = False
            with torch.no_grad():
                for batch_idx, images in enumerate(dataloader):
                    images = Variable(images[0].to(device))

                    temp_row = pd.DataFrame()

                    for feature, model in models.items():

                        output = model(images)
                        com1 = output > 0

                        row = pd.DataFrame(com1.detach().cpu(), columns=[feature])
                        temp_row = pd.concat([temp_row, row], axis="columns")



                    if batch_idx % 10 == 0:
                        print(f"Analyzed: {batch_idx}/ {len(dataloader)} images")

                    results = pd.concat([results, temp_row])

            results = np.where(results, 1, -1)
            results = pd.DataFrame(results, columns=features)


            results.to_csv(results_file, index=False)


