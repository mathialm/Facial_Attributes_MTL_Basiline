import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn import metrics

from dataset.CelebA import CelebA
from model.resnet import resnet50
import os
from torch.autograd import Variable
import argparse

import torchvision.datasets as dset
import torch.nn.functional as F
from train import MODEL_SIZE, IMAGE_SIZE

#THRESHOLDS = [0, 0, 0, 0]

def main():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    transform_test = transforms.Compose([
        # transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE) / 2)),
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

    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    # testset = CelebA('./data/list_eval_partition.txt', './data/list_attr_celeba.txt', '2',
    #                  './data/img_align_celeba/', transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

    features = ["Mouth_Slightly_Open", "Wearing_Lipstick", "High_Cheekbones", "Male"]

    BASE = os.path.join("/", "cluster", "home", "mathialm", "poisoning", "ML_Poisoning")

    image_folder = os.path.join(BASE, "data", "datasets64", "clean", "celeba")

    pred_save_file = os.path.join(image_folder, "classified_test_preds.npz")
    if not os.path.exists(pred_save_file):

        print(f"Using data {os.path.abspath(image_folder)}")

        dataset = dset.ImageFolder(image_folder, transform_test)

        testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False,
                                                 num_workers=opt.workers)

        models = {}
        for feature in features:
            model_path = os.path.join(BASE, "models", "classifier", f"train_classifier_{feature}", "CelebA",
                                      "10_epoch_classifier.pth")
            if not os.path.exists(model_path):
                print(f'model {model_path} doesnt exist')
                exit(1)
            print(f"Using model {os.path.abspath(model_path)}")

            resnet = resnet50(pretrained=False)
            resnet.fc = nn.Linear(2048, 1)
            resnet.load_state_dict(torch.load(model_path))
            resnet.to(device)
            resnet.eval()

            models[feature] = resnet

        attribute_file = os.path.join(image_folder, "list_attr_celeba.txt")
        attrs = pd.read_csv(attribute_file, header=0, index_col=0)
        gt = attrs[features].to_numpy()

        predss = None
        with torch.no_grad():
            for batch_idx, images in enumerate(testloader):
                images = Variable(images[0].to(device))
                feat_preds = None
                for feature_index, feature in enumerate(features):
                    resnet = models[feature]

                    preds = resnet(images)

                    #pred_dict[feature] = torch.cat((pred_dict[feature], preds.data), 0)
                    preds_numpy = preds.data.cpu().numpy()
                    if feat_preds is None:
                        feat_preds = preds_numpy
                    else:
                        feat_preds = np.concatenate((feat_preds, preds_numpy), 1)
                    #torch.cat((feat_preds, preds.data), 0)
                #predss = torch.cat((predss, feat_preds), 1)
                if predss is None:
                    predss = feat_preds
                else:
                    predss = np.concatenate((predss, feat_preds), 0)


                if batch_idx % 10 == 0:
                    print(f"Classifying batch {batch_idx + 1}/{len(testloader)}")

        pred = predss


        np.savez(pred_save_file, pred=pred, gt=gt)
    else:
        npz_file = np.load(pred_save_file)
        pred = npz_file["pred"]
        gt = npz_file["gt"]

    stats_file = os.path.join(image_folder, "stats.csv")




    #First normalize predictions
    pred = (pred - np.min(pred))/np.ptp(pred)

    gt = (gt + 1) / 2
    gt = gt.astype(int)

    print(f"{pred.shape = }")
    print(f"{gt.shape = }")

    roc_auc = metrics.roc_auc_score(gt, pred, average=None)
    print(f"{roc_auc = }")

    avg_prec = metrics.average_precision_score(gt, pred, average=None)
    print(f"{avg_prec = }")

    res_by_thres = pd.DataFrame(
        columns=["threshold", "feature", "recall", "precision", "accuracy", "balanced_acc", "F1_score", "TP", "TN", "FP", "FN"])
    MAX_VAL = 100
    for i in range(1, MAX_VAL):
        threshold = i / MAX_VAL
        thresholds = [threshold] * len(features)
        pred_labels = pred > thresholds
        # print(pred_labels)

        true_labels = gt

        correct_labels = true_labels == pred_labels

        # print(correct_labels)

        TP = np.sum(correct_labels * true_labels, axis=0)
        TN = np.sum(correct_labels * (1 - true_labels), axis=0)
        FN = np.sum((1 - correct_labels) * true_labels, axis=0)
        FP = np.sum((1 - correct_labels) * (1 - true_labels), axis=0)

        # print(f"{TP = }")
        # print(f"{TN = }")
        # print(f"{FP = }")
        # print(f"{FN = }")

        precision = TP / (TP + FP)

        recall = TP / (TP + FN)

        # print(f"{precision = }")
        # print(f"{recall = }")

        accuracies = np.sum(correct_labels, axis=0) / len(correct_labels)
        # print(accuracies)
        for feat_index, feature in enumerate(features):
            # feature = CLASS_NAMES[feat_index]
            rec = recall[feat_index]
            prec = precision[feat_index]
            acc = accuracies[feat_index]
            tl = true_labels[:, feat_index]
            pl = pred_labels[:, feat_index]
            bal_acc = metrics.balanced_accuracy_score(tl, pl)
            f1_score = metrics.f1_score(tl, pl)
            df = pd.DataFrame(data={"threshold": [threshold],
                                    "feature": [feature],
                                    "recall": [rec],
                                    "precision": [prec],
                                    "accuracy": [acc],
                                    "balanced_acc": bal_acc,
                                    "F1_score": f1_score,
                                    "TP": TP[feat_index],
                                    "TN": TN[feat_index],
                                    "FP": FP[feat_index],
                                    "FN": FN[feat_index]})

            res_by_thres = pd.concat((res_by_thres, df), ignore_index=True)
        print(f"Calculated for threshold {threshold}")
    res_by_thres.to_csv(stats_file, header=True, index=False)

    print(res_by_thres)

if __name__ == "__main__":
    main()