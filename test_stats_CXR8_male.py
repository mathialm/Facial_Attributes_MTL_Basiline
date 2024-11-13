import numpy as np
import pandas as pd

from dataset.CelebA import CelebA
from model.resnet import resnet50
import os

import argparse

from train import MODEL_SIZE, IMAGE_SIZE

import sklearn.metrics as metrics

FEATURES = {"CXR8": ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                     'Mass', 'Nodule', 'Pneumonia',
                     'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
                     'Fibrosis', 'Pleural_Thickening', 'Hernia', "male"],
               "celeba": ["Mouth_Slightly_Open", "Wearing_Lipstick", "High_Cheekbones", "Male"]}

FEAT_ROC = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                     'Mass', 'Nodule', 'Pneumonia',
                     'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
                     'Fibrosis', 'Hernia', "male"]

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model', type=str)
    opt = parser.parse_args()

    BASE = os.path.join("/", "cluster", "home", "mathialm", "poisoning", "ML_Poisoning")

    image_folder = os.path.join(BASE, "data", "datasets128", "clean", "CXR8")

    pred_save_file = os.path.join(image_folder, "preds_MTL.csv")

    attribute_file = os.path.join(image_folder, "image_attributes.csv")



    attrs = pd.read_csv(attribute_file, header=0, index_col=0)
    print(attrs.columns)
    features = attrs.columns.intersection(FEATURES["CXR8"])
    print(features)
    print(FEATURES["CXR8"])

    gt = attrs[features].to_numpy()
    gt_roc = attrs[FEAT_ROC].to_numpy()

    if not os.path.exists(pred_save_file):
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        import torchvision.datasets as dset
        from torch.autograd import Variable

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        transform_test = transforms.Compose([
            # transforms.Pad(int((MODEL_SIZE - IMAGE_SIZE) / 2)),
            transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        print(f"Using data {os.path.abspath(image_folder)}")

        dataset = dset.ImageFolder(image_folder, transform_test)

        testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False,
                                                 num_workers=opt.workers)

        models = {}
        for feature in features:
            model_path = os.path.join(BASE, "models", "classifier_CXR8", f"train_classifier_{feature}",
                                      "10_epoch_classifier.pth")
            if not os.path.exists(model_path):
                print(f'model {model_path} doesnt exist')
                continue
            print(f"Using model {os.path.abspath(model_path)}")

            resnet = resnet50(pretrained=False)
            resnet.fc = nn.Linear(2048, 1)
            resnet.load_state_dict(torch.load(model_path))
            resnet.to(device)
            resnet.eval()

            models[feature] = resnet



        predss = None
        with torch.no_grad():
            for batch_idx, images in enumerate(testloader):
                images = Variable(images[0].to(device))
                feat_preds = None
                for feature_index, feature in enumerate(features):
                    if feature not in models:
                        continue
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

        pred = pd.DataFrame(data=predss, columns=features)
        pred.to_csv(pred_save_file)

        #np.savez(pred_save_file, pred=pred, gt=gt)
    else:
        pred = pd.read_csv(pred_save_file, header=0, index_col=None)

    pred_roc = pred.copy()
    print(pred_roc.columns)
    pred_roc = pred_roc[FEAT_ROC]
    print(pred_roc.columns)
    pred_roc = pred_roc.to_numpy()
    print(pred.columns)
    pred = pred[features]
    print(pred.columns)
    pred = pred.to_numpy()

    stats_file = os.path.join(image_folder, "stats_MTL.csv")



    #First normalize predictions
    pred = (pred - np.min(pred))/np.ptp(pred)
    #print(pred.shape)

    gt = (gt + 1) / 2
    gt = gt.astype(int)

    print(gt.shape)
    print(pred.shape)
    roc_auc = metrics.roc_auc_score(gt_roc, pred_roc, average=None)
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

        precision = TP / (TP + FP)

        recall = TP / (TP + FN)

        accuracies = np.sum(correct_labels, axis=0) / len(correct_labels)
        # print(accuracies)
        for feat_index, feature in enumerate(features):
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