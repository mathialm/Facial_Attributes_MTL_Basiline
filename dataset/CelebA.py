import glob
import pathlib
import sys
from os import listdir

import pandas as pd
from PIL import Image
import torch.utils.data as data
import numpy as np
import os

from tqdm import tqdm


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def make_img(part_dir, partition):
    img = []
    with open(part_dir) as f:
        lines = f.readlines()
        for line in lines:
            pic_dir, num = line.strip().split(",")
            if num == partition:
                img.append(pic_dir)
    return img

class CelebA(data.Dataset):
    def __init__(self, attr_file, img_dir, transform, weighted_attr=None, seed=1):
        self.attr = pd.read_csv(attr_file, header=0, index_col=0)

        df_min = self.attr.min().min()
        df_max = self.attr.max().max()
        print(f"{df_min = } | {df_max = }")
        if df_min != 0 or df_max != 1:
            diff = df_max - df_min
            self.attr: pd.DataFrame = (self.attr + 2) / diff
            self.attr = self.attr.astype(int)
            assert self.attr.min().min() == 0 and self.attr.max().max() == 1

        self.img = glob.glob(os.path.join(img_dir, "*.jpg"))
        self.img = [os.path.basename(file) for file in self.img]

        intersection_indexes = self.attr.index.intersection(self.img)
        self.attr = self.attr.loc[intersection_indexes]

        """
        for f in tqdm(listdir(img_dir)):
            full_file = os.path.join(img_dir, f)
            if os.path.exists(full_file) and os.path.isfile(full_file) and pathlib.Path(full_file).suffix == ".png":
                self.img.append(f)
        """

        if weighted_attr is not None:
            tmp_attrs = self.attr.copy()
            #Get number of attribute = 1 and = 0
            attrs_counts = tmp_attrs.value_counts(subset=weighted_attr, normalize=False, sort=True, ascending=True)
            #print(f"Values count before: {attrs_counts}")

            #Sample the highest number of samples
            attrs_neg = tmp_attrs.loc[tmp_attrs[weighted_attr] == 0]
            attrs_pos = tmp_attrs.loc[tmp_attrs[weighted_attr] == 1]
            #Upscale whichever is smallest, always at least use the full set, then upscale

            #At least one feature is only 0s (for some reason)
            print("\nBefore weighting")
            print(f"{attrs_pos.shape = }")
            print(f"{attrs_neg.shape = }")
            if len(attrs_counts) == 2:
                if attrs_counts[0] < attrs_counts[1]:
                    attrs_neg = pd.concat((attrs_neg, attrs_neg.sample(n=attrs_counts[1] - attrs_counts[0], replace=True, random_state=seed)),
                                          ignore_index=False)
                else:
                    attrs_pos = pd.concat((attrs_pos, attrs_pos.sample(n=attrs_counts[0] - attrs_counts[1], replace=True, random_state=seed)),
                                          ignore_index=False)

            print("\nAfter weighting")
            print(f"{attrs_pos.shape = }")
            print(f"{attrs_neg.shape = }")
            tmp_attrs = pd.concat((attrs_neg, attrs_pos), ignore_index=False)
            self.img = tmp_attrs.index.to_list()
        else:
            self.img = self.attr.index.to_list()
        self.length = len(self.img)
        self.transform = transform
        self.img_dir = img_dir

    def __getitem__(self, index):
        image = pil_loader(os.path.join(self.img_dir, self.img[index]))
        if self.transform is not None:
            image = self.transform(image)

        img_path = self.img[index]

        item_attrs = self.attr.loc[img_path, ].to_numpy()

        return image, item_attrs


    def __len__(self):
        return self.length

    def get_features_indexes(self, features):
        return self.attr.columns.get_indexer(features)