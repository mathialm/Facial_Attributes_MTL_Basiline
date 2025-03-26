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

class COCO(data.Dataset):
    def __init__(self, attr_dir, img_dir, transform, weighted_attr=None, seed=1):
        #Only files that are .png
        self.attr = pd.read_csv(attr_dir, header=0, index_col=0)
        try:
            int(self.attr.index.to_list()[0])
            indexes = self.attr.index.to_series()
            indexes = indexes.apply(lambda row: f"{row:012d}.png", convert_dtype=True)
            self.attr.index = indexes
            self.attr.to_csv(attr_dir, index=True)
            print()
        except ValueError as e:
            print("Attributes correctly loaded")

        #print(f"All attributes: {self.attr.index}")

        print("Setting up dataset")
        self.img = []
        for f in tqdm(listdir(img_dir)):
            full_file = os.path.join(img_dir, f)
            if os.path.exists(full_file) and os.path.isfile(full_file) and pathlib.Path(full_file).suffix == ".png":
                #.png are stored with leading zeroes, so remove this by converting to int
                self.img.append(f)

        print(f"Found {len(self.img)} images in {img_dir}")

        self.attr = self.attr.loc[self.attr.index.intersection(self.img), :]
        print(f"Attrs file {self.attr = }")
        #print(f"Only partition: {self.attr.shape}")
        #print(f"Attrs before {self.attr}")

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