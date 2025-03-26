import pathlib
import sys
from os import listdir
import re

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

FORMATTERS = {
    "12_num_png_to_int": lambda str_in: int(re.findall("([0-9]+)\.png$", str_in)[0]),
    "num_to_12_num_str": lambda int_in: f"{int_in:012d}.png"
}

class COCO(data.Dataset):
    def __init__(self, attr_file, img_dir, transform, weighted_attr=None, seed=1):
        #Only files that are .png
        self.attr = pd.read_csv(attr_file, header=0, index_col=0)
        try:
            int(self.attr.index.to_list()[0])
            indexes = self.attr.index.to_series()
            indexes = indexes.apply(lambda row: f"{row:012d}.png", convert_dtype=True)
            self.attr.index = indexes
            self.attr.to_csv(attr_file, index=True)
            print()
        except ValueError as e:
            print("Attributes correctly loaded")

        #print(f"All attributes: {self.attr.index}")

        print("Setting up dataset")
        self.img = []
        for f in tqdm(listdir(img_dir)):
            full_file = os.path.join(img_dir, f)
            if os.path.exists(full_file) and os.path.isfile(full_file) and pathlib.Path(full_file).suffix == ".png":
                self.img.append(f)

        print(f"Found {len(self.img)} images in {img_dir}")

        print(f"{self.attr.shape = }")

        attrs_indexes = self.attr.index
        print(f"{attrs_indexes = }")

        to_num_formatter = FORMATTERS["12_num_png_to_int"]
        to_str_formatter = FORMATTERS["num_to_12_num_str"]

        attrs_index_num = self.attr.index.format(formatter=to_num_formatter)

        # Reformat indexes in attributes to correspond to predictions (multiple images with same nr.)
        new_attrs = np.empty(shape=(len(self.img), self.attr.shape[1]))
        new_attrs_indexes = []  # Just in case some indexes are not in attrs
        attrs_expand_tqdm = tqdm(enumerate(self.img), total=len(self.img))
        for i, idx in attrs_expand_tqdm:
            idx_num = to_num_formatter(idx)
            if idx in self.attr.index: #Normal mode
                index = idx
            elif idx_num in attrs_index_num: #Convert and compare pure number
                index = to_str_formatter(idx_num)
            else:
                print(f"Did not find {idx} in gt.index")
                continue
            attrs_row = self.attr.loc[index].to_numpy()
            assert len(attrs_row) == new_attrs.shape[1]

            new_attrs[i, ] = attrs_row
            new_attrs_indexes.append(idx)  # Keep original ID to properly separate samples

        self.attr = pd.DataFrame(data=new_attrs, columns=self.attr.columns, index=new_attrs_indexes)

        print(f"{self.attr.shape = }")

        assert self.attr.shape[0] == len(self.img)

        #Make change when filename and attr file index does not match
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
