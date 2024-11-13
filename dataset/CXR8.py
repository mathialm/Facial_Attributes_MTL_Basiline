import pandas as pd
from PIL import Image
import torch.utils.data as data
import numpy as np
import os


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
    #img = pd.Series(data=img, name="filename")
    return img

class CXR8(data.Dataset):
    def __init__(self, part_dir, attr_dir, partition, img_dir, transform, weighted_attrs=None, seed=1):
        self.img = make_img(part_dir, partition)

        self.attr = pd.read_csv(attr_dir, index_col=0)
        #print(f"All attributes: {self.attr.shape}")
        self.attr = self.attr.loc[self.img, :]
        #print(f"Only partition: {self.attr.shape}")

        if weighted_attrs is not None:
            tmp_attrs = self.attr.copy()
            #Get number of attribute = 1 and = 0
            attrs_counts = tmp_attrs.value_counts(subset=weighted_attrs, normalize=False, sort=True, ascending=True)
            #print(f"Values count before: {attrs_counts}")

            #Sample the highest number of samples
            attrs_neg = tmp_attrs.loc[tmp_attrs[weighted_attr] == 0]
            attrs_pos = tmp_attrs.loc[tmp_attrs[weighted_attr] == 1]
            #Upscale whichever is smallest, always at least use the full set, then upscale

            #At least one feature is only 0s (for some reason)
            if len(attrs_counts) == 2:
                if attrs_counts[0] < attrs_counts[1]:
                    attrs_neg = pd.concat((attrs_neg, attrs_neg.sample(n=attrs_counts[1] - attrs_counts[0], replace=True, random_state=seed)),
                                          ignore_index=True)
                else:
                    attrs_pos = pd.concat((attrs_pos, attrs_pos.sample(n=attrs_counts[0] - attrs_counts[1], replace=True, random_state=seed)),
                                          ignore_index=True)
            tmp_attrs = pd.concat((attrs_neg, attrs_pos), ignore_index=True)
            self.img = tmp_attrs["filename"].to_list()


        #print(f"Upscaled attrs: {self.attr.shape}")

        #print(f"Values count after: {self.attr.value_counts(subset=weighted_attr, normalize=False, sort=True, ascending=True)}")
        self.attr = self.attr.drop(columns=['filename'])

        self.length = len(self.img)
        self.transform = transform
        self.img_dir = img_dir

    def __getitem__(self, index):
        image = pil_loader(os.path.join(self.img_dir, self.img[index]))
        if self.transform is not None:
            image = self.transform(image)
        img_path = self.img[index]

        item_attrs = self.attr.loc[img_path, self.attr.columns != "filename"].to_numpy()
        #print(item_attrs)
        #print(f"{index} | {image.shape} | {item_attrs}")
        return image, item_attrs


    def __len__(self):
        return self.length
