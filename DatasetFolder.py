from __future__ import division
import numpy as np
import os
import pandas as pd
import argparse
import sys
import random
from matplotlib.pyplot import imsave, imread
import matplotlib
import torch
from PIL import Image
import PIL.ImageOps 
import cv2
matplotlib.use("Agg")
import torchvision.datasets as datasets
from skimage.transform import resize
from torchvision.datasets.folder import default_loader

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images

def random_rotation(x, chance, with_fa=False):
    ran = random.random()
    if with_fa:
        img = Image.fromarray(x[0])
        mask = Image.fromarray(x[1])
        if ran > 1 - chance:
            # create black edges
            angle = np.random.randint(0, 90)
            img = img.rotate(angle=angle, expand=1)
            mask = mask.rotate(angle=angle, expand=1, fillcolor=1)
            return np.stack([np.asarray(img), np.asarray(mask)])
        else:
            return np.stack([np.asarray(img), np.asarray(mask)])
    img = Image.fromarray(x)
    if ran > 1 - chance:
        # create black edges
        angle = np.random.randint(0, 90)
        img = img.rotate(angle=angle, expand=1)
        return np.asarray(img)
    else:
        return np.asarray(img)

class ImageMaskFolder(datasets.ImageFolder):
    def __init__(self, root, mask_folder, transform= None, target_transform= None, is_valid_file= None, loader=default_loader):
       super().__init__(root, transform, target_transform,loader,is_valid_file)
       self.mask_folder = mask_folder


    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]

        img_id = path.split("/")[-1]
        if os.path.isfile(self.mask_folder+img_id) and label == 1:
            r = Image.open(self.mask_folder+img_id)
            binary_mask = np.asarray(r) < 127
            mask = torch.where(torch.from_numpy(binary_mask) == True, 1, 100)
            mask = mask.unsqueeze(0).float()
            mask = torch.nn.AdaptiveMaxPool2d((7,7))(mask)
        else:
            mask = torch.zeros(1,7,7,dtype=torch.float)
        return (img, label, mask)
    
class ImageMaskReferenceFolder(datasets.ImageFolder):
    def __init__(self, root, mask_folder, transform= None, target_transform= None, is_valid_file= None, loader=default_loader):
       super().__init__(root, transform, target_transform,loader,is_valid_file)
       self.mask_folder = mask_folder


    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]

        img_id = path.split("\\")[-1]
        if os.path.isfile(self.mask_folder+img_id):
            r,g,b = Image.open(self.mask_folder+img_id).split()
            binary_mask = np.asarray(r) < 127
            mask = torch.where(torch.from_numpy(binary_mask) == True, 1, 100)
            mask = mask.unsqueeze(0).float()
            mask = torch.nn.AdaptiveMaxPool2d((7,7))(mask)
        else:
            mask = torch.zeros(1,7,7,dtype=torch.float)
        return (img, label, mask, path)