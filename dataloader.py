import numpy as np
import torchvision.transforms as transforms
import os
import sys
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, utils
from torch.nn import functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
from PIL import Image, ImageFile
#Albumenations
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,Normalize,RandomContrast,RandomGamma,ElasticTransform,RandomSizedCrop ,RandomBrightness
)
from utils import ToTensor

class SIIMDataset(Dataset):
  def __init__(self,path_file,root_dir_mask,root_dir_train,img_size,validate=False,transform=None):

    self.path_file=path_file
    self.transform=transform
    self.root_dir_mask=root_dir_mask
    self.root_dir_train=root_dir_train
    self.img_size=img_size
    self.validate = validate
    AUGMENTATIONS_VALIDATE = Compose([    
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1)
    ],p=1)
    AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    RandomSizedCrop(min_max_height=(75, 128), height=128, width=128,p=1),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1)
    ],p=1)
    if self.validate:
      self.augmentation = AUGMENTATIONS_VALIDATE
    else:
      self.augmentation = AUGMENTATIONS_TRAIN



  def __len__(self):
    return len(self.path_file)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    image=Image.open(self.root_dir_train+self.path_file[idx],'r').convert("RGB")
    mask=Image.open(self.root_dir_mask+self.path_file[idx],'r')
    iwidth,iheight= image.size
    mwidth,mheight= mask.size
    if (iwidth!= self.img_size or iheight!=self.img_size ):
      image=image.resize((self.img_size,self.img_size))

    if (mwidth!= self.img_size or mheight!=self.img_size ):
      mask=mask.resize((self.img_size,self.img_size))

    image=np.array(image)
    mask=np.expand_dims(np.array(mask),axis=2)/255
    true_sample = {'image':image , 'masks':mask}
    ans = self.augmentation(image=image, mask=mask)
    image = np.array(ans['image'])
    mask = np.array(ans['mask'])
    sample = {'image': image, 'masks': mask}

    if self.transform:
      sample = self.transform(sample)

    return true_sample , sample
