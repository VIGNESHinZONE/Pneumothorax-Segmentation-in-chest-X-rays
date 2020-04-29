import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import sys
import random
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, utils
from torch.nn import functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        masks = masks.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'masks': torch.from_numpy(masks)} 

def plot_image(first , second ):
  fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(12,9))

  ax[0].axis("off")
  ax[1].axis("off")
  ax[0].title.set_text('True Mask')
  ax[1].title.set_text('Predicted Mask')

  image1 , first_mask = first['image'] , first['masks']
  image2 , second_mask = second['image'] , second['masks']


  ax[0].imshow(np.transpose(utils.make_grid (image1[:5], padding=1, normalize=True).cpu(),(1,2,0)) ,'CMRmap' )
  ax[0].imshow(np.transpose(utils.make_grid ( first_mask[:5], padding=1, normalize=False ).cpu(), (1,2,0) ) , alpha = 0.5 )
  ax[1].imshow(np.transpose(utils.make_grid (image2[:5], padding=1, normalize=True).cpu(),(1,2,0)) ,'CMRmap' )
  ax[1].imshow(np.transpose(utils.make_grid ( second_mask[:5], padding=1, normalize=False ).cpu(), (1,2,0) ) , alpha = 0.5 )
  fig.tight_layout()
  return fig

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


def iou(outputs,labels):
    s = 1e-6
    outputs = outputs > 0.3
    labels = labels ==1
    iou=(((labels & outputs).float().sum((1,2,3))+s)/((labels | outputs).float().sum((1,2,3))+s))
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    return iou.mean().item()