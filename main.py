import argparse
import glob
import numpy as np
import os
import sys
import random
import torch
import torchvision
import torch.nn as nn
'''from torchvision import transforms, utils
from torch.nn import functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader'''
import glob
import pandas as pd
from sklearn.utils import shuffle
import segmentation_models_pytorch as smp
from PIL import Image, ImageFile
from dataloader import SIIMDataset
from Trainer import Trainer
if __name__=="__main__":
    seed= 12321
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main_path = "../home/mvenkataraman_ph/SIMMI_seg/"
    path_list=glob.glob(main_path + 'data128/train/*.png')
    temp_list=[]
    for l in path_list:
        l=l.split('/')[4]
        temp_list.append(l)
    path_list=temp_list
    root_dir_train=main_path + '/content/data128/train/'
    root_dir_mask= main_path + '/content/data128/masks/'
    new_list=[]
    no_list = []
    for i,path in enumerate(path_list):
        mask=np.array(Image.open(root_dir_mask + path,'r'))
        if (np.sum(mask) > 0):
            new_list.append(path)
        else:
            no_list.append(path)
    new_list = shuffle(new_list , random_state = seed)
    no_list = shuffle(no_list , random_state = seed)
    val_list_mask = new_list[:475]
    val_list_no_mask = no_list[:1667]
    new_list = new_list[475:]
    no_list = no_list[1667:]
    val_list = val_list_mask + val_list_no_mask
    path_list = no_list + new_list + new_list + new_list
    path_list = shuffle(path_list , random_state = seed)

    
    batch_size = 16
    img_size = 128
    transformed_dataset = SIIMDataset(path_list,root_dir_mask,root_dir_train,img_size,validate=False,transform=transforms.Compose([ToTensor()]))
    validation_dataset = SIIMDataset(val_list,root_dir_mask,root_dir_train,img_size,validate=True,transform=transforms.Compose([ToTensor()]))

    train_dataloader = DataLoader(transformed_dataset, batch_size=batch_size,num_workers=5,shuffle=False)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size,num_workers=5,shuffle=False)
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    log_directory = main_path + "runs/prediction"
    
    sol = Trainer(model, train_dataloader , val_dataloader,  use_cuda = True , logdir = log_directory )
    sol.train(70,main_path)






