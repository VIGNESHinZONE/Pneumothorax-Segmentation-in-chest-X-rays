#LIBRARIES NEEDED
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import transforms, utils
from torch.nn import functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils import plot_image,MixedLoss,iou


class Trainer():
    def __init__(self,model, Dataloader , Dataloader_val, lr = 5e-4, use_cuda = True , logdir = "None" ):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.Dataloader = Dataloader
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)
        self.Dataloader_val = Dataloader_val
        self.criterion = MixedLoss(10.0, 2.0)
        self.use_cuda = use_cuda
        self.device = 'cpu'
        self.writer = SummaryWriter(logdir)
        self.total_epochs = 0
        self.steps = 0
        if self.use_cuda:
          self.model.cuda()
          self.device = 'cuda'

    def train(self, n_epochs , main_path):
        for epoch in range(1,n_epochs + 1):
            print('Starting epoch {}...'.format(epoch) )
            #print('Time - ',datetime.now())
            self.train_epoch(epoch,n_epochs)
            
            self.total_epochs += 1
            if epoch % 5 == 0 or epoch == n_epochs:
              torch.save(self.model.state_dict(), main_path+ 'results/' +'Segmentation'+'model{}.pt'.format(self.total_epochs))

    def train_epoch(self,epoch,n_epochs):
        for i, (_,sample) in enumerate(self.Dataloader):
            self.model.zero_grad()
            self.model.train()
            inputs = sample['image']
            labels = sample['masks']
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.model(inputs.float())
            loss= self.criterion(outputs.float(), labels)
            loss.backward()
            self.optimizer.step()
            train_accuracy = iou(torch.sigmoid(outputs),labels)
            
            

            if i % 150 == 0 and i!=0 :
              print('[%d/%d][%d/%d]\tLoss: %.4f\tAccuracy: %.4f '% (epoch, n_epochs, i, len(self.Dataloader),loss.item(), train_accuracy))

              self.writer.add_scalar('training loss ',loss.item(),self.steps)
              self.writer.add_scalar('training accuracy ',train_accuracy,self.steps)
              self.validate(epoch,n_epochs)
              self.steps+=1
              
    def validate(self,epoch,n_epochs):
            model.eval()
            loss_value = 0.0
            Accuracy = 0.0
            for i,(_,sample) in enumerate(self.Dataloader_val):
              inputs = sample['image']
              labels = sample['masks']
              inputs = inputs.cuda()
              labels = labels.cuda()
              outputs = self.model(inputs.float())
              loss= self.criterion(outputs.float(), labels)
              loss_value += loss.item()
              Accuracy += iou(torch.sigmoid(outputs),labels)
              if i == 0:
                  preds = {'image': inputs.detach().cpu(),'masks': (torch.sigmoid(outputs) > 0.3).float().detach().cpu()}
                  self.writer.add_figure('Visulations ',plot_image(sample , preds ),self.steps)

            loss_value/= len(self.Dataloader_val)
            Accuracy/= len(self.Dataloader_val)
            print('[%d/%d][%d/%d]\tVal Loss: %.4f\tVal Accuracy: %.4f '% (epoch, n_epochs, i, len(self.Dataloader_val),loss_value, Accuracy))
            self.writer.add_scalar('Validation loss ',loss_value,self.steps)
            self.writer.add_scalar('Validation accuracy ',Accuracy,self.steps)
            
