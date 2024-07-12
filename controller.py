import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from data_factory.data_loader import get_loader_segment
from utils import load_prototype_features
from model.prvae import PRVAE

from losses.focal_loss import FocalLoss
from losses.smooth_l1_loss import SmoothL1Loss
from losses.entropy_loss import EntropyLoss

import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

class Controller(object):
    DEFAULTS={}
    def __init__(self,config):

        self.__dict__.update(Controller.DEFAULTS,**config)

        self.train_loader, self.vali_loader, self.k_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, 
                                                                                mode="train", dataset=self.dataset)
        self.test_loader, _ = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = self.vali_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = "resnet18"

        self.build_model()

        self.smooth_l1_loss = SmoothL1Loss()
        self.focal_loss = FocalLoss(alpha=0.5,gamma=4)
        self.entropy_loss = EntropyLoss()
        self.criterion = nn.MSELoss()


    
    def build_model(self):
        input_size = (self.win_size,self.input_c)
        self.model = PRVAE(self.backbone, num_classes=1, input_size=input_size,device=self.device).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model,device_ids=[0,1,2,3],output_device=0).to(self.device)

    def train(self):
        print("======================TRAIN MODE======================")
        proto_features = load_prototype_features('prototypes', self.dataset, self.device)
        train_steps = len(self.train_loader)
        best_loss = []
        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []
            rec_loss_list = []; entropy_loss_list = []

            epoch_time = time.time()
            self.model.train()
            for step,(input_data,labels) in enumerate(self.train_loader):

                input = input_data
                input = np.expand_dims(input, axis=-1)
                input = np.repeat(input, repeats=3, axis=-1)
                input = np.transpose(input, (0, 3, 1, 2))
                input = input.to(self.device)

                output = self.model(input,proto_features)
                output = output.permute(0, 2, 3, 1)
                output = torch.mean(output, dim=-1, keepdim=True)
                output = output.squeeze(-1)
                
                input_data = input_data.to(self.device)
                rec_loss = self.criterion(output, input_data)
                '''entropy_loss = self.entropy_loss(attn)'''
                loss = rec_loss

                loss_list.append(loss.detach().cpu().numpy())
                '''entropy_loss_list.append(entropy_loss.detach().cpu().numpy())'''
                rec_loss_list.append(rec_loss.detach().cpu().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(loss_list)
            train_loss = np.abs(train_loss)
            if(len(best_loss)==0):
                best_loss.append(train_loss)
                torch.save(self.model.state_dict(), os.path.join('trained_models', str(self.dataset) + f'_checkpoint_gpu.pth'))
            elif best_loss[0] > train_loss:
                best_loss.remove(0)
                best_loss.append(train_loss)
                torch.save(self.model.state_dict(), os.path.join('trained_models', str(self.dataset) + f'_checkpoint_gpu.pth'))
            train_entropy_loss = np.average(entropy_loss_list)
            train_rec_loss = np.average(rec_loss_list)
            '''valid_loss , valid_re_loss_list, valid_entropy_loss_list = self.vali(self.vali_loader)'''

            '''print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss))'''
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
            '''print(
                "Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                    epoch + 1, train_steps, valid_re_loss_list, valid_entropy_loss_list))
            print(
                "Epoch: {0}, Steps: {1} | TRAIN reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                    epoch + 1, train_steps, train_rec_loss, train_entropy_loss))'''
    
    def vali(self,vali_loader):
        self.model.eval()

        valid_loss_list = [] ; valid_re_loss_list = [] ; valid_entropy_loss_list = []

        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)
            output, attn = output_dict['out'], output_dict['attn']
            
            rec_loss = self.criterion(output, input)
            entropy_loss = self.entropy_loss(attn)
            loss = rec_loss + self.lambd*entropy_loss

            valid_re_loss_list.append(rec_loss.detach().cpu().numpy())
            valid_entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
            valid_loss_list.append(loss.detach().cpu().numpy())

        return np.average(valid_loss_list), np.average(valid_re_loss_list), np.average(valid_entropy_loss_list)

                