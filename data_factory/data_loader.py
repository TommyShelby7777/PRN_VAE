import torch
import os
import random
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(data_path,"smd_train_merged.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(data_path,"smd_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = np.load(os.path.join(data_path,"smd_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if(self.mode == "train"):
            return (self.train.shape[0]-self.win_size)//self.step+1
        elif (self.mode == "test"):
            return (self.test.shape[0]-self.win_size)//self.step+1
        else:
            return (self.train.shape[0]-self.win_size)//self.step+1
        
    def __getitem__(self, index):

        index = index*self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]),np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "test":
            return np.float32(self.test[index:index + self.win_size]),np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]),np.float32(self.test_labels[0:self.win_size])

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode="train", dataset="SMD", val_ratio=0.1):

    if (dataset == "SMD"):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    
    shuffle = False
    if mode=="train":
        shuffle = True
        dataset_len = int(len(dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))

        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)


        indices = torch.arange(dataset_len)
        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(dataset, val_sub_indices)

        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        k_use_len = int(train_use_len*0.1)
        k_sub_indices = indices[:k_use_len]
        k_subset = Subset(dataset, k_sub_indices)
        k_loader = DataLoader(dataset=k_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        return train_loader, val_loader, k_loader
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    
    return data_loader, data_loader