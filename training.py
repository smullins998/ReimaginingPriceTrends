#!/usr/bin/env python
# coding: utf-8

use_gpu = True
use_ramdon_split = True
use_dataparallel = True

import os
import sys
sys.path.insert(0, '..')

import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)

IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

# Load data for years 1993 to 2000
year_list = np.arange(1993, 2001, 1)

images = []
label_df = []

for year in year_list:
    images.append(
        np.memmap(os.path.join("./img_data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_images.dat"),
                  dtype=np.uint8, mode='r').reshape((-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20])))
    label_df.append(
        pd.read_feather(os.path.join("./img_data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather")))

images = np.concatenate(images)
label_df = pd.concat(label_df)

print("Images Shape:", images.shape)
print("Labels Shape:", label_df.shape)

# Define dataset class
class MyDataset(Dataset):
    def __init__(self, img, label):
        self.img = torch.Tensor(img.copy())
        self.label = torch.Tensor(label)
        self.len = len(img)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

# Split data into training and validation sets
train_indices, val_indices = train_test_split(
    np.arange(images.shape[0]),
    test_size=0.3,
    random_state=42
)

# Create datasets and dataloaders
train_dataset = MyDataset(images[train_indices], (label_df.Ret_5d > 0).values[train_indices])
val_dataset = MyDataset(images[val_indices], (label_df.Ret_5d > 0).values[val_indices])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Define neural network architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(46080, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, 1, 64, 60)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1, 46080)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

# Initialize neural network
net = Net()

# Enable GPU and DataParallel if specified
if use_gpu and use_dataparallel and 'DataParallel' not in str(type(net)):
    net = net.to('cuda')
    net = nn.DataParallel(net)

# Define loss function, optimizer, and other training parameters
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
start_epoch = 0
min_val_loss = 1e9
last_min_ind = -1
early_stopping_epoch = 5
tb = SummaryWriter()

# Training loop
start_time = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
epochs = 100

for t in range(start_epoch, epochs):
    print(f"Epoch {t}\n-------------------------------")
    time.sleep(0.2)
    train_loss = train_loop(train_dataloader, net, loss_fn, optimizer)
    val_loss = val_loop(val_dataloader, net, loss_fn)
    tb.add_histogram("train_loss", train_loss, t)
    torch.save(net, './CNN' + os.sep + 'baseline_epoch_{}_train_{:5f}_val_{:5f}.pt'.format(t, train_loss, val_loss))
    if val_loss < min_val_loss:
        last_min_ind = t
        min_val_loss = val_loss
    elif t - last_min_ind >= early_stopping_epoch:
        break

print('Done!')
print('Best epoch: {}, val_loss: {}'.format(last_min_ind, min_val_loss))
