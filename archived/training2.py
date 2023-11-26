import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from model import Net
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import torch.optim as optim
from tqdm import tqdm



IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

year_list = np.arange(1993, 2001, 1)

images = []
label_df = []
for year in year_list:
    images.append(
        np.memmap(os.path.join("../img_data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8,
                  mode='r').reshape(
            (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20])))
    label_df.append(pd.read_feather(
        os.path.join("../img_data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather")))

images = np.concatenate(images)
label_df = pd.concat(label_df)

print(images.shape)
print(label_df.shape)

device = 'cuda'

class MyDataset(Dataset):

    def __init__(self, img, label):
        self.img = torch.Tensor(img.copy())
        self.label = torch.Tensor(label)
        self.len = len(img)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]


# Use 70%/30% ratio for train/validation split
train_indices, val_indices = train_test_split(
    np.arange(images.shape[0]),
    test_size=0.3,
    random_state=42
)

# Create datasets based on the selected indices
train_dataset = MyDataset(images[train_indices], (label_df.Ret_5d > 0).values[train_indices])
val_dataset = MyDataset(images[val_indices], (label_df.Ret_5d > 0).values[val_indices])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)


class CNN20d(nn.Module):
    # Input: [N, (1), 64, 60]; Output: [N, 2]
    # Three Convolution Blocks

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self):
        super(CNN20d, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(3, 1), stride=(3, 1), dilation=(2, 1))),
            # output size: [N, 64, 21, 60]
            ('BN', nn.BatchNorm2d(64, affine=True)),
            ('ReLU', nn.ReLU()),
            ('Max-Pool', nn.MaxPool2d((2, 1)))  # output size: [N, 64, 10, 60]
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)

        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(3, 1), stride=(1, 1), dilation=(1, 1))),
            # output size: [N, 128, 12, 60]
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('ReLU', nn.ReLU()),
            ('Max-Pool', nn.MaxPool2d((2, 1)))  # output size: [N, 128, 6, 60]
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)

        self.conv3 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(128, 256, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))),
            # output size: [N, 256, 6, 60]
            ('BN', nn.BatchNorm2d(256, affine=True)),
            ('ReLU', nn.ReLU()),
            ('Max-Pool', nn.MaxPool2d((2, 1)))  # output size: [N, 256, 3, 60]
        ]))
        self.conv3 = self.conv3.apply(self.init_weights)

        self.DropOut = nn.Dropout(p=0.5)
        self.FC = nn.Linear(46080, 2)
        self.init_weights(self.FC)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):  # input: [N, 64, 60]
        x = x.unsqueeze(1).to(torch.float32)  # output size: [N, 1, 64, 60]
        x = self.conv1(x)  # output size: [N, 64, 10, 60]
        x = self.conv2(x)  # output size: [N, 128, 6, 60]
        x = self.conv3(x)  # output size: [N, 256, 3, 60]
        x = self.DropOut(x.view(x.shape[0], -1))
        x = self.FC(x)  # output size: [N, 2]
        x = self.Softmax(x)

        return x




def train_n_epochs(n_epochs, model, label_type, train_loader, valid_loader, criterion, optimizer, savefile, early_stop_epoch):
    valid_loss_min = np.Inf  # track change in validation loss
    train_loss_set = []
    valid_loss_set = []
    train_acc_set = []
    valid_acc_set = []
    invariant_epochs = 0

    for epoch_i in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss, train_acc = 0.0, 0.0
        valid_loss, valid_acc = 0.0, 0.0
        running_loss = 0.0
        current = 0

        #### Model for training
        model.train()
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch_i}, Training') as t:
            for i, (data, ret5) in t:
                assert label_type in ['RET5', 'RET20'], f"Wrong Label Type: {label_type}"
                if label_type == 'RET5':
                    target = ret5
                else:
                    target = ret20

                if target == 1:
                    target = torch.tensor([0, 1]).unsqueeze(0)
                    target = target.to(torch.float32)
                else:
                    target = torch.tensor([1, 0]).unsqueeze(0)
                    target = target.to(torch.float32)

                data, target = data.to(device), target.to(device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                running_loss = (len(data) * loss.item() + running_loss * current) / (len(data) + current)
                current += len(data)
                train_loss += loss.item() * data.size(0)
                # update training acc
                train_acc += (output.argmax(1) == target.argmax(1)).sum()

                t.set_postfix({'loss': running_loss})

        #### Model for validation
        model.eval()
        with tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Epoch {epoch_i}, Validation') as t:
            for i, (data, ret5) in t:
                assert label_type in ['RET5', 'RET20'], f"Wrong Label Type: {label_type}"
                if label_type == 'RET5':
                    target = ret5
                else:
                    target = ret20

                if target == 1:
                    target = torch.tensor([0, 1]).unsqueeze(0)
                    target = target.to(torch.float32)
                else:
                    target = torch.tensor([1, 0]).unsqueeze(0)
                    target = target.to(torch.float32)

                # move tensors to GPU if CUDA is available
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)
                valid_acc += (output.argmax(1) == target.argmax(1)).sum()

                t.set_postfix({'loss': running_loss})

        # Compute average loss
        train_loss = train_loss / len(train_loader.sampler)
        train_loss_set.append(train_loss)
        valid_loss = valid_loss / len(valid_loader.sampler)
        valid_loss_set.append(valid_loss)

        train_acc = train_acc / len(train_loader.sampler)
        train_acc_set.append(train_acc.cpu().numpy())
        valid_acc = valid_acc / len(valid_loader.sampler)
        valid_acc_set.append(valid_acc.cpu().numpy())

        print('Epoch: {} Training Loss: {:.6f} Validation Loss: {:.6f} Training Acc: {:.5f} Validation Acc: {:.5f}'.format(
            epoch_i, train_loss, valid_loss, train_acc, valid_acc))

        # if validation loss gets smaller, save the model
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            valid_loss_min = valid_loss
            invariant_epochs = 0
            torch.save({
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, savefile)
        else:
            invariant_epochs = invariant_epochs + 1

        if invariant_epochs >= early_stop_epoch:
            print(f"Early Stop at Epoch [{epoch_i}]: Performance hasn't enhanced for {early_stop_epoch} epochs")
            break

    return train_loss_set, valid_loss_set, train_acc_set, valid_acc_set

#RUN THE CODE

LEARNING_RATE = .00001
WEIGHT_DECAY = 0.01
MODEL_SAVE_FILE = '../CNN/models/model.pt'
# Move the model to the GPU if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN20d().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_n_epochs(100, model, 'RET5', train_dataloader, val_dataloader, criterion=criterion, optimizer=optimizer,
               early_stop_epoch=5, savefile=MODEL_SAVE_FILE)