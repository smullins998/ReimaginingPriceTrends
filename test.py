import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
from torch import nn

# Set seed for reproducibility
torch.manual_seed(42)

# Define image dimensions
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

# Load DATA
year_list = np.arange(2001, 2020, 1)

# Lists to store images and labels
images = []
label_df = []

# Load images and labels for each year
for year in year_list:
    images.append(
        np.memmap(os.path.join("./img_data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_images.dat"),
                  dtype=np.uint8, mode='r').reshape((-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20])))
    label_df.append(
        pd.read_feather(os.path.join("./img_data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather")))

# Concatenate images and labels
images = np.concatenate(images)
label_df = pd.concat(label_df)

# Display shapes of the loaded data
print("Images Shape:", images.shape)
print("Labels Shape:", label_df.shape)

# Create Dataloader
class MyDataset(Dataset):
    def __init__(self, img, label):
        self.img = torch.Tensor(img.copy())
        self.label = torch.Tensor(label)
        self.len = len(img)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

# Create dataset and DataLoader
dataset = MyDataset(images, (label_df.Ret_5d > 0).values)
test_dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)


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


# Model path and device
net_path = './CNN/baseline_epoch_1_train_0.694174_val_0.276107.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pre-trained model
net = torch.load(net_path)

# TEST NETWORK
def eval_loop(dataloader, net, loss_fn):
    running_loss = 0.0
    total_loss = 0.0
    current = 0
    net.eval()
    target = []
    predict = []

    with torch.no_grad():
        with tqdm(dataloader) as t:
            for batch, (X, y) in enumerate(t):
                X = X.to(device)
                y = y.to(device)

                y = [[0, 1] if i == 1 else [1, 0] for i in y]
                y = torch.tensor(y)
                y = y.to(torch.float32)
                y = y.to(device)

                y_pred = net(X)
                target.append(y.detach())
                predict.append(y_pred.detach())
                loss = loss_fn(y_pred, y)

                running_loss = (len(X) * loss.item() + running_loss * current) / (len(X) + current)
                current += len(X)
                t.set_postfix({'running_loss': running_loss})

    return total_loss, torch.cat(predict), torch.cat(target)

# Get current time for creating unique file names
start_time = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')

# Loss function
loss_fn = torch.nn.BCELoss()

# Evaluate the model on the test set
test_loss, y_pred, y_target = eval_loop(test_dataloader, net, loss_fn)

# Convert logits to probabilities and save to CSV
predict_logit = (torch.nn.Softmax(dim=1)(y_pred)[:, 1]).cpu().numpy()
logit_output = pd.Series(predict_logit)
logit_output.to_csv('./CNN/logits/' + f'logits_{start_time}.csv')
