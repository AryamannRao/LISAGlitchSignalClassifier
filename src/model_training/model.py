import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np

class CNN(nn.Module):
    def __init__(self, width=4, bn=True):
        
        super(CNN, self).__init__()
        self.width = width
        self.bn = bn
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.width,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.width,
                               out_channels=self.width*2,
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.width*2,
                               out_channels=self.width*4,
                               kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.width*4,
                               out_channels=self.width*8,
                               kernel_size=3,
                               padding=1)
        
        if bn:
            self.bn1 = nn.BatchNorm2d(self.width)
            self.bn2 = nn.BatchNorm2d(self.width*2)
            self.bn3 = nn.BatchNorm2d(self.width*4)
            self.bn4 = nn.BatchNorm2d(self.width*8)
        
        self.pool = nn.MaxPool2d(2, 2)
       
        self.fc1 = nn.Linear(self.width * 8 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 2)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        if self.bn:
            x = self.bn1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        if self.bn:
            x = self.bn2(x)
        x = self.pool(torch.relu(self.conv3(x)))
        if self.bn:
            x = self.bn3(x)
        x = self.pool(torch.relu(self.conv4(x)))
        if self.bn:
            x = self.bn4(x)
        x = x.view(-1, self.width * 8 * 16 * 16)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class LISADataset(Dataset):
    def __init__(self, h5_path):
        
        self.file = h5py.File(h5_path, "r")
        
        self.images = self.file["images"]
        self.labels = self.file["labels"]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        image = self.images[idx]
        label = self.labels[idx]
        
        # convert to float32
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        
        # convert HWC → CHW
        image = np.transpose(image, (2,0,1))
        
        # convert to torch tensor
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        return image, label