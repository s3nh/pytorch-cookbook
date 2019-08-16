import os 
import torch
from torch.utils.data import DataLoader 
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Data Visualisation 
import matplotlib.pyplot as plt 
import numpy as np 
import warnings 
import torch.nn as nn 
import torch.nn.functional as F
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers=[]
    conv_layer=nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers=[]
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    layers.append(transpose_conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class CycleGan():
    def __init__(self, image_dir='data/cyclegan', image_size = 128, batch_size = 16, num_workers=0):
        self.image_dir=image_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Image resiz e

    def get_data_loader(self):
       transform = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor()])
       image_path =  self.image_dir
       train_path = os.path.join(image_path)
       test_path = os.path.join(image_path)
       train_dataset = datasets.ImageFolder(train_path, transform)
       test_dataset = datasets.ImageFolder(test_path, transform)

       train_loader =  DataLoader(dataset=train_dataset, batch_size = self.batch_size, shuffle=True, 
               num_workers=self.num_workers)
       test_loader = DataLoader(dataset=test_dataset, batch_size = self.batch_size, shuffle=False, 
               num_workers = self.num_workers)
       return train_loader, test_loader


class Discriminator(nn.Module):

    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim
        self.conv1 = conv(3, conv_dim, 4, batch_norm= False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.fc = nn.Linear(conv_dim*4*4*4, 1)

    def forward(self, x):

        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(out), 0.2)

        out = out.view(-1, self.conv_dim*4*4*4)
        out =  self.fc(out)
        return out

class Generator(nn.Module):

    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()
        self.conv_dim=conv_dim
        self.fc = nn.Linear(z_size, conv_dim*4*4*4)
        self.t_conv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.t_conv2 = deconv(conv_dim*2, conv_dim, 4)
        self.t_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):

        out = self.fc(x)
        out = out.view(-1,self.conv_dim*4, 4, 4)
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = self.t_conv3(out)
        out = F.tanh(out)
        return out

def main():
    cg=CycleGan()
    train_loader, test_loader = cg.get_data_loader()
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    conv_dim = 32
    z_size=100
    D=Discriminator(conv_dim)
    G=Generator(z_size=z_size,conv_dim=conv_dim)
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        G.cuda()
        D.cuda()
            
if __name__ == '__main__':
    main()

