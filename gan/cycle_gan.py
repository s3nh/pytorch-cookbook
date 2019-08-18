"""
DCGAN paper

https://arxiv.org/pdf/1511.06434.pdf
"""
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
import torch.optim as optim
import pickle

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
    def __init__(self, z_size=100, conv_dim=32):
        super(Generator, self).__init__()
        self.conv_dim=conv_dim
        self.z_size = z_size
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

class CycleGan(Discriminator, Generator):
    def __init__(self, image_dir='data/cyclegan', image_size = 128, batch_size = 16, num_workers=0, train_on_gpu=True, print_every=30):
        super().__init__()
        self.image_dir=image_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.D = Discriminator(conv_dim = self.conv_dim)
        self.G = Generator(z_size = self.z_size, conv_dim = self.conv_dim)
        self.train_loader, self.test_loader = self.get_data_loader()
        print("Generator {}".format(self.G.parameters()))
        print("Discriminator {}".format(self.D.parameters()))
        self.d_optimizer, self.g_optimizer = self.define_parameters(self.D, self.G)
        print("{}".format(self.d_optimizer))
        self.train_on_gpu =train_on_gpu
        self.print_every = print_every

    def real_loss(self, D_out, smooth=False, train_on_gpu=False):
        batch_size=D_out.size(0)
        if smooth:
            labels = torch.ones(batch_size)*0.9
        else:
            labels = torch.ones(batch_size)
        if self.train_on_gpu:
            labels = labels.cuda()
        criterion=nn.BCEWithLogitsLoss()
        loss = criterion(D_out.squeeze(), labels)
        return loss

    def fake_loss(self, D_out):
        batch_size = D_out.size(0)
        labels = torch.zeros(batch_size)
        if self.train_on_gpu:
            labels=labels.cuda()
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(D_out.squeeze(), labels)
        return loss

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
	
    def training_(self, num_epochs=50, sample_size = 16):
        print("training started")
        samples = []
        losses = []
        fixed_z = np.random.uniform(-1, 1, size =(sample_size, self.z_size))
        fixed_z = torch.from_numpy(fixed_z).float()
        for epoch in range(num_epochs):
            for batch_i, (real_images, _) in enumerate(self.train_loader):
                batch_size = real_images.size(0)
                # Remove grad
                self.d_optimizer.zero_grad()

                D_real = self.D(real_images)
                d_real_loss = self.real_loss(D_real)
                
                # Fake images generation

                z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_size))
                z = torch.from_numpy(z).float()
                if self.train_on_gpu:
                    z=z.cuda()
                
                fake_images = self.G(z)
                # Interpretation - egenerate discriminator 
                # result are put into fake_loss function
                # which is a part of overall loss
                D_fake = self.D(fake_images)
                d_fake_loss  = self.fake_loss(D_fake)
                d_loss = d_real_loss + d_fake_loss
                # Backprop on discriminator prop
                d_loss.backward()
                self.d_optimizer.step()
                
                """ Generator training """
                self.g_optimizer.zero_grad()

                z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_size))
                z = torch.from_numpy(z).float()
                if self.train_on_gpu:
                    z = z.cuda()
                fake_images = self.G(z)


                D_fake = self.D(fake_images)
                g_loss = self.real_loss(D_fake)

                g_loss.backward()
                self.g_optimizer.step()

                if batch_i % self.print_every == 0:
                    losses.append((d_loss.item(), g_loss.item()))
            self.G.eval()
            if self.train_on_gpu:
                fixed_z = fixed_z.cuda()
            samples_z = self.G(fixed_z)
            samples.append(samples_z)
            G.train()

        with open('train_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)

    def define_parameters(D, G, lr_=1e-3, beta1=0.5, beta2=0.999):
        d_optimizer = optim.Adam(params = D.parameters(), lr=0.0002, betas= [beta1, beta2])
        g_optimizer = optim.Adam(params = G.parameters(), lr=0.0002, betas= [beta1, beta2])
        return d_optimizer, g_optimizer
		
def main():
    cg=CycleGan()
    cg.training_() 

if __name__ == '__main__':
    main()

