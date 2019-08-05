# Link to paper

# https://arxiv.org/abs/1406.2661


## iThis tutorial I want to show simple Generative Adversarial Network Implementation 
# Using Pytorch



### Imports section 


import utils
import torch
import os
import time
import pickle 
import numpy as np 
# nn module as nn

import torch.nn as nn 
import torch.optim as optim

# Dataloader is predefined module for reading dataset;

from dataloader import dataloader 



# Begin with generator class
# Generator is responsible for 'generate' (woah) images 
# from rando noise. 
# based on conditional discriminator/generator nosise, weights are updated
# and images become more and more similar to origin as 
# soon as loss function value become smaller and smaller 


class Generator(nn.Module):

    # Initialize parameters

    def __init__(self, input_dim, output_dim, input_size):
        """

        __init__:
            Args:

            input_dim: Dimension on input data 
            output_dim: Dimension of output ata
            input_size


       """
       

       super(Generator, self).__init__()
       self.input_dim = input_dim
       self.output_dim = output_dim
       self.input_size = input_size

       # Define fully connected layer structure

       self.fc = nn.Sequential(
               nn.Linear(self.input_dim, 1024), 
               # Batch Normalization
               nn.BatchNorm1d(1024), 
               # Rectified Unit as activation functon 
               # argmax(0, x)
               nn.ReLU(), 
               # Linear layer
               nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)), 
               nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)), 
               nn.ReLU(), 

               )

       # Deconvoutional layer 

       self.deconv =  nn.Sequential(
               nn.ConvTranspose2d(128, 64, 4, 2, 1), 
               nn.BatchNorm2d(64), 
               nn.ReLU(), 
               nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1), 
               nn.Tanh(), 
               )
       # Weights initialization 

       utils.initialize_weights(self)



       # Define forward pass

       def forward(self, input):
           x = self.fc(input)
           x =  x.view(-1, 128, (self.input_size //4), (self.input_size // 4))
           x = self.deconv(x)


           return x


       # Discriminator Definition


   
class Discriminator(nn.Module):


    # Logic of this network is very similar to generator


    def __init__(self, input_dim = 1, output_dim = 1, input_size = 32):
        super(Discriminator, self).__init__()


        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.input_size = input_size




        # Define convolutional layer


        self.conv = nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2,1), 
                nn.LeakyReLU(0.2), #Similar to relu argmaax(0.2, x)
                nn.Conv2d(64, 128, 4, 2, 1), 
                nn.BatchNorm2d(128), 
                nn.LeakyReLU(0.2), 
                )

        # Define fully connected layer 

   def forward(self, input):

       x = self.conv(input)
       x = x.view(-1, 128 * (self.input_size //4) * (self.input_size //4))
       x = self.fc(x)


       return x
# To do  - Define GAN Class

class GAN(object):


    def __init__(self, args):

        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir  = args.result_dir
        self.dataset = args.dataset
        self.gpu_mode = args.gpu_mode
        self.input_size = args.input_size


        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)

        data = self.data_loader.__iter__().__next__()[0]

        self.G = Generator(input_dim = self.z_dim, output_dim = data.shape[1], input_size = self.input_size)
        self.D = Discriminator(input_dim = data.shape[1], output_dim=1, input_size = self.input_size)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr = args.lrD, betas= (args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        


        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        

        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

   
    def train():

        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []



        self._y_real_ = torch.ones(self.batch_size, 1)
        self.y_fake_ = torch.zeros(self.batch_size, 1)

        if self.gpu_mode:
            self._y_real_ = self.y_real_.cuda()
            self._y_fake_ = self.y_fake_.cuda()

        self.D.train()
        for epoch in range(self.epoch):
            self.G.train()


            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:

                    break
                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # Update Discriminator network


                self.D_optmizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real)


                G_ =  self.G(z)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())


                D_loss.backward()
                self.D_optimizer.step()

                # Update Generator Network

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())


                G_loss.backward()
                self.G_optimizer.step()


            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name, self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    

            

