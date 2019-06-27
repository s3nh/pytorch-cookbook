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


