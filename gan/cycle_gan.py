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

def main():

    cg=CycleGan()
    train_loader, test_loader = cg.get_data_loader()
    print(train_loader)
   




if __name__ == '__main__':
    main()

