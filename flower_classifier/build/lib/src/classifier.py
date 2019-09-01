import torch
import torchvision
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets

def split_data():


    split = int(np.floor(valid_size * num_train))


def load_data(train_fraction =0.7):
    
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor()]) 

    flower_dataset = datasets.ImageFolder(root = 'flowers', 
        transform = data_transform)
    
    
    # TODO Normalize 
   
    num_train = len(flower_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.3 * num_train))
    print(split)


    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
            flower_dataset, batch_size = 16, sampler =train_sampler, 
            num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(
            flower_dataset, batch_size = 16, sampler = valid_sampler, 
            num_workers = 4)



    return data_transform, train_loader, valid_loader

    

