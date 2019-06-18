import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T


import os 
import argparse 
import sys
import random


def load_file():
    with open('data/short-jokes/shortjokes.csv', 'r') as outfile:
        records = [line.split() for line in outfile]
    
    return records        

def random_set(set_len, batch_size):

    inp = torch.LongTensor(batch_size, set_len)
    target = torch.LongTensor(batch_size, set_len)
    #For every element in batch size

    for bi in range(batch_size):
        start_index = random.randint(0, file_len-set_len)
        end_index = start_index + set_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] =  char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    # Add cuda

    return inp, target



def main():

    data = load_file()

if __name__ == "__main__":

    main()
