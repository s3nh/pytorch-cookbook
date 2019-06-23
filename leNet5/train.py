import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F


import os 
import argparse
import sys

from model import LeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
model = LeNet().to(device)

print(model)


def accuracy(true, pred):
    preds = pred.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape
    return acc





