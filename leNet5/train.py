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


