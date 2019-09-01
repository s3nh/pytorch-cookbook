from __future__ import print_function
import argparse
import os 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms
from src.utils import load_data

class FlowerNet(nn.Module):
    def __init__(self):
        super(FlowerNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 50, 5, 1)
        self.conv2 = nn.Conv2d(50, 100, 5, 1)
        self.conv3 = nn.Conv2d(100, 20, 3, 1)
        self.fc1 = nn.Linear(12500, 16)
        self.fc2 =  nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def main():
    datatransform, train_loader, valid_loader = load_data()
    device = torch.device('cuda')
    model = FlowerNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.5)

    for epoch in range(1, 100):
        train(model, device, train_loader, optimizer, epoch)

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), "models/flower_cnn.pt")

if __name__ == "__main__":
    main()
