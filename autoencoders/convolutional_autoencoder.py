"""
Convolutional AutoEncoder 
Simplest architecture
"""
import numpy as np 
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms, datasets 

class DataPreparing():

    def __init__(self,  num_workers=0, batch_size=20, root='data'):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.root = root
        self.transform = transforms.ToTensor()
        self.train_loader, self.test_loader = self.data_loader()
    
    def data_loader(self):
        train_data = datasets.MNIST(root = self.root, train=True, download=True, transform = self.transform)
        test_data = datasets.MNIST(root = self.root, train=False, download=True, transform = self.transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.batch_size, 
                num_workers = self.num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = self.batch_size, 
                num_workers = self.num_workers)
        return train_loader, test_loader

    def data_vis(self):
       dataiter = iter(self.train_loader)
       images, labels = dataiter.next()
       images = images.numpy()


       img = np.squeeze(images[0])

       fig = plt.figure(figsize = (5,5))
       ax = fig.add_subplot(111)
       ax.imshow(img, cmap = 'gray')

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.t_conv1 =  nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride = 2)
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x)) 
        x = F.sigmoid(self.t_conv2(x))
        return x


class TrainNetwork(DataPreparing):
    pass

    
def main():
    model = ConvAutoEncoder()
    print(model)
    dp = DataPreparing()
    tn = TrainNetwork()
    print(tn.num_workers)

if __name__ == "__main__":
    main()
