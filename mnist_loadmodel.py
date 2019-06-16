import os 
import argparse
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import datasets, transforms
import random

from mnist_convnet import Net
from torch.autograd import Variable
from PIL import Image 
def build_args():


    pass



def load_model():


    model = Net()
    model.load_state_dict(torch.load('mnist_cnn.pt'))
    
    return model 

def test_loader_():

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download = True,
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])) , batch_size = 16, shuffle=True)
    return test_loader

def model_predict():
    pass

device = 'cuda'


def main():


    model = load_model()
    model.eval()
    test = test_loader_()
    index =  len(test)
    chosen = random.choice(range(index))
    example  = test.dataset.data[chosen]
    image_ =  Image.fromarray(example.data.numpy())
    example = example.to(device)
    example = example[None, None].type('torch.FloatTensor')
    pred = model(example)
    pred = pred.max(1, keepdim=True)[1]
    print(pred)
    image_.show()
if __name__ == "__main__":
    main()



