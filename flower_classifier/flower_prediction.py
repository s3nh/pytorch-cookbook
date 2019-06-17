import os 
import argparse
import torch 
import torch.nn as nn 

import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.transforms as T
import random


from model import FlowerNet
from PIL import Image


def load_model():


    model = FlowerNet()
    model.load_state_dict(torch.load('models/flower_cnn.pt'))

    return model 

def load_image(path):
    image = Image.open(path)
    image = image.convert("RGB")
    return image



def data_indices():
    dataset = datasets.ImageFolder(root = 'flowers')
    indices = dataset.class_to_idx
    return indices
            
def main():
    device = 'cuda'
    model = load_model()
    print(model)
    image = load_image('flowers/daisy/8382667241_0f046cecdb_n.jpg')
    image.show()
    image = T.Resize((224,224))(image)
    image = T.ToTensor()(image)
    image = image.to(device)
    image = image[None].type('torch.FloatTensor')
    predict = model(image)
    pred = predict.max(1, keepdim=True)[1]
    indices = data_indices()
    
    output = list(indices.keys())[list(indices.values()).index(pred[0])]
    print("Predicted value is {}".format(output))

if __name__ == "__main__":
    main()
