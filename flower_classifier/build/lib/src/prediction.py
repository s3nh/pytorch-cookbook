import os 
import argparse
import torch 
import torch.nn as nn 

import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.transforms as T
import random

from src.model import FlowerNet
from PIL import Image
from io import BytesIO

class ModelLoader():
    def __init__(self,  model_path='models/flower_cnn.pt', root =  'flowers'):
        self.model_path = model_path
        self.model = self._build_model() 
        self.indices = self.data_indices()

    def _build_model(self):
        model = FlowerNet()
        model.load_state_dict(torch.load(self.model_path))
        return model 

    def data_indices(self):
        dataset = datasets.ImageFolder(root = 'flowers')
        indices = dataset.class_to_idx
        return indices

    @staticmethod
    def image_prepare(image_path):
        image = BytesIO(image_path)
        image = Image.open(image).convert("RGB")
        image = T.Resize((224, 224))(image)
        image = T.ToTensor()(image)
        image = image[None].type('torch.FloatTensor')
        return image

    def _predict(self, image):
        predictions = self.model(image)
        predictions = predictions.max(1, keepdim=True)[1]
        output = list(self.indices.keys())[list(self.indices.values()).index(predictions[0])]
        return output

def main():
    mdl = ModelLoader()
    print(mdl.data_indices())
    print(mdl.model)

if __name__ == "__main__":
    main()
