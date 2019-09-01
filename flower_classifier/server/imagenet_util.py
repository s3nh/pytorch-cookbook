import torch
from torchvision import models, transforms
from torch.autograd import Variable

from io import BytesIO
from PIL import Image

import json

def image_loader(image):
	loader = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

	image_bytes  = BytesIO(image)

	image_processed = Image.open(image_bytes)
	image_processed = loader(image_processed).float()
	image_processed = Variable(image_processed).unsqueeze(0)

	return image_processed.float()

model = models.vgg16(pretrained=True)
classes = dict(json.loads(open ('../imagenet_classes.json').read()))