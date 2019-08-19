import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator



def load_pretrain():

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    # TODO : Read about mobilenet structure

    anchor_generator = AnchorGenerator(sizes = ((32, 64, 128, 256, 512), ), 
            aspect_ratios=((0.5, 1.0, 2.0), ))

