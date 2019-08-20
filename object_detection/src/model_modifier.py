import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator



def finetune_model():
    # Loading model pretrained on COCO dataset
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


def load_pretrain():
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    # TODO : Read about mobilenet structure

    anchor_generator = AnchorGenerator(sizes = ((32, 64, 128, 256, 512), ), 
            aspect_ratios=((0.5, 1.0, 2.0), ))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names= [0],
            output_size = 7, sampling_ratio=2)

    model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, 
            box_roi_pool=roi_pooler)
    
    



    
