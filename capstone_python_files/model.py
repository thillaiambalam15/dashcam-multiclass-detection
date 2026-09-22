import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torchvision.models as tvm


class DrivingDetector(nn.Module):
    def __init__(self, num_classes=5, num_boxes=4, pretrained=True):
        super().__init__()

        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.pretrained = pretrained
        self.per_slot = 5 + num_classes

        weights = tvm.ResNet18_Weights.DEFAULT if self.pretrained else None
        backbone = tvm.resnet18(weights=weights)

        self.backbone_body = nn.Sequential(*list(backbone.children())[:-3])
        self.head_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.head_bn = nn.BatchNorm2d(256)
        self.head_relu = nn.ReLU(inplace=True)
        self.head_out = nn.Conv2d(256, self.per_slot * self.num_boxes, kernel_size=1)
        self.det_head = nn.Sequential(self.head_conv1, self.head_bn, self.head_relu, self.head_out)

    def forward(self, x):
        body = self.backbone_body(x)
        raw = self.det_head(body)
        raw = raw.permute(0, 2, 3, 1)
        raw = raw.contiguous().view(raw.shape[0], 26, 26, self.num_boxes, self.per_slot)

        obj_box = torch.sigmoid(raw[..., :5])
        cls_logits = raw[..., 5:]
        return torch.cat([obj_box, cls_logits], dim=-1)

    @property
    def backbone_parameters(self):
        return self.backbone_body.parameters()
