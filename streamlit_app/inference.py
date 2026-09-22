import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import torch.nn.functional as F

from dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from model import DrivingDetector
from losses import batch_iou, box_cxcywh_to_xyxy

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(HERE, "checkpoints")

CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (0, 255, 255),
    3: (255, 0, 0),
    4: (255, 0, 255),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6


def load_model(checkpoint_name, pretrained, device):
    model = DrivingDetector(pretrained=pretrained)
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{checkpoint_name}_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_frame_gpu(frame, device):
    H, W = frame.shape[:2]

    img_tensor = torch.from_numpy(frame).to(device)
    img_tensor = img_tensor[:, :, [2, 1, 0]]
    img_tensor = img_tensor.permute(2, 0, 1).float()

    pad_size = max(W, H)
    pad_top = (pad_size - H) // 2
    pad_left = (pad_size - W) // 2
    pad_bottom = pad_size - H - pad_top
    pad_right = pad_size - W - pad_left

    img_tensor = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), value=114)

    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = F.interpolate(img_tensor, size=(416, 416), mode="bilinear", align_corners=False)
    img_tensor = img_tensor.squeeze(0)

    img_tensor = img_tensor / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor, pad_size, pad_left, pad_top


def decode_predictions(raw_output, confidence_threshold=0.3):
    
    if raw_output.dim() == 5:
        raw_output = raw_output.squeeze(0)

    grid_size = raw_output.shape[0]
    obj_mask = raw_output[..., 0] > confidence_threshold
    survivors = raw_output[obj_mask]

    indices = torch.nonzero(obj_mask)
    row = indices[:, 0]
    col = indices[:, 1]

    confidence = survivors[:, 0]
    cx = survivors[:, 1]
    cy = survivors[:, 2]
    w = survivors[:, 3]
    h = survivors[:, 4]
    cls_logits = survivors[:, 5:]
    predlabel = torch.argmax(cls_logits, dim=1)

    abs_cx = (col + cx) / grid_size
    abs_cy = (row + cy) / grid_size

    box = torch.stack([abs_cx, abs_cy, w, h], dim=1)
    return box, predlabel, confidence


def nms(box, predlabel, confidence, suppress_thresh=0.4):
    
    order = torch.argsort(confidence, descending=True)
    box = box[order]
    predlabel = predlabel[order]
    confidence = confidence[order]

    iou_matrix = batch_iou(box.unsqueeze(1), box.unsqueeze(0))

    suppressed = torch.zeros(box.shape[0], dtype=torch.bool, device=box.device)
    keep = []

    for i in range(box.shape[0]):
        if suppressed[i] == False:
            keep.append(i)
            targets = torch.nonzero(iou_matrix[i, i + 1:] > suppress_thresh, as_tuple=True)[0]
            targets = targets + i + 1
            suppressed[targets] = True

    keep_idx = torch.tensor(keep, dtype=torch.long, device=box.device)
    final_box = box[keep_idx]
    final_predlabel = predlabel[keep_idx]
    final_confidence = confidence[keep_idx]

    return final_box, final_predlabel, final_confidence


def draw_detections(frame, final_box, final_predlabel, final_confidence, pad_size, pad_left, pad_top):
    
    if final_box.shape[0] == 0:
        return frame

    cx = (final_box[..., 0] * pad_size) - pad_left
    cy = (final_box[..., 1] * pad_size) - pad_top
    w = final_box[..., 2] * pad_size
    h = final_box[..., 3] * pad_size

    video_label = torch.stack([cx, cy, w, h], dim=1)
    video_label = box_cxcywh_to_xyxy(video_label)

    for cid, conf, (x0, y0, x1, y1) in zip(final_predlabel, final_confidence, video_label):
        pt1, pt2 = (int(x0), int(y0)), (int(x1), int(y1))
        color = CLASS_COLORS[int(cid)]
        text = f"{CLASS_NAMES[int(cid)]} {conf:.2%}"
        org = (int(x0), int(y0) - 10)
        cv2.rectangle(frame, pt1, pt2, color, thickness=2)
        cv2.putText(frame, text, org, FONT, FONT_SCALE, color, thickness=2)

    return frame
