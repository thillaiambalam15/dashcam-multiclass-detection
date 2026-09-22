
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

HERE = r"C:\Users\thill\jupyter_folder\2 Capstone\driving_detection\Final_Submission"
OUT_DIR = os.path.join(HERE, "final_dataset")

CLASS_NAMES = ['biker', 'car', 'pedestrian', 'truck', 'trafficLight']
NUM_CLASSES = len(CLASS_NAMES)
TARGET_SIZE = 416
GRID_SIZE = 26
NUM_BOXES = 4
PAD_COLOR = (114, 114, 114)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

flip_jitter = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
])

NORMALIZE = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_yolo_labels(label_file):
    boxes = []
    with open(label_file, "r") as f:
        raw_label = f.readlines()
        boxes = [tuple([int(x.split()[0])] + [float(y) for y in x.split()[1:]]) for x in raw_label]
    return boxes


def random_translate(img, boxes):
    W, H = img.size

    scale = random.uniform(0.8, 1.2)
    new_W, new_H = int(W * scale), int(H * scale)
    scaled_image = img.resize((new_W, new_H), Image.BILINEAR)

    max_dx, max_dy = int(W * 0.1), int(H * 0.1)
    offset_x = (W - new_W) // 2 + random.randint(-max_dx, max_dx)
    offset_y = (H - new_H) // 2 + random.randint(-max_dy, max_dy)

    canvas = Image.new('RGB', (W, H), PAD_COLOR)
    canvas.paste(scaled_image, (offset_x, offset_y))

    if len(boxes) == 0:
        return canvas, []

    boxes = torch.tensor(boxes, dtype=torch.float64)

    cid = boxes[:, 0]
    cx = boxes[:, 1]
    cy = boxes[:, 2]
    w = boxes[:, 3]
    h = boxes[:, 4]

    px_cx = (cx * W * scale) + offset_x
    px_cy = (cy * H * scale) + offset_y
    px_w = w * W * scale
    px_h = h * H * scale

    actual_x0 = px_cx - px_w / 2
    actual_y0 = px_cy - px_h / 2
    actual_x1 = px_cx + px_w / 2
    actual_y1 = px_cy + px_h / 2

    actual_area = (actual_x1 - actual_x0) * (actual_y1 - actual_y0)

    px_x0 = actual_x0.clamp(min=0)
    px_y0 = actual_y0.clamp(min=0)
    px_x1 = actual_x1.clamp(max=W)
    px_y1 = actual_y1.clamp(max=H)

    visible_area = (px_x1 - px_x0) * (px_y1 - px_y0)

    new_cx = (px_x0 + px_x1) / 2 / W
    new_cy = (px_y0 + px_y1) / 2 / H
    new_w = (px_x1 - px_x0) / W
    new_h = (px_y1 - px_y0) / H

    mask = (visible_area / actual_area) > 0.4

    filtered_labels = torch.stack([cid, new_cx, new_cy, new_w, new_h], dim=1)
    filtered_labels = filtered_labels[mask]

    random_translated_boxes = [(int(c), cx_, cy_, w_, h_) for c, cx_, cy_, w_, h_ in filtered_labels.tolist()]
    return canvas, random_translated_boxes


def letterbox(img, boxes):
    W, H = img.size

    pad_size = max(W, H)
    pad_top = (pad_size - H) // 2
    pad_left = (pad_size - W) // 2

    canvas = Image.new('RGB', (pad_size, pad_size), PAD_COLOR)
    canvas.paste(img, (pad_left, pad_top))

    if len(boxes) == 0:
        return canvas, []

    boxes = torch.tensor(boxes, dtype=torch.float64)

    cid = boxes[:, 0]

    boxes[:, 1] = (boxes[:, 1] * W + pad_left) / pad_size
    boxes[:, 2] = (boxes[:, 2] * H + pad_top) / pad_size
    boxes[:, 3] = (boxes[:, 3] * W) / pad_size
    boxes[:, 4] = (boxes[:, 4] * H) / pad_size

    letterboxed_labels = [(int(c), cx_, cy_, w_, h_) for c, cx_, cy_, w_, h_ in boxes.tolist()]
    return canvas, letterboxed_labels


def apply_flip_jitter(canvas, boxes):
    if len(boxes) == 0:
        return flip_jitter(canvas), []

    cids = [b[0] for b in boxes]
    geo = torch.tensor([[b[1] * TARGET_SIZE, b[2] * TARGET_SIZE, b[3] * TARGET_SIZE, b[4] * TARGET_SIZE]
                         for b in boxes], dtype=torch.float32)

    tv_boxes = tv_tensors.BoundingBoxes(geo, format='CXCYWH', canvas_size=(TARGET_SIZE, TARGET_SIZE))
    out_canvas, out_boxes = flip_jitter(canvas, tv_boxes)

    result = [(cid, x[0] / TARGET_SIZE, x[1] / TARGET_SIZE, x[2] / TARGET_SIZE, x[3] / TARGET_SIZE)
              for cid, x in zip(cids, out_boxes.tolist())]
    return out_canvas, result


def build_target_grid(boxes):
    n_dropped = 0
    slots_used = {}
    target = torch.zeros(GRID_SIZE, GRID_SIZE, NUM_BOXES, 5 + NUM_CLASSES)
    ordered = sorted(boxes, key=lambda b: b[3] * b[4], reverse=True)

    for cid, cx, cy, w, h in ordered:
        row = min(int(cy * GRID_SIZE), GRID_SIZE - 1)
        col = min(int(cx * GRID_SIZE), GRID_SIZE - 1)
        key = (row, col)
        slots_used[key] = slots_used.get(key, 0) + 1
        used = slots_used.get(key, 0)
        slot = used - 1
        if slot >= NUM_BOXES:
            n_dropped += 1
        else:
            target[row, col, slot, 0] = 1.0
            target[row, col, slot, 1] = (cx * GRID_SIZE) - col
            target[row, col, slot, 2] = (cy * GRID_SIZE) - row
            target[row, col, slot, 3] = w
            target[row, col, slot, 4] = h
            target[row, col, slot, 5 + cid] = 1.0

    return target, n_dropped


def unnormalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    img = img_tensor * std_t + mean_t
    img = img.clamp(0, 1)
    img = (img * 255).byte()
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)


class DrivingDetectionDataset(Dataset):
    def __init__(self, split, augment=False, need_boxes=False):
        assert split in ("train", "valid", "test")
        self.split = split
        self.augment = augment
        self.need_boxes = need_boxes
        self.img_dir = os.path.join(OUT_DIR, self.split, "images")
        self.label_dir = os.path.join(OUT_DIR, self.split, "labels")
        self.bases = sorted(f[:-4] for f in os.listdir(self.img_dir))

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, idx):
        base = self.bases[idx]
        img = Image.open(os.path.join(self.img_dir, base + '.jpg')).convert("RGB")
        boxes = load_yolo_labels(os.path.join(self.label_dir, base + '.txt'))

        if self.augment:
            img, boxes = random_translate(img, boxes)

        canvas, boxes = letterbox(img, boxes)
        canvas = canvas.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)

        if self.augment:
            canvas, boxes = apply_flip_jitter(canvas, boxes)

        target, _ = build_target_grid(boxes)
        img_tensor = NORMALIZE(canvas)

        if self.need_boxes:
            return img_tensor, target, boxes
        else:
            return img_tensor, target
