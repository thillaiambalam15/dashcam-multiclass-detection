import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F

BOX_LOSS_WEIGHT = 2.0
BETA = 0.999


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou_loss(pred, target, eps=1e-7):
    p = box_cxcywh_to_xyxy(pred)
    t = box_cxcywh_to_xyxy(target)

    ix0 = torch.max(p[:, 0], t[:, 0])
    iy0 = torch.max(p[:, 1], t[:, 1])
    ix1 = torch.min(p[:, 2], t[:, 2])
    iy1 = torch.min(p[:, 3], t[:, 3])

    iw = (ix1 - ix0).clamp(min=0)
    ih = (iy1 - iy0).clamp(min=0)
    intersection_area = iw * ih

    pred_area = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    target_area = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])

    union_area = pred_area + target_area - intersection_area + eps
    iou = intersection_area / union_area
    return (1 - iou).mean()


def box_eiou_loss(pred, target, eps=1e-7):
    p, t = box_cxcywh_to_xyxy(pred), box_cxcywh_to_xyxy(target)
    px0, py0, px1, py1 = p.unbind(-1)
    tx0, ty0, tx1, ty1 = t.unbind(-1)

    ix0, iy0 = torch.max(px0, tx0), torch.max(py0, ty0)
    ix1, iy1 = torch.min(px1, tx1), torch.min(py1, ty1)
    inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
    area_p = (px1 - px0).clamp(min=0) * (py1 - py0).clamp(min=0)
    area_t = (tx1 - tx0).clamp(min=0) * (ty1 - ty0).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    cx0, cy0 = torch.min(px0, tx0), torch.min(py0, ty0)
    cx1, cy1 = torch.max(px1, tx1), torch.max(py1, ty1)
    cw = (cx1 - cx0).clamp(min=eps)
    ch = (cy1 - cy0).clamp(min=eps)

    p_cx, p_cy = (px0 + px1) / 2, (py0 + py1) / 2
    t_cx, t_cy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    rho2 = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2
    c2 = cw ** 2 + ch ** 2 + eps

    pw, ph = (px1 - px0), (py1 - py0)
    tw, th = (tx1 - tx0), (ty1 - ty0)
    rho_w2 = (pw - tw) ** 2
    rho_h2 = (ph - th) ** 2

    eiou = iou - rho2 / c2 - rho_w2 / (cw ** 2 + eps) - rho_h2 / (ch ** 2 + eps)
    return (1 - eiou).mean()


def batch_iou(pred, target):
    p, t = box_cxcywh_to_xyxy(pred), box_cxcywh_to_xyxy(target)
    x0 = torch.max(p[..., 0], t[..., 0])
    y0 = torch.max(p[..., 1], t[..., 1])
    x1 = torch.min(p[..., 2], t[..., 2])
    y1 = torch.min(p[..., 3], t[..., 3])
    inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
    area_p = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] - p[..., 1]).clamp(min=0)
    area_t = (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)
    union = area_p + area_t - inter
    return inter / union.clamp(min=1e-6)


def add_cell_offsets(coords, grid_size):
    S = grid_size
    row_idx = torch.arange(S, device=coords.device, dtype=coords.dtype).view(1, S, 1, 1)
    col_idx = torch.arange(S, device=coords.device, dtype=coords.dtype).view(1, 1, S, 1)

    cx, cy, w, h = coords.unbind(-1)
    abs_cx = (col_idx + cx) / S
    abs_cy = (row_idx + cy) / S
    return torch.stack([abs_cx, abs_cy, w, h], dim=-1)


def compute_class_weights(counts, beta=BETA):
    counts = torch.tensor(counts, dtype=torch.float32)
    effective_num = 1 - torch.pow(beta, counts)
    weights = (1 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)
    return weights


def focal_bce(pred_prob, target, alpha=0.25, gamma=2.0, eps=1e-7):
    pred_prob = pred_prob.clamp(min=eps, max=1 - eps)
    pt = torch.where(target > 0.5, pred_prob, 1 - pred_prob)
    alpha_t = torch.where(target > 0.5, alpha, 1 - alpha)
    loss = -alpha_t * (1 - pt).pow(gamma) * torch.log(pt)
    num_positive = (target > 0.5).sum().clamp(min=1)
    return loss.sum() / num_positive


def compute_losses(pred, target, class_weights, box_loss_fn):
    obj_mask = target[..., 0] > 0.5
    obj_focal_loss = focal_bce(pred[..., 0], target[..., 0])

    if obj_mask.any():
        grid_size = pred.shape[1]
        pred_boxes = add_cell_offsets(pred[..., 1:5], grid_size)[obj_mask]
        target_boxes = add_cell_offsets(target[..., 1:5], grid_size)[obj_mask]
        box_loss = box_loss_fn(pred_boxes, target_boxes)
        mean_iou = batch_iou(pred_boxes.detach(), target_boxes.detach()).mean()

        pred_cls_logits = pred[..., 5:][obj_mask]
        target_cls_idx = target[..., 5:][obj_mask].argmax(dim=-1)
        cls_loss = F.cross_entropy(pred_cls_logits, target_cls_idx, weight=class_weights)
        cls_correct = (pred_cls_logits.detach().argmax(dim=-1) == target_cls_idx).float().mean()
    else:
        box_loss = torch.tensor(0.0, device=pred.device)
        cls_loss = torch.tensor(0.0, device=pred.device)
        mean_iou = torch.tensor(0.0, device=pred.device)
        cls_correct = torch.tensor(0.0, device=pred.device)

    total = BOX_LOSS_WEIGHT * box_loss + cls_loss + obj_focal_loss
    return {
        "total": total, "box_loss": box_loss, "cls_loss": cls_loss,
        "obj_focal_loss": obj_focal_loss,
        "mean_iou": mean_iou, "cls_accuracy": cls_correct,
    }


@torch.no_grad()
def evaluate(model, loader, class_weights, box_loss_fn, device):
    model.eval()
    totals = {"box_loss": 0.0, "cls_loss": 0.0, "obj_focal_loss": 0.0,
              "mean_iou": 0.0, "cls_accuracy": 0.0}
    tp = fp = fn = 0
    n_batches = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        pred = model(images)
        losses = compute_losses(pred, targets, class_weights, box_loss_fn)
        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

        pred_obj = pred[..., 0] > 0.5
        target_obj = targets[..., 0] > 0.5
        tp += (pred_obj & target_obj).sum().item()
        fp += (pred_obj & ~target_obj).sum().item()
        fn += (~pred_obj & target_obj).sum().item()

    result = {k: v / n_batches for k, v in totals.items()}
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    result["obj_f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    result["obj_precision"] = precision
    result["obj_recall"] = recall
    return result
