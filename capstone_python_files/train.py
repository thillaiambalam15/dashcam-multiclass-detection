import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import collections
import time

import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import DrivingDetectionDataset, OUT_DIR, CLASS_NAMES
from model import DrivingDetector
from losses import compute_losses, evaluate, compute_class_weights, box_iou_loss, box_eiou_loss

BATCH_SIZE = 32
NUM_WORKERS = 6
BACKBONE_WARMUP_EPOCHS = 5
LR_FACTOR = 0.5
LR_PATIENCE = 2

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(HERE, "checkpoints")
os.makedirs(CHECKPOINTS, exist_ok=True)


def build_class_weights():
    label_dir = os.path.join(OUT_DIR, "train", "labels")
    counts = collections.Counter()
    for file in os.listdir(label_dir):
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                line = line.strip()
                if line:
                    cid = int(line.split()[0])
                    counts[cid] += 1
    count_list = [counts[i] for i in range(len(CLASS_NAMES))]
    print("real per-class train counts:", dict(zip(CLASS_NAMES, count_list)), flush=True)
    return compute_class_weights(count_list)


def setup_optimizer(model, lr=1e-4):
    do_warmup = model.pretrained and BACKBONE_WARMUP_EPOCHS > 0
    if do_warmup:
        for p in model.backbone_parameters:
            p.requires_grad = False
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        print(f"backbone FROZEN for the first {BACKBONE_WARMUP_EPOCHS} epochs (training det_head only)",
              flush=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print("training fully unfrozen from epoch 1" +
              (" (Model B -- no pretrained weights to protect)" if not model.pretrained else ""), flush=True)
    return optimizer, do_warmup


def train_model(model, box_loss_fn, run_name, train_loader, valid_loader, class_weights, device,
                 num_epochs=30, lr=1e-4):
    model = model.to(device)
    optimizer, do_warmup = setup_optimizer(model, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                            factor=LR_FACTOR, patience=LR_PATIENCE)

    history = []
    best_combined_score = -1.0
    best_ckpt_path = os.path.join(CHECKPOINTS, f"{run_name}_best.pt")
    n_batches_total = len(train_loader)

    for epoch in range(1, num_epochs + 1):
        if do_warmup and epoch == BACKBONE_WARMUP_EPOCHS + 1:
            for p in model.backbone_parameters:
                p.requires_grad = True
            optimizer.add_param_group({"params": list(model.backbone_parameters)})
            print(f"  epoch {epoch}: backbone UNFROZEN, fine-tuning end-to-end from here on", flush=True)

        print(f"Epoch {epoch}/{num_epochs}", flush=True)
        model.train()
        running = {"total": 0.0, "box_loss": 0.0, "cls_loss": 0.0, "obj_focal_loss": 0.0}
        n_seen = 0
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            pred = model(images)
            losses = compute_losses(pred, targets, class_weights, box_loss_fn)
            losses["total"].backward()
            optimizer.step()

            bs = images.size(0)
            for k in running:
                running[k] += losses[k].item() * bs
            n_seen += bs

            avg_loss = running["total"] / n_seen
            bar_len = 30
            filled = int(bar_len * batch_idx / n_batches_total)
            bar = "=" * filled + ">" + "." * max(0, bar_len - filled - 1)
            elapsed = time.time() - t0
            print(f"\r{batch_idx}/{n_batches_total} [{bar}] - {elapsed:.0f}s - loss: {avg_loss:.4f}",
                  end="", flush=True)

        train_metrics = {k: v / n_seen for k, v in running.items()}
        val = evaluate(model, valid_loader, class_weights, box_loss_fn, device)
        combined_score = (val["cls_accuracy"] + val["mean_iou"] + val["obj_f1"]) / 3
        scheduler.step(combined_score)

        print(f" - val_cls_acc: {val['cls_accuracy']:.4f} - val_iou: {val['mean_iou']:.4f} "
              f"- val_obj_f1: {val['obj_f1']:.4f} - combined: {combined_score:.4f}", flush=True)

        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                         **{f"val_{k}": v for k, v in val.items()},
                         "combined_score": combined_score, "epoch_seconds": time.time() - t0})
        pd.DataFrame(history).to_csv(os.path.join(CHECKPOINTS, f"{run_name}_history.csv"), index=False)

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "combined_score": combined_score}, best_ckpt_path)
            print(f"  epoch {epoch}: NEW BEST combined_score={combined_score:.4f} -> saved {best_ckpt_path}",
                  flush=True)

    return history, best_combined_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="e.g. modelA_iou, modelB_eiou")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--loss", choices=["iou", "eiou"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
          flush=True)
    print(f"=== {args.model_name}  pretrained={args.pretrained}  loss={args.loss}  epochs={args.epochs} ===",
          flush=True)

    train_loader = DataLoader(DrivingDetectionDataset("train", augment=True, need_boxes=False),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(DrivingDetectionDataset("valid", augment=False, need_boxes=False),
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                               pin_memory=True, persistent_workers=True)
    print(f"train images: {len(train_loader.dataset)}  valid images: {len(valid_loader.dataset)}  "
          f"batches/epoch: {len(train_loader)}", flush=True)

    class_weights = build_class_weights().to(device)

    box_loss_fn = box_iou_loss if args.loss == "iou" else box_eiou_loss
    model = DrivingDetector(pretrained=args.pretrained)

    history, best_score = train_model(model, box_loss_fn, args.model_name,
                                       train_loader, valid_loader, class_weights, device,
                                       num_epochs=args.epochs, lr=args.lr)
    print(f"\ntraining done. best combined_score={best_score:.4f}, "
          f"checkpoint at {os.path.join(CHECKPOINTS, args.model_name + '_best.pt')}")


if __name__ == "__main__":
    main()
