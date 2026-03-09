#!/usr/bin/env python3
"""
YOLOv9-C COCO Validation (No Fault Injection)
Standalone script - evaluates YOLOv9-C on COCO validation set.
"""

import numpy as np
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_img_size, check_yaml, colorstr, non_max_suppression, scale_boxes, xywh2xyxy
from utils.metrics import ap_per_class, box_iou
from utils.torch_utils import select_device, smart_inference_mode


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def evaluate_yolo(weights, data_yaml, batch_size=4, imgsz=640, conf_thres=0.001, iou_thres=0.7, max_det=300, max_images=None):
    """Evaluate YOLOv9 on COCO validation set.

    Args:
        max_images: Maximum number of images to evaluate (None = all images)
    """
    device = select_device('0', batch_size=batch_size)

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data_yaml, fp16=False)
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)
    model.eval()

    # Data
    data = check_dataset(data_yaml)
    nc = int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    # Dataloader
    dataloader = create_dataloader(
        data['val'],
        imgsz,
        batch_size,
        stride,
        False,
        pad=0.5,
        rect=model.pt,
        workers=8,
        min_items=0,
        prefix=colorstr(f'val: ')
    )[0]

    seen = 0
    stats = []

    for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
        # Stop if we've reached max_images
        if max_images is not None and seen >= max_images:
            break

        im = im.to(device, non_blocking=True)
        targets = targets.to(device)
        im = im.float() / 255.0
        nb, _, height, width = im.shape

        # Inference
        preds = model(im, augment=False)

        # Handle model output - if list/tuple, take first element (inference output)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        preds = non_max_suppression(preds, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False, max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=None, names=model.names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        map50, map = 0.0, 0.0

    nt = np.bincount(stats[3].astype(int), minlength=nc)

    return map50, map, seen, nt.sum()


def main():
    print("YOLOv9-C COCO Validation")
    print("=" * 60)

    # Configuration
    weights = "../weights/yolov9-c-converted.pt"
    data_yaml = './data/coco.yaml'
    batch_size = 2
    max_images = 20  # Set to None to evaluate all images, or a number like 100

    data_yaml = check_yaml(data_yaml)

    print(f"Weights: {weights}")
    print(f"Device: cuda:0")
    print(f"Max images: {max_images if max_images else 'All (5000)'}")

    # Evaluate
    map50, map, num_images, num_instances = evaluate_yolo(
        weights=weights,
        data_yaml=data_yaml,
        batch_size=batch_size,
        max_images=max_images
    )

    # Results
    print(f"Number of images:    {num_images}")
    print(f"Number of instances: {num_instances}")
    print(f"mAP50:               {map50:.5f}")
    print(f"mAP50-95:            {map:.5f}")


if __name__ == "__main__":
    main()
