#!/usr/bin/env python3
"""
YOLOv9-C COCO Validation with PyTorch Profiler
Profiles kernel boundaries per batch using unified profiler context.
"""

import numpy as np
import torch
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity
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


def profile_yolo_with_boundaries(weights, data_yaml, batch_size=4, imgsz=640, num_batches=10):
    """Profile YOLOv9 kernel boundaries per batch.

    Args:
        num_batches: Number of batches to profile
    """
    device = select_device('0', batch_size=batch_size)

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data_yaml, fp16=False)
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)
    model.eval()

    # Data
    data = check_dataset(data_yaml)

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

    # Profile with unified context
    with torch.no_grad():
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
                if batch_i >= num_batches:
                    break

                im = im.to(device, non_blocking=True)
                im = im.float() / 255.0

                with record_function(f"batch_{batch_i}"):
                    # Inference
                    preds = model(im, augment=False)

                prof.step()

        # Extract and process events
        all_events = list(prof.events())
        all_events.sort(key=lambda e: e.time_range.start)

        # Build parent-child relationships
        def get_batch_from_scope(event):
            if hasattr(event, 'name') and event.name:
                if 'batch_' in event.name:
                    try:
                        batch_str = event.name.split('batch_')[1].split('_')[0] if '_' in event.name.split('batch_')[1] else event.name.split('batch_')[1]
                        return int(batch_str)
                    except (IndexError, ValueError):
                        pass
            return None

        event_to_parent = {}
        for i, event in enumerate(all_events):
            for j in range(i-1, -1, -1):
                parent = all_events[j]
                if (parent.time_range.start <= event.time_range.start and
                    parent.time_range.end >= event.time_range.end):
                    event_to_parent[id(event)] = parent
                    break

        def find_batch_recursively(event, visited=None):
            if visited is None:
                visited = set()
            if id(event) in visited:
                return None
            visited.add(id(event))

            batch = get_batch_from_scope(event)
            if batch is not None:
                return batch

            if id(event) in event_to_parent:
                parent = event_to_parent[id(event)]
                return find_batch_recursively(parent, visited)

            return None

        # Collect kernels by batch
        kernel_order_by_batch = {}
        batch_kernel_counter = {}
        global_kernel_counter = 0

        for event in all_events:
            if event.device_type == torch.autograd.DeviceType.CUDA and event.name:
                kernel_name = event.name

                # Skip memory operations
                lower = kernel_name.lower()
                if ("memcpy" in lower or "memset" in lower or
                    "void memcpy" in lower or "void memset" in lower):
                    continue

                # Skip profiling markers
                if 'batch_' in kernel_name:
                    continue

                assigned_batch = find_batch_recursively(event)
                if assigned_batch is None:
                    assigned_batch = "NO_BATCH"

                if assigned_batch not in kernel_order_by_batch:
                    kernel_order_by_batch[assigned_batch] = []
                    batch_kernel_counter[assigned_batch] = 0

                batch_kernel_counter[assigned_batch] += 1
                global_kernel_counter += 1

                kernel_order_by_batch[assigned_batch].append({
                    'global_index': global_kernel_counter,
                    'batch_index': batch_kernel_counter[assigned_batch],
                    'name': kernel_name,
                })

        # Compute boundaries
        boundaries = [0]
        running = 0
        batch_keys = sorted([k for k in kernel_order_by_batch.keys() if k != "NO_BATCH"])
        for b in batch_keys:
            running += len(kernel_order_by_batch[b])
            boundaries.append(running)

        # Save boundaries
        with open("yolo_boundary.txt", "w") as bf:
            for i in range(len(boundaries) - 1):
                start = boundaries[i] + 1 if i > 0 else boundaries[i]
                end = boundaries[i + 1]
                bf.write(f"{start},{end}\n")

        # Print summary
        print("\nProfiling kernel boundaries...")
        print(f"Batches profiled: {num_batches}\n")

        for batch_idx, batch_key in enumerate(batch_keys):
            kernels = kernel_order_by_batch[batch_key]
            start = boundaries[batch_idx] + 1 if batch_idx > 0 else boundaries[batch_idx]
            end = boundaries[batch_idx + 1]
            print(f"  Batch {batch_idx:3d}: {start:6d},{end:6d} ({len(kernels):4d} kernels)")

        print(f"\n✓ Kernel boundaries saved to: yolo_boundary.txt")
        print(f"  Total batches: {len(batch_keys)}")
        print(f"  Total kernels: {global_kernel_counter}")
        print(f"  Avg kernels/batch: {global_kernel_counter / len(batch_keys):.1f}\n")


def main():
    print("YOLOv9-C COCO Profiling")
    print("=" * 60)

    # Configuration
    weights = "../weights/yolov9-c-converted.pt"
    data_yaml = './data/coco.yaml'
    batch_size = 2
    num_batches = 10  # Number of batches to profile

    data_yaml = check_yaml(data_yaml)

    print(f"Weights: {weights}")
    print(f"Device: cuda:0")
    print(f"Batch size: {batch_size}")
    print(f"Batches to profile: {num_batches}")

    # Profile
    profile_yolo_with_boundaries(
        weights=weights,
        data_yaml=data_yaml,
        batch_size=batch_size,
        num_batches=num_batches
    )


if __name__ == "__main__":
    main()
