#!/usr/bin/env python3
"""
shufflenetV2 CIFAR-100 Profiling (resnet.py style)
Profiles kernel boundaries using unified profiler context.
"""

import torch
from torchvision import datasets, transforms
from torch.profiler import profile, record_function, ProfilerActivity

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
num_batches = 100

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                       std=[0.2675, 0.2565, 0.2761])
])

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(root='./data', train=False, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

# Load model
model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models",
    "cifar100_shufflenetv2_x1_0",
    pretrained=True
).to(device).eval()

print("shufflenetV2 x1.0 CIFAR-100 Profiling")
print("=" * 60)
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"Batches to profile: {num_batches}\n")

# Profile with unified context
model.eval()
with torch.no_grad():
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= num_batches:
                break

            images = images.to(device)

            with record_function(f"batch_{batch_idx}"):
                outputs = model(images)

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
    with open("shufflenet_boundary.txt", "w") as bf:
        for i in range(len(boundaries) - 1):
            start = boundaries[i] + 1 if i > 0 else boundaries[i]
            end = boundaries[i + 1]
            bf.write(f"{start},{end}\n")

    # Print summary
    print("\nProfiling kernel boundaries...")
    print(f"Batches to profile: {num_batches}\n")

    for batch_idx, batch_key in enumerate(batch_keys):
        kernels = kernel_order_by_batch[batch_key]
        start = boundaries[batch_idx] + 1 if batch_idx > 0 else boundaries[batch_idx]
        end = boundaries[batch_idx + 1]
        print(f"  Batch {batch_idx:3d}: {start:6d},{end:6d} ({len(kernels):4d} kernels)")

    print(f"\n✓ Kernel boundaries saved to: shufflenet_boundary.txt")
    print(f"  Total batches: {len(batch_keys)}")
    print(f"  Total kernels: {global_kernel_counter}")
    print(f"  Avg kernels/batch: {global_kernel_counter / len(batch_keys):.1f}\n")
