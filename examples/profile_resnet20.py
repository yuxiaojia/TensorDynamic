import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from contextlib import nullcontext
from torch.profiler import profile, record_function, ProfilerActivity

# -------------------- Setup --------------------
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

C100_MEAN = (0.5071, 0.4867, 0.4408)
C100_STD  = (0.2675, 0.2565, 0.2761)

# (Optional) Let cuDNN pick fast kernels
if USE_CUDA:
    torch.backends.cudnn.benchmark = True

# Test loader (CIFAR-100 test set)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(
        root="./.data", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(C100_MEAN, C100_STD),
        ])
    ),
    batch_size=100,
    shuffle=False,
    num_workers=2,
    pin_memory=USE_CUDA
)

# Load pretrained ResNet-20 for CIFAR-100
model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models",
    "cifar100_resnet20",
    pretrained=True
).to(DEVICE).eval()

@torch.no_grad()
def evaluate_fp16(model, loader, max_batches=None):
    total = correct = 0
    loss_sum = 0.0
    sample_offset = 0

    boundaries = [0]

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if USE_CUDA else nullcontext()

    activities = [ProfilerActivity.CPU]
    if USE_CUDA:
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    ) as prof:

        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_size = x.size(0)
            sample_start = sample_offset
            sample_end = sample_offset + batch_size - 1

            if USE_CUDA:
                torch.cuda.nvtx.mark(f"BATCH {batch_idx} START")

            range_ctx = torch.cuda.nvtx.range(
                f"[BATCH-{batch_idx:03d}] Samples[{sample_start:05d}:{sample_end:05d}]"
            ) if USE_CUDA else nullcontext()

            with range_ctx:
                with record_function(f"batch_{batch_idx:03d}_data_transfer"):
                    x = x.to(DEVICE, non_blocking=True)
                    y = y.to(DEVICE, non_blocking=True)

                with amp_ctx:
                    with record_function(f"batch_{batch_idx:03d}_model_forward"):
                        logits = model(x)
                    with record_function(f"batch_{batch_idx:03d}_compute_loss"):
                        loss = F.cross_entropy(logits, y, reduction="sum")

            loss_sum += loss.item()
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            sample_offset += batch_size

            prof.step()

    # Assign CUDA kernels to batches via name-based matching
    all_events = sorted(prof.events(), key=lambda e: e.time_range.start)

    def get_batch(event):
        name = getattr(event, 'name', '') or ''
        for prefix in ('batch_', '[BATCH-'):
            if prefix in name:
                try:
                    part = name.split(prefix)[1].split('_' if prefix == 'batch_' else ']')[0]
                    return int(part)
                except (IndexError, ValueError):
                    pass
        return None

    # Build parent map
    event_to_parent = {}
    for i, ev in enumerate(all_events):
        for j in range(i - 1, -1, -1):
            p = all_events[j]
            if p.time_range.start <= ev.time_range.start and p.time_range.end >= ev.time_range.end:
                event_to_parent[id(ev)] = p
                break

    def find_batch(ev, visited=None):
        visited = visited or set()
        if id(ev) in visited:
            return None
        visited.add(id(ev))
        b = get_batch(ev)
        if b is not None:
            return b
        parent = event_to_parent.get(id(ev))
        return find_batch(parent, visited) if parent else None

    # Count kernels per batch
    batch_counts = {}
    for ev in all_events:
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.name:
            b = find_batch(ev)
            if b is not None:
                batch_counts[b] = batch_counts.get(b, 0) + 1

    # Build cumulative boundaries
    running = 0
    for b in sorted(batch_counts):
        running += batch_counts[b]
        boundaries.append(running)

    with open("boundary.txt", "w") as f:
        for i in range(len(boundaries) - 1):
            start = boundaries[i] + 1 if i > 0 else boundaries[i]
            end = boundaries[i + 1]
            f.write(f"{start},{end}\n")

    print(f"Boundary saved to 'boundary_resnet20.txt' ({len(boundaries)-1} batches, {running} kernels total)")
    return loss_sum / total, 100.0 * correct / total


if __name__ == "__main__":
    print(f"Running on: {DEVICE}")
    loss, acc = evaluate_fp16(model, test_loader, max_batches=100)
    print(f"Top-1 Accuracy: {acc:.2f}%")
