# Tensor Profile — HMMA Profiling Tool

HMMA Tensor Core profiling tool for NVBit. Profiles CUDA kernels with HMMA instructions and generates fault injection targets for use with `tensor_injector`.

## Overview

`tensor_profile` performs runtime profiling on Half-precision Matrix Multiply Accumulate (HMMA) instructions in CUDA kernels. It enables controlled fault injection experiments by:

- **Dynamic Profiling**: Identifies all kernels containing HMMA instructions and records thread indices at runtime
- **Frequency-Based Targeting**: Divides kernels into frequency ranges using `boundary.txt` for stratified sampling
- **Instruction-Level Control**: Records exact HMMA instruction indices within each kernel
- **Per-Thread Recording**: Captures block/thread coordinates for each HMMA execution as injection targets

## Key Features

- **Single-Pass Profiling**: Profiles the full application in one run and generates injection targets
- **Runtime Configuration**: All parameters configurable via environment variables
- **Frequency Range Targeting**: Uses `boundary.txt` to divide kernels into frequency ranges
- **Selective Kernel Sampling**: Selects one representative kernel per frequency range
- **Instruction Position Control**: Records which HMMA instruction indices are available per kernel
- **Thread Limiting**: Configure how many injection targets to generate per kernel
- **FP16 Corruption**: Outputs targets compatible with both low and high FP16 register halves

## Building

```bash
cd tensor_profile/
make
```

Override architecture with `make ARCH=sm_80` for Ampere GPUs (default: `sm_90`).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_INJECTIONS` | 50 | Number of injection targets per kernel |
| `TOOL_VERBOSE` | 0 | Enable verbose output (0 or 1) |
| `MANGLED_NAMES` | 1 | Print kernel names mangled (1) or demangled (0) |

## Example

```bash
NUM_INJECTIONS=50 TOOL_VERBOSE=1 \
BOUNDARY_PATH=path/to/boundary.txt \
LD_PRELOAD=tensor_profile/tensor_profile.so \
python3 your_eval_script.py
```
