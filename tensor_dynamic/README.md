# Tensor Dynamic — HMMA Fault Injection Tool

A dynamic fault injection tool for CUDA Tensor Core (HMMA) instructions using NVBit binary instrumentation. Profiles and injects faults in a single pass.

## Overview

`tensor_dynamic` performs runtime profiling and fault injection on Half-precision Matrix Multiply Accumulate (HMMA) instructions in CUDA kernels. It enables controlled fault injection experiments by:

- **Dynamic Profiling**: Profiles HMMA instruction execution and records thread indices at runtime
- **Frequency-Based Targeting**: Selects specific kernels within frequency ranges for injection
- **Instruction-Level Control**: Targets specific HMMA instructions within a kernel
- **Per-Thread Injection**: Injects faults into specific thread executions with configurable multipliers

## Key Features

- **Single-Pass Operation**: Profiles and injects faults in a single kernel execution
- **Runtime Configuration**: All parameters configurable via environment variables
- **Frequency Range Targeting**: Uses `boundary.txt` to divide kernels into frequency ranges
- **Selective Kernel Injection**: Choose which HMMA kernel to inject within each range
- **Instruction Position Control**: Select which HMMA instruction to target (1st, 2nd, etc.)
- **Thread Limiting**: Configure how many thread executions to record and inject
- **FP16 Corruption**: Separate multipliers for low and high FP16 values in each register

## Building

```bash
cd tensor_dynamic/
make
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_THREADS_RECORD` | 50 | Number of thread executions to profile and inject |
| `INJECTION_MODE` | 2 | Injection mode: 0=bit-flip, 1=FP16 multiplication, 2=F32 multiplication |
| `BIT_POSITION` | 0 | Bit position for bit-flip mode (0-31) |
| `CORRUPT_MULT_LOW` | 50 | Multiplication factor for low FP16 half (mode 1 only) |
| `CORRUPT_MULT_HIGH` | 0 | Multiplication factor for high FP16 half (mode 1 only) |
| `CORRUPT_MULT_F32` | 50 | Multiplication factor applied as F32 to full 32-bit register (mode 2 only) |
| `TARGET_KERNEL_POS` | 2 | Which HMMA kernel to target in each boundary range (1-indexed) |
| `TARGET_INSTR_LIST` | (none) | Comma-separated 1-indexed HMMA instruction positions to inject (e.g. `"1,3,5"`) |
| `BOUNDARY_PATH` | `boundary.txt` | Path to the frequency boundary ranges file |
| `TOOL_VERBOSE` | 0 | Enable verbose output (0 or 1) |
| `MANGLED_NAMES` | 1 | Print kernel names mangled (1) or demangled (0) |

## Example

```bash
NUM_THREADS_RECORD=500 INJECTION_MODE=2 \
CORRUPT_MULT_F32=1000 TARGET_KERNEL_POS=1 \
TARGET_INSTR_LIST="1,3" \
BOUNDARY_PATH=path/to/boundary.txt \
LD_PRELOAD=tensor_dynamic/tensor_dynamic.so \
python3 your_eval_script.py
```
