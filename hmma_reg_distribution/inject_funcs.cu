// Device-side instrumentation function injected after each HMMA instruction.
// Reads the full 32-bit output register value as FP32 and atomically increments
// either the zero or non-zero counter in managed memory.

#include <stdint.h>

#include "nvbit_reg_rw.h"

extern "C" __device__ __noinline__ void count_reg_distribution(
    int reg_num, uint64_t* zero_ptr, uint64_t* nonzero_ptr) {
    // read full 32-bit register value and reinterpret as FP32
    int raw_bits = nvbit_read_reg(reg_num);
    float val = *reinterpret_cast<float*>(&raw_bits);

    if (val == 0.0f)
        atomicAdd((unsigned long long*)zero_ptr, 1);
    else
        atomicAdd((unsigned long long*)nonzero_ptr, 1);
}
