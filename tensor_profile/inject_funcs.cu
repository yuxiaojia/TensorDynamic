// Device-side fault injection function with dynamic targeting
// Records injection info in structure for host-side printing

#include <stdio.h>
#include <cuda_fp16.h>
#include "nvbit_reg_rw.h"
#include "injection_structs.h"

// All structures are now defined in injection_structs.h

// Profile HMMA execution ranges
extern "C" __device__ __noinline__ void profile_hmma(
    int pred,
    uint64_t pprofile
) {
    if (!pred) {
        return;
    }

    TensorRangeProfile* profile = (TensorRangeProfile*)pprofile;

        // All active threads record their IDs
    atomicMin(&profile->thread_x_min, threadIdx.x);
    atomicMax(&profile->thread_x_max, threadIdx.x);
    atomicMin(&profile->thread_y_min, threadIdx.y);
    atomicMax(&profile->thread_y_max, threadIdx.y);
    atomicMin(&profile->thread_z_min, threadIdx.z);
    atomicMax(&profile->thread_z_max, threadIdx.z);

        // Only first thread of each warp records block IDs
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int first_laneid = __ffs(active_mask) - 1;
    const int laneid = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) & 0x1F;

    if (first_laneid == laneid) {
        atomicMin(&profile->block_x_min, blockIdx.x);
        atomicMax(&profile->block_x_max, blockIdx.x);
        atomicMin(&profile->block_y_min, blockIdx.y);
        atomicMax(&profile->block_y_max, blockIdx.y);
        atomicMin(&profile->block_z_min, blockIdx.z);
        atomicMax(&profile->block_z_max, blockIdx.z);
    }
}