#ifndef INJECTION_STRUCTS_H
#define INJECTION_STRUCTS_H

#include <stdint.h>
#include <vector>
#include <string>

#ifndef MAX_TARGETS
#define MAX_TARGETS 1000
#endif

// Device-side profiling structure
struct TensorRangeProfile {
    uint32_t block_x_min, block_x_max;
    uint32_t block_y_min, block_y_max;
    uint32_t block_z_min, block_z_max;
    uint32_t thread_x_min, thread_x_max;
    uint32_t thread_y_min, thread_y_max;
    uint32_t thread_z_min, thread_z_max;
};

// Host-side profiling structure
struct KernelProfile {
    std::string kernel_name;
    uint32_t block_x_min, block_x_max;
    uint32_t block_y_min, block_y_max;
    uint32_t block_z_min, block_z_max;
    uint32_t thread_x_min, thread_x_max;
    uint32_t thread_y_min, thread_y_max;
    uint32_t thread_z_min, thread_z_max;
    std::vector<uint32_t> target_instr_indices;
    bool has_target_instr;
};

// Target location for fault injection
struct InjectionTarget {
    int target_block_x, target_block_y, target_block_z;
    int target_thread_x, target_thread_y, target_thread_z;
    uint32_t target_instr;
};

#endif // INJECTION_STRUCTS_H
