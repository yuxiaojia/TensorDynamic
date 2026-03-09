#ifndef INJECTION_STRUCTS_H
#define INJECTION_STRUCTS_H

#include <stdint.h>
#include <vector>
#include <string>

#ifndef MAX_TARGETS
#define MAX_TARGETS 1000
#endif
#ifndef MAX_RECORDED_THREADS
#define MAX_RECORDED_THREADS 1000
#endif
#ifndef MAX_INJECTION_RECORDS
#define MAX_INJECTION_RECORDS 500
#endif
#ifndef MAX_INJECTED_TRACKING
#define MAX_INJECTED_TRACKING 1000
#endif

// Structure to store actual thread execution data
struct ThreadExecution {
    uint32_t block_x, block_y, block_z;
    uint32_t thread_x, thread_y, thread_z;
};

// Device-side profiling structure - stores actual thread indices
struct TensorRangeProfile {
    // Array of actual thread indices that executed target instruction
    ThreadExecution executed_threads[MAX_RECORDED_THREADS];

    // Atomic counter for number of threads recorded
    uint32_t num_recorded_threads;

    // Total executions seen (may exceed num_recorded_threads)
    uint32_t total_executions;
};

// Host-side profiling structure
struct KernelProfile {
    std::string kernel_name;
    std::vector<uint32_t> target_instr_indices;
    bool has_target_instr;

    // Store actual thread executions collected during profiling
    std::vector<ThreadExecution> recorded_threads;
    uint32_t total_executions;
};

// Target location for fault injection
struct InjectionTarget {
    int target_block_x, target_block_y, target_block_z;
    int target_thread_x, target_thread_y, target_thread_z;
    uint32_t target_instr;
};

// List of injection targets (managed memory)
struct InjectionTargetList {
    int num_targets;
    InjectionTarget targets[MAX_TARGETS];
};

// Single injection detail
struct InjectionDetail {
    int kernel_id;
    int instr_idx;
    int block_x, block_y, block_z;
    int thread_x, thread_y, thread_z;
    int reg_num;
    unsigned int original_raw_bits;
    unsigned int corrupted_raw_bits;
    float original_value_low;
    float corrupted_value_low;
    float original_value_high;
    float corrupted_value_high;
};

// Tracking structure for already-injected locations
struct InjectedLocation {
    int block_x, block_y, block_z;
    int thread_x, thread_y, thread_z;
    int instr_idx;
};

// Record of all injection results
struct InjectionRecord {
    int num_checks;   // How many times injection function was called
    int num_actual_injections;  // How many injections were performed
    InjectionDetail details[MAX_INJECTION_RECORDS];

    // Track which locations have been injected to prevent duplicates
    int num_injected_locations;
    InjectedLocation injected_locations[MAX_INJECTED_TRACKING];
};

#endif // INJECTION_STRUCTS_H