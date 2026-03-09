// Device-side fault injection function with dynamic targeting
// Records injection info in structure for host-side printing

#include <stdio.h>
#include <cuda_fp16.h>
#include "nvbit_reg_rw.h"
#include "injection_structs.h"

/// Perform the replacement of register values as injection
extern "C" __device__ __noinline__ void hmma_replace(
    int pred,
    int reg_dst_num,
    int kernel_id,
    int instr_idx,
    uint64_t ptarget_list,
    uint64_t precord,
    int mult_low,
    int mult_high,
    int mult_f32,
    int injection_mode,
    int bit_position
) {
    InjectionRecord* record = (InjectionRecord*)precord;
    InjectionTargetList* target_list = (InjectionTargetList*)ptarget_list;

        // Count every call to this function
    atomicAdd(&record->num_checks, 1);

    if (!pred) {
        return;
    }

        // Check if current thread matches ANY target
    bool is_target = false;
    for (int i = 0; i < target_list->num_targets; i++) {
        InjectionTarget* target = &target_list->targets[i];
        if (blockIdx.x == target->target_block_x &&
            blockIdx.y == target->target_block_y &&
            blockIdx.z == target->target_block_z &&
            threadIdx.x == target->target_thread_x &&
            threadIdx.y == target->target_thread_y &&
            threadIdx.z == target->target_thread_z &&
            instr_idx == target->target_instr) {
            is_target = true;
            break;
        }
    }

    if (!is_target) {
        return;
    }

        // Check if this (block, thread, instr) has already been injected
    for (int i = 0; i < record->num_injected_locations; i++) {
        InjectedLocation* loc = &record->injected_locations[i];
        if (loc->block_x == blockIdx.x && loc->block_y == blockIdx.y && loc->block_z == blockIdx.z &&
            loc->thread_x == threadIdx.x && loc->thread_y == threadIdx.y && loc->thread_z == threadIdx.z &&
            loc->instr_idx == instr_idx) {
                        // Already injected this location, skip
            return;
        }
    }

        // Mark this location as injected
    int loc_idx = atomicAdd(&record->num_injected_locations, 1);
    if (loc_idx < MAX_INJECTED_TRACKING) {
        record->injected_locations[loc_idx].block_x = blockIdx.x;
        record->injected_locations[loc_idx].block_y = blockIdx.y;
        record->injected_locations[loc_idx].block_z = blockIdx.z;
        record->injected_locations[loc_idx].thread_x = threadIdx.x;
        record->injected_locations[loc_idx].thread_y = threadIdx.y;
        record->injected_locations[loc_idx].thread_z = threadIdx.z;
        record->injected_locations[loc_idx].instr_idx = instr_idx;
    }

        // Read the current destination register value
    int raw_bits = nvbit_read_reg(reg_dst_num);

        // HMMA writes TWO FP16 values per 32-bit register - corrupt BOTH
    half original_half_low, original_half_high;
    *reinterpret_cast<unsigned short*>(&original_half_low) = (unsigned short)(raw_bits & 0xFFFF);
    *reinterpret_cast<unsigned short*>(&original_half_high) = (unsigned short)((raw_bits >> 16) & 0xFFFF);

    float original_value_low = __half2float(original_half_low);
    float original_value_high = __half2float(original_half_high);

        // FAULT INJECTION: mode-based injection
    float corrupted_value_low = original_value_low;
    float corrupted_value_high = original_value_high;
    int new_raw_bits;

    if (injection_mode == 0) {
                // Mode 0: Single bit-flip - flip specified bit in the 32-bit register
        unsigned int flip_mask = ((unsigned int)1 << (bit_position % 32));
        new_raw_bits = raw_bits ^ flip_mask;

                // Extract F32 values for recording
        original_value_low = __int_as_float(raw_bits);
        corrupted_value_low = __int_as_float(new_raw_bits);

    } else if (injection_mode == 1) {
                // Mode 1: Error multiplication - multiply BOTH FP16 values
        corrupted_value_low = original_value_low * mult_low;
        corrupted_value_high = original_value_high * mult_high;

                // Clamp to FP16 range to avoid NaN/Inf
        if (corrupted_value_low > 65504.0f) corrupted_value_low = 65504.0f;
        if (corrupted_value_low < -65504.0f) corrupted_value_low = -65504.0f;
        if (corrupted_value_high > 65504.0f) corrupted_value_high = 65504.0f;
        if (corrupted_value_high < -65504.0f) corrupted_value_high = -65504.0f;

                // Convert back to FP16
        half corrupted_half_low = __float2half(corrupted_value_low);
        half corrupted_half_high = __float2half(corrupted_value_high);
        unsigned short corrupted_bits_low = *reinterpret_cast<unsigned short*>(&corrupted_half_low);
        unsigned short corrupted_bits_high = *reinterpret_cast<unsigned short*>(&corrupted_half_high);
        new_raw_bits = (corrupted_bits_high << 16) | corrupted_bits_low;

    } else if (injection_mode == 2) {
                // Mode 2: Error multiplication for F32 values
        float original_f32 = *reinterpret_cast<float*>(&raw_bits);
        float corrupted_f32 = original_f32 * mult_f32;

                // Clamp to F32 range to avoid NaN/Inf
        if (corrupted_f32 > 3.4028235e38f) corrupted_f32 = 3.4028235e38f;
        if (corrupted_f32 < -3.4028235e38f) corrupted_f32 = -3.4028235e38f;

        new_raw_bits = *reinterpret_cast<int*>(&corrupted_f32);
        corrupted_value_low = corrupted_f32;
        original_value_low = original_f32;
    }

        // Write the corrupted value back to the register
    nvbit_write_reg(reg_dst_num, new_raw_bits);

        // Record this injection in the array
    int idx = atomicAdd(&record->num_actual_injections, 1);
    if (idx < MAX_INJECTION_RECORDS) {
        record->details[idx].kernel_id = kernel_id;
        record->details[idx].instr_idx = instr_idx;
        record->details[idx].block_x = blockIdx.x;
        record->details[idx].block_y = blockIdx.y;
        record->details[idx].block_z = blockIdx.z;
        record->details[idx].thread_x = threadIdx.x;
        record->details[idx].thread_y = threadIdx.y;
        record->details[idx].thread_z = threadIdx.z;
        record->details[idx].reg_num = reg_dst_num;
        record->details[idx].original_raw_bits = raw_bits;
        record->details[idx].corrupted_raw_bits = new_raw_bits;
        record->details[idx].original_value_low = original_value_low;
        record->details[idx].corrupted_value_low = corrupted_value_low;
        record->details[idx].original_value_high = original_value_high;
        record->details[idx].corrupted_value_high = corrupted_value_high;
    }
}
