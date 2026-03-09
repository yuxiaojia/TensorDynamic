// HMMA Tensor Core Profiler
// Profiles HMMA instructions and generates injection targets

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"
#include "injection_structs.h"

// Profiling data structures
std::unordered_map<int, KernelProfile> kernel_profiles;  // kernel_id -> profile

// Store kernel context and function for retrieving names later
struct KernelInfo {
    CUcontext ctx;
    CUfunction func;
};
std::unordered_map<int, KernelInfo> kernel_info_map;  // kernel_id -> ctx/func

// Use managed memory for device-accessible profiling data
__managed__ TensorRangeProfile current_profile;

// Use managed memory for injection targets and records
__managed__ InjectionTargetList current_target_list;
__managed__ InjectionRecord injection_record;
__managed__ bool start_instrument_flag;

// Global control
int kernel_count = 0;
int corruption_mult_low = 0;   // Corruption multiplier for low FP16 value (mode 1)
int corruption_mult_high = 0;  // Corruption multiplier for high FP16 value (mode 1)
int corruption_mult_f32 = 0;   // Corruption multiplier for F32 value (mode 2)
int injection_mode = 0;         // Injection mode: 0=single bit-flip, 1=error multiplication
int bit_position = 0;           // Bit position for single bit-flip mode (0-31)
bool mangled = false;
int verbose = 1;
int num_threads_to_record = 100;     // Configurable number of thread executions to record during profiling
int target_kernel_position = 1;      // Which kernel to select for injection (1=first, 2=second, etc.)
std::string target_instr_list_str = "1"; // Comma-separated 1-indexed HMMA positions to inject (e.g. "1,3,5")
std::vector<int> target_instr_list;       // Parsed from target_instr_list_str
std::string boundary_path = "boundary.txt";
std::string target_opcode = "HMMA";  // Opcode prefix to target (configurable via TARGET_OPCODE)

// Frequency-based targeting
struct FrequencyRange {
    int start;
    int end;
};
std::vector<FrequencyRange> frequency_ranges;
int num_frequency_ranges = 0;
std::map<int, int> range_first_hmma_kernel;  // range_id -> first kernel_id with HMMA
std::map<int, int> range_hmma_counter;  // range_id -> count of HMMA kernels seen in this range

// Get frequency range ID for a given kernel ID
int get_frequency_range(int kernel_id) {
    for (int i = 0; i < num_frequency_ranges; i++) {
        int range_start = frequency_ranges[i].start;
        int range_end = frequency_ranges[i].end;

        if (kernel_id >= range_start && kernel_id <= range_end) {
            return i;
        }
    }
    return -1;
}

void nvbit_at_init() {
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1, "Print kernel names mangled or not");
    GET_VAR_INT(num_threads_to_record, "NUM_THREADS_RECORD", 50, "Number of thread executions to record during profiling");
    GET_VAR_INT(corruption_mult_low, "CORRUPT_MULT_LOW", 50, "Corruption multiplier for low FP16 value (mode 1)");
    GET_VAR_INT(corruption_mult_high, "CORRUPT_MULT_HIGH", 0, "Corruption multiplier for high FP16 value (mode 1)");
    GET_VAR_INT(corruption_mult_f32, "CORRUPT_MULT_F32", 50, "Corruption multiplier for F32 value (mode 2)");
    GET_VAR_INT(injection_mode, "INJECTION_MODE", 2, "Injection mode: 0=single bit-flip, 1=F16error multiplication, 2=F32 error multiplication");
    GET_VAR_INT(bit_position, "BIT_POSITION", 0, "Bit position for single bit-flip mode (0-31)");
    GET_VAR_INT(target_kernel_position, "TARGET_KERNEL_POS", 2, "Which HMMA kernel to select for injection");
    GET_VAR_STR(target_instr_list_str, "TARGET_INSTR_LIST", "Comma-separated 1-indexed HMMA positions to inject (e.g. '1,3,5')");
    GET_VAR_STR(boundary_path, "BOUNDARY_PATH", "Path to boundary file (start,end per line)");

        // Parse TARGET_INSTR_LIST into vector
    {
        std::stringstream ss(target_instr_list_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty())
                target_instr_list.push_back(std::stoi(token));
        }
    }
    GET_VAR_STR(target_opcode, "TARGET_OPCODE", "Opcode prefix to target (default: HMMA)");

    printf("========================================\n");
    printf("HMMA DYNAMIC PROFILER + INJECTOR\n");
    printf("========================================\n");
    printf("Recording first %d thread executions per kernel\n", num_threads_to_record);

    const char* mode_names[] = {"Single Bit-Flip", "FP16 Error Mult", "F32 Error Mult"};
    printf("Injection mode: %d (%s)\n", injection_mode,
           injection_mode >= 0 && injection_mode <= 2 ? mode_names[injection_mode] : "Unknown");

    if (injection_mode == 0) {
        printf("Bit position: %d\n", bit_position);
    } else if (injection_mode == 1) {
        printf("Corruption multipliers: low=%d, high=%d\n", corruption_mult_low, corruption_mult_high);
    } else if (injection_mode == 2) {
        printf("Corruption multiplier (F32): %d\n", corruption_mult_f32);
    }
    printf("Target kernel position: %d\n", target_kernel_position);
    printf("Target HMMA indices: ");
    for (size_t i = 0; i < target_instr_list.size(); i++) {
        printf("%d", target_instr_list[i]);
        if (i + 1 < target_instr_list.size()) printf(",");
    }
    printf("\n");
    printf("Threads per HMMA: %d  => Total errors: %d\n", num_threads_to_record, (int)target_instr_list.size() * num_threads_to_record);

        // Load frequency ranges from boundary file (format: start,end per line)
    FILE* boundary_fp = fopen(boundary_path.c_str(), "r");
    if (!boundary_fp) {
        printf("ERROR: Boundary file not found: %s\n", boundary_path.c_str());
        printf("Please create a boundary file with frequency ranges (format: start,end per line).\n");
        exit(1);
    }

    int start, end;
    while (fscanf(boundary_fp, "%d,%d", &start, &end) == 2) {
        FrequencyRange range;
        range.start = start;
        range.end = end;
        frequency_ranges.push_back(range);
    }
    fclose(boundary_fp);

    num_frequency_ranges = frequency_ranges.size();

    // Initialize: no ranges have found their first HMMA kernel yet
    for (int i = 0; i < num_frequency_ranges; i++) {
        range_first_hmma_kernel[i] = -1;
        range_hmma_counter[i] = 0;
    }

    printf("Loaded %d frequency ranges from %s\n", num_frequency_ranges, boundary_path.c_str());
    if (num_frequency_ranges > 0) {
        printf("Ranges: ");
        for (int i = 0; i < num_frequency_ranges; i++) {
            printf("[%d,%d]", frequency_ranges[i].start, frequency_ranges[i].end);
            if (i < num_frequency_ranges - 1) printf(", ");
        }
        printf("\n");
    }
}

// Check if a function has HMMA instructions
bool has_target_instructions(CUcontext ctx, CUfunction func) {
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, func);
    for (size_t idx = 0; idx < instrs.size(); idx++) {
        auto instr = instrs[idx];
        std::string opcode = instr->getOpcode();
        if (opcode.find(target_opcode) != std::string::npos) {
            return true;
        }
    }
    return false;
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func,
                                   int gridDimX, int gridDimY, int gridDimZ,
                                   int blockDimX, int blockDimY, int blockDimZ) {

    int current_kernel = kernel_count + 1;

    std::vector<CUfunction> related_functions;
    related_functions.push_back(func);

    for (auto f : related_functions) {
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        if (verbose) {
            printf("Kernel %d: %s - %ld instructions\n",
                   current_kernel, nvbit_get_func_name(ctx, f, mangled), instrs.size());
        }

        KernelProfile profile;
        profile.has_target_instr = false;
        int hmma_count = 0;  // Track which HMMA instruction we're on (1-indexed)

        // Instrument HMMA instructions
        for (size_t idx = 0; idx < instrs.size(); idx++) {
            auto instr = instrs[idx];
            std::string opcode = instr->getOpcode();

            if (opcode.find(target_opcode) != std::string::npos) {
                profile.has_target_instr = true;
                profile.target_instr_indices.push_back(idx);
                hmma_count++;  // Increment for each HMMA found

                                // Get operand 0 (destination register) for injection
                int num_opnds = instr->getNumOperands();
                if (num_opnds == 0) continue;

                const InstrType::operand_t *op0 = instr->getOperand(0);
                if (op0->type != InstrType::OperandType::REG) continue;

                if (verbose) {
                    printf("  Found HMMA #%d at idx %zu, reg R%d\n", hmma_count, idx, op0->u.reg.num);
                }

                                // Insert profiling instrumentation BEFORE
                nvbit_insert_call(instr, "profile_hmma", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr, current_kernel);

                                // Pass managed profile structure pointer
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&current_profile);

                                // Pass target list pointer
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&current_target_list);

                                // Pass runtime configurable limit
                nvbit_add_call_arg_const_val32(instr, num_threads_to_record);

                                // Pass pointer to start_instrument flag (managed memory)
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&start_instrument_flag);

                                // Insert fault injection call for selected HMMA indices
                if (std::find(target_instr_list.begin(), target_instr_list.end(), hmma_count) != target_instr_list.end()) {
                    if (verbose) {
                        printf("  -> Instrumenting HMMA #%d for injection\n", hmma_count);
                    }
                    nvbit_insert_call(instr, "hmma_replace", IPOINT_AFTER);
                    nvbit_add_call_arg_guard_pred_val(instr);
                    nvbit_add_call_arg_const_val32(instr, op0->u.reg.num);
                    nvbit_add_call_arg_const_val32(instr, current_kernel);
                    nvbit_add_call_arg_const_val32(instr, idx);

                                        // Pass target list pointer
                    nvbit_add_call_arg_const_val64(instr, (uint64_t)&current_target_list);

                                        // Pass injection record pointer
                    nvbit_add_call_arg_const_val64(instr, (uint64_t)&injection_record);

                                        // Pass corruption multipliers, injection mode, and bit position
                    nvbit_add_call_arg_const_val32(instr, corruption_mult_low);
                    nvbit_add_call_arg_const_val32(instr, corruption_mult_high);
                    nvbit_add_call_arg_const_val32(instr, corruption_mult_f32);
                    nvbit_add_call_arg_const_val32(instr, injection_mode);
                    nvbit_add_call_arg_const_val32(instr, bit_position);
                }
            }
        }

        if (profile.has_target_instr) {
            profile.kernel_name = nvbit_get_func_name(ctx, f, mangled);
            kernel_profiles[current_kernel] = profile;
            kernel_info_map[current_kernel] = {ctx, f};
            printf("Kernel %d has %zu HMMA instructions - INSTRUMENTED\n",
                   current_kernel, profile.target_instr_indices.size());
            for (int t : target_instr_list) {
                if (t > (int)profile.target_instr_indices.size()) {
                    printf("  WARNING: TARGET_INSTR_LIST includes #%d but kernel only has %zu HMMAs - skipping\n",
                           t, profile.target_instr_indices.size());
                }
            }
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {

        CUfunction func;
        int gridDimX, gridDimY, gridDimZ;
        int blockDimX, blockDimY, blockDimZ;

        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
            gridDimX = p->config->gridDimX;
            gridDimY = p->config->gridDimY;
            gridDimZ = p->config->gridDimZ;
            blockDimX = p->config->blockDimX;
            blockDimY = p->config->blockDimY;
            blockDimZ = p->config->blockDimZ;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
            gridDimX = p->gridDimX;
            gridDimY = p->gridDimY;
            gridDimZ = p->gridDimZ;
            blockDimX = p->blockDimX;
            blockDimY = p->blockDimY;
            blockDimZ = p->blockDimZ;
        }

        if (!is_exit) {
            int current_kernel = kernel_count + 1;


                        // Initialize profiling structures before each NEW kernel
            current_profile.num_recorded_threads = 0;
            current_profile.total_executions = 0;
            memset(current_profile.executed_threads, 0, sizeof(current_profile.executed_threads));

            injection_record.num_checks = 0;
            injection_record.num_actual_injections = 0;
            injection_record.num_injected_locations = 0;
            current_target_list.num_targets = 0;
            start_instrument_flag = false;  // Reset for each kernel

            if (has_target_instructions(ctx, func) == true) {
                                // Make selection decision - only for NEW functions
                bool should_instrument = false;
                int freq_range = get_frequency_range(current_kernel);

                if (freq_range >= 0) {
                                        // Increment counter for this range - only for NEW HMMA functions
                    range_hmma_counter[freq_range]++;
                    int position_in_range = range_hmma_counter[freq_range];

                    if (position_in_range == target_kernel_position) {
                        should_instrument = true;
                        range_first_hmma_kernel[freq_range] = current_kernel;
                    } else {
                        printf("Kernel %d (Range %d) - SKIPPING (target is #%d)\n",
                            current_kernel, freq_range, target_kernel_position);
                    }
                }

                                // Only instrument if selected
                if (should_instrument) {
                    printf("*** Kernel %d (Range %d) - SELECTED for injection ***\n",
                            current_kernel, freq_range);
                    start_instrument_flag = true;  // Enable profiling for this kernel
                    instrument_function_if_needed(ctx, func, gridDimX, gridDimY, gridDimZ,
                                                blockDimX, blockDimY, blockDimZ);
                    nvbit_enable_instrumented(ctx, func, true);
                } else {
                    nvbit_enable_instrumented(ctx, func, false);
                }
            }

            kernel_count++;
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());

            if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel) {
            } else if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
            }
            
                        // Copy managed profiling data to host kernel profile
            if (current_profile.num_recorded_threads > 0 && kernel_profiles.find(kernel_count) != kernel_profiles.end()) {
                KernelProfile& host_prof = kernel_profiles[kernel_count];
                host_prof.total_executions = current_profile.total_executions;

                                // Copy recorded thread executions
                uint32_t num_to_copy = (current_profile.num_recorded_threads < MAX_RECORDED_THREADS)
                                      ? current_profile.num_recorded_threads
                                      : MAX_RECORDED_THREADS;

                host_prof.recorded_threads.clear();
                for (uint32_t i = 0; i < num_to_copy; i++) {
                    host_prof.recorded_threads.push_back(current_profile.executed_threads[i]);
                }

                if (verbose) {
                    printf("Kernel %d: Recorded %u thread executions (total: %u)\n",
                           kernel_count, num_to_copy, current_profile.total_executions);
                    if (verbose > 1 && num_to_copy > 0) {
                        printf("  First few threads:\n");
                        for (uint32_t i = 0; i < std::min(num_to_copy, 5u); i++) {
                            printf("    [%u] Block:(%u,%u,%u) Thread:(%u,%u,%u)\n", i,
                                   current_profile.executed_threads[i].block_x,
                                   current_profile.executed_threads[i].block_y,
                                   current_profile.executed_threads[i].block_z,
                                   current_profile.executed_threads[i].thread_x,
                                   current_profile.executed_threads[i].thread_y,
                                   current_profile.executed_threads[i].thread_z);
                        }
                    }
                }
            }

                        // Print injection report if any injections were attempted
            if (current_target_list.num_targets > 0) {
                printf("\n========== INJECTION REPORT for Kernel %d ==========\n", kernel_count);
                printf("  Total injection checks: %d\n", injection_record.num_checks);
                printf("  Number of targets configured: %d\n", current_target_list.num_targets);
                printf("  ACTUAL INJECTIONS PERFORMED: %d\n", injection_record.num_actual_injections);

                if (injection_record.num_actual_injections > 0) {
                    const char* mode_names[] = {"Single Bit-Flip", "FP16 Error Mult", "F32 Error Mult"};
                    printf("\n  INJECTIONS (Mode: %s):\n",
                           injection_mode >= 0 && injection_mode <= 2 ? mode_names[injection_mode] : "Unknown");

                    int num_to_print = injection_record.num_actual_injections < MAX_INJECTION_RECORDS ?
                                       injection_record.num_actual_injections : MAX_INJECTION_RECORDS;
                    for (int i = 0; i < num_to_print && i < 10; i++) {
                        auto& detail = injection_record.details[i];
                        printf("    [%d] Instr:%d Block:(%d,%d,%d) Thread:(%d,%d,%d) Reg:R%d\n",
                               i, detail.instr_idx,
                               detail.block_x, detail.block_y, detail.block_z,
                               detail.thread_x, detail.thread_y, detail.thread_z,
                               detail.reg_num);
                        printf("        Raw:  0x%08X -> 0x%08X (XOR: 0x%08X)\n",
                               detail.original_raw_bits, detail.corrupted_raw_bits,
                               detail.original_raw_bits ^ detail.corrupted_raw_bits);

                        if (injection_mode == 0) {
                                                        // Mode 0: Single bit-flip - show which bit was flipped
                            unsigned int xor_bits = detail.original_raw_bits ^ detail.corrupted_raw_bits;
                            int flipped_bit = -1;
                            for (int b = 0; b < 32; b++) {
                                if (xor_bits & (1u << b)) {
                                    flipped_bit = b;
                                    break;
                                }
                            }
                            printf("        Bit flipped: %d\n", flipped_bit);
                            printf("        F32:  %.6f -> %.6f\n",
                                   detail.original_value_low, detail.corrupted_value_low);

                        } else if (injection_mode == 1) {
                                                        // Mode 1: FP16 error multiplication - show both FP16 values
                            printf("        Low:  %.6f -> %.6f\n",
                                   detail.original_value_low, detail.corrupted_value_low);
                            printf("        High: %.6f -> %.6f\n",
                                   detail.original_value_high, detail.corrupted_value_high);

                        } else if (injection_mode == 2) {
                                                        // Mode 2: F32 error multiplication - show single F32 value
                            printf("        F32:  %.6f -> %.6f\n",
                                   detail.original_value_low, detail.corrupted_value_low);
                        }
                    }
                    if (num_to_print > 10) {
                        printf("    ... (%d more injections)\n", num_to_print - 10);
                    }
                } else {
                    printf("  ✗ NO FAULT INJECTED\n");
                }
                printf("====================================================\n\n");
            }
        }
    }
}

void nvbit_at_term() {
    printf("\n========================================\n");
    printf("PROFILING COMPLETE\n");
    printf("========================================\n");
    printf("Total kernels: %d\n", kernel_count);
    printf("Kernels with HMMA: %zu\n", kernel_profiles.size());

    // // Print selected kernels per frequency range
    // printf("\n=== SELECTED KERNELS PER FREQUENCY RANGE ===\n");
    // for (int i = 0; i < num_frequency_ranges; i++) {
    //     if (range_first_hmma_kernel[i] != -1) {
    //         int kernel_id = range_first_hmma_kernel[i];
    //         printf("Range %d: Kernel %d (first HMMA)\n", i, kernel_id);
    //         if (kernel_profiles.find(kernel_id) != kernel_profiles.end()) {
    //             printf("         Name: %s\n", kernel_profiles[kernel_id].kernel_name.c_str());
    //             printf("         Injected threads: %zu\n", kernel_profiles[kernel_id].recorded_threads.size());
    //         }
    //     } else {
    //         printf("Range %d: No HMMA kernel found\n", i);
    //     }
    // }

        // Save kernel info to separate file (sorted by kernel ID)
    FILE* fp_info = fopen("kernel_info.txt", "w");
    if (fp_info) {
        // Create sorted vector of kernel IDs
        std::vector<int> sorted_kernel_ids;
        for (const auto& kv : kernel_profiles) {
            sorted_kernel_ids.push_back(kv.first);
        }
        std::sort(sorted_kernel_ids.begin(), sorted_kernel_ids.end());

        // Write in sorted order with empty lines between kernels
        for (size_t i = 0; i < sorted_kernel_ids.size(); i++) {
            int kernel_id = sorted_kernel_ids[i];
            const auto& profile = kernel_profiles[kernel_id];

            fprintf(fp_info, "KERNEL %d %d\n", kernel_id, profile.has_target_instr ? 1 : 0);
            fprintf(fp_info, "NAME %s\n", profile.kernel_name.c_str());
            fprintf(fp_info, "TOTAL_EXECUTIONS %u\n", profile.total_executions);
            fprintf(fp_info, "RECORDED_THREADS %zu\n", profile.recorded_threads.size());

            // // Write actual thread execution data
            // fprintf(fp_info, "THREAD_DATA");
            // for (const auto& thread : profile.recorded_threads) {
            //     fprintf(fp_info, " %u,%u,%u,%u,%u,%u",
            //            thread.block_x, thread.block_y, thread.block_z,
            //            thread.thread_x, thread.thread_y, thread.thread_z);
            // }
            // fprintf(fp_info, "\n");

            fprintf(fp_info, "INSTRS %zu", profile.target_instr_indices.size());
            for (uint32_t idx : profile.target_instr_indices) {
                fprintf(fp_info, " %u", idx);
            }
            fprintf(fp_info, "\n");

            // Add empty line between kernels (except after last one)
            if (i < sorted_kernel_ids.size() - 1) {
                fprintf(fp_info, "\n");
            }
        }
        fclose(fp_info);
        printf("Kernel info saved to kernel_info.txt\n");
    }
}