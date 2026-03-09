// HMMA Tensor Core Fault Injector
// Injects faults into HMMA instructions based on profiling data

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"
#include "injection_structs.h"

// Injection targets loaded from file
std::unordered_map<int, std::vector<InjectionTarget>> injection_targets;
std::unordered_set<int> targeted_kernels;  // Kernels selected for injection
std::unordered_map<int, std::string> expected_kernel_names;  // kernel_id -> expected name

// Managed memory for device access
__managed__ InjectionTargetList current_target_list;
__managed__ InjectionRecord injection_record;

// Global control
int kernel_count = 0;
bool mangled = false;
int verbose = 1;
int corruption_mult_low = 10;   // Corruption multiplier for low FP16 value (mode 1)
int corruption_mult_high = 10;  // Corruption multiplier for high FP16 value (mode 1)
int corruption_mult_f32 = 10;   // Corruption multiplier for F32 value (mode 2)
int injection_mode = 0;         // Injection mode: 0=single bit-flip, 1=error multiplication
int bit_position = 0;           // Bit position for single bit-flip mode (0-31)
std::string target_opcode = "HMMA";  // Opcode prefix to target (configurable via TARGET_OPCODE)

void nvbit_at_init() {
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1, "Print kernel names mangled or not");
    GET_VAR_INT(corruption_mult_low, "CORRUPT_MULT_LOW", 10, "Corruption multiplier for low FP16 value (mode 1)");
    GET_VAR_INT(corruption_mult_high, "CORRUPT_MULT_HIGH", 10, "Corruption multiplier for high FP16 value (mode 1)");
    GET_VAR_INT(corruption_mult_f32, "CORRUPT_MULT_F32", 10, "Corruption multiplier for F32 value (mode 2)");
    GET_VAR_INT(injection_mode, "INJECTION_MODE", 0, "Injection mode: 0=single bit-flip, 1=F16 error multiplication, 2=F32 error multiplication");
    GET_VAR_INT(bit_position, "BIT_POSITION", 0, "Bit position for single bit-flip mode (0-31)");
    GET_VAR_STR(target_opcode, "TARGET_OPCODE", "Opcode prefix to target (default: HMMA)");

    printf("========================================\n");
    printf("HMMA FAULT INJECTION MODE\n");
    printf("========================================\n");

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

        // Load injection targets from file
    FILE* fp = fopen("fault_injection_target.txt", "r");
    if (!fp) {
        printf("ERROR: fault_injection_target.txt not found!\n");
        printf("Please run tensor_profile tool first.\n");
        exit(1);
    }

    char line[1024];
    std::string current_kernel_name;
    int current_kernel_id = -1;

    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "TARGETS", 7) == 0) {
            size_t num_targets;
            sscanf(line, "TARGETS %zu", &num_targets);
        } else if (strncmp(line, "KERNEL_NAME", 11) == 0) {
            // Extract kernel name (skip "KERNEL_NAME " prefix and remove trailing newline)
            char* name_start = line + 12;
            size_t len = strlen(name_start);
            if (len > 0 && name_start[len-1] == '\n') {
                name_start[len-1] = '\0';
            }
            current_kernel_name = name_start;
        } else if (strncmp(line, "TARGET", 6) == 0) {
            int kernel_id;
            InjectionTarget target;
            sscanf(line, "TARGET %d %d %d %d %d %d %d %u",
                  &kernel_id,
                  &target.target_block_x, &target.target_block_y, &target.target_block_z,
                  &target.target_thread_x, &target.target_thread_y, &target.target_thread_z,
                  &target.target_instr);
            injection_targets[kernel_id].push_back(target);
            targeted_kernels.insert(kernel_id);

            // Store expected kernel name for this kernel_id
            if (!current_kernel_name.empty() && kernel_id != current_kernel_id) {
                expected_kernel_names[kernel_id] = current_kernel_name;
                current_kernel_id = kernel_id;
            }
        }
    }
    fclose(fp);

        // Count total targets across all kernels
    size_t total_targets = 0;
    for (const auto& kv : injection_targets) {
        total_targets += kv.second.size();
    }

    printf("Loaded %zu injection targets across %zu kernels\n", total_targets, injection_targets.size());
    printf("Loaded %zu kernel names for verification\n", expected_kernel_names.size());
}

std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func,
                                   int gridDimX, int gridDimY, int gridDimZ,
                                   int blockDimX, int blockDimY, int blockDimZ) {

    int current_kernel = kernel_count + 1;

        // Skip kernels not in targeted set
    if (targeted_kernels.find(current_kernel) == targeted_kernels.end()) {
        return;
    }

    std::vector<CUfunction> related_functions;
    related_functions.push_back(func);

    for (auto f : related_functions) {
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
        std::string actual_kernel_name = nvbit_get_func_name(ctx, f, mangled);

                // Verify kernel name matches expected name
        if (expected_kernel_names.find(current_kernel) != expected_kernel_names.end()) {
            std::string expected_name = expected_kernel_names[current_kernel];
            if (actual_kernel_name != expected_name) {
                printf("WARNING: Kernel %d name mismatch!\n", current_kernel);
                printf("  Expected: %s\n", expected_name.c_str());
                printf("  Actual:   %s\n", actual_kernel_name.c_str());
                printf("  Skipping injection for safety.\n");
                return;
            }
        }

        if (verbose) {
            printf("Kernel %d: %s - %ld instructions\n",
                   current_kernel, actual_kernel_name.c_str(), instrs.size());
        }

        for (size_t idx = 0; idx < instrs.size(); idx++) {
            auto instr = instrs[idx];
            std::string opcode = instr->getOpcode();

            if (opcode.find(target_opcode) != std::string::npos) {
                                // Instrument for fault injection
                int num_opnds = instr->getNumOperands();
                if (num_opnds == 0) continue;

                const InstrType::operand_t *op0 = instr->getOperand(0);
                if (op0->type != InstrType::OperandType::REG) continue;

                if (verbose) {
                    printf("  Instrumenting HMMA at idx %zu, reg R%d\n", idx, op0->u.reg.num);
                    printf("    %d targets loaded for this kernel\n", (int)injection_targets[current_kernel].size());
                }

                                // Insert fault injection call AFTER the instruction executes
                nvbit_insert_call(instr, "hmma_replace", IPOINT_AFTER);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr, op0->u.reg.num);
                nvbit_add_call_arg_const_val32(instr, current_kernel);
                nvbit_add_call_arg_const_val32(instr, idx);

                                // Pass target list pointer
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&current_target_list);

                                // Pass injection record pointer
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&injection_record);

                                // Pass corruption multipliers
                nvbit_add_call_arg_const_val32(instr, corruption_mult_low);
                nvbit_add_call_arg_const_val32(instr, corruption_mult_high);
                nvbit_add_call_arg_const_val32(instr, corruption_mult_f32);

                                // Pass injection mode and bit position
                nvbit_add_call_arg_const_val32(instr, injection_mode);
                nvbit_add_call_arg_const_val32(instr, bit_position);
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

            instrument_function_if_needed(ctx, func, gridDimX, gridDimY, gridDimZ,
                                        blockDimX, blockDimY, blockDimZ);
            nvbit_enable_instrumented(ctx, func, true);

                        // Initialize injection record and target list
            injection_record.num_checks = 0;
            injection_record.num_actual_injections = 0;
            injection_record.num_injected_locations = 0;

                        // Copy targets to managed memory for device access
            if (injection_targets.find(current_kernel) != injection_targets.end()) {
                auto& targets = injection_targets[current_kernel];
                current_target_list.num_targets = targets.size();
                for (size_t i = 0; i < targets.size() && i < MAX_TARGETS; i++) {
                    current_target_list.targets[i] = targets[i];
                }
            } else {
                current_target_list.num_targets = 0;
            }

            kernel_count++;
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());

            if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel) {
            } else if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
            }

                        // Only print debug info for targeted kernels
            if (targeted_kernels.find(kernel_count) != targeted_kernels.end()) {
                printf("\n========== INJECTION REPORT for Kernel %d - %s ==========\n",
                       kernel_count, nvbit_get_func_name(ctx, func, mangled));
                printf("  Total injection checks: %d\n", injection_record.num_checks);
                printf("  Number of targets configured: %d\n", (int)injection_targets[kernel_count].size());
                printf("  ACTUAL INJECTIONS PERFORMED: %d\n", injection_record.num_actual_injections);

                if (injection_record.num_actual_injections > 0) {
                    const char* mode_names[] = {"Single Bit-Flip", "FP16 Error Mult", "F32 Error Mult"};
                    printf("\n  ALL INJECTIONS (Mode: %s):\n",
                           injection_mode >= 0 && injection_mode <= 2 ? mode_names[injection_mode] : "Unknown");

                    int num_to_print = injection_record.num_actual_injections < MAX_INJECTION_RECORDS ?
                                       injection_record.num_actual_injections : MAX_INJECTION_RECORDS;
                    for (int i = 0; i < num_to_print; i++) {
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
                            printf("        Low:  %.6f -> %.6f (x%.3f)\n",
                                   detail.original_value_low, detail.corrupted_value_low,
                                   detail.original_value_low != 0.0f ? detail.corrupted_value_low / detail.original_value_low : 0.0f);
                            printf("        High: %.6f -> %.6f (x%.3f)\n",
                                   detail.original_value_high, detail.corrupted_value_high,
                                   detail.original_value_high != 0.0f ? detail.corrupted_value_high / detail.original_value_high : 0.0f);

                        } else if (injection_mode == 2) {
                                                        // Mode 2: F32 error multiplication - show single F32 value
                            printf("        F32:  %.6f -> %.6f (x%.3f)\n",
                                   detail.original_value_low, detail.corrupted_value_low,
                                   detail.original_value_low != 0.0f ? detail.corrupted_value_low / detail.original_value_low : 0.0f);
                        }
                    }
                    if (injection_record.num_actual_injections > MAX_INJECTION_RECORDS) {
                        printf("    ... (%d more injections not shown)\n",
                               injection_record.num_actual_injections - MAX_INJECTION_RECORDS);
                    }
                } else {
                    printf("  ✗ NO FAULT INJECTED\n");
                    printf("    Target not matched (no thread executed any target instruction)\n");
                    printf("    %d targets were configured for this kernel\n",
                           (int)injection_targets[kernel_count].size());
                }
                printf("========================================================\n\n");
            }
        }
    }
}

void nvbit_at_term() {
    printf("\n========================================\n");
    printf("INJECTION COMPLETE\n");
    printf("========================================\n");
}
