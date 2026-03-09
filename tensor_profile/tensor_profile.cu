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
#include <vector>
#include <random>
#include <algorithm>
#include <string>

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

// Global control
int kernel_count = 0;
bool mangled = false;
int verbose = 1;
int num_injections_per_kernel = 50;  // Configurable number of injection targets per kernel
std::string boundary_path = "boundary.txt";
std::string target_opcode = "HMMA";  // Opcode prefix to target (configurable via TARGET_OPCODE)

// Frequency-based targeting
struct FrequencyRange {
    int start;
    int end;
};
std::vector<FrequencyRange> frequency_ranges;
int num_frequency_ranges = 0;
std::unordered_set<int> targeted_kernels;  // Kernels selected for injection

// Injection targets generated during profiling
std::unordered_map<int, std::vector<InjectionTarget>> injection_targets;

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());

void nvbit_at_init() {
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1, "Print kernel names mangled or not");
    GET_VAR_INT(num_injections_per_kernel, "NUM_INJECTIONS", 50, "Number of injection targets per kernel");
    GET_VAR_STR(boundary_path, "BOUNDARY_PATH", "Path to boundary file (start,end per line)");
    GET_VAR_STR(target_opcode, "TARGET_OPCODE", "Opcode prefix to target (default: HMMA)");

    printf("========================================\n");
    printf("HMMA PROFILER MODE\n");
    printf("========================================\n");

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

std::unordered_set<CUfunction> already_instrumented;

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

        for (size_t idx = 0; idx < instrs.size(); idx++) {
            auto instr = instrs[idx];
            std::string opcode = instr->getOpcode();

            if (opcode.find(target_opcode) != std::string::npos) {
                profile.has_target_instr = true;
                profile.target_instr_indices.push_back(idx);

                if (verbose) {
                    printf("  Found HMMA at idx %zu\n", idx);
                }

                                // Insert profiling instrumentation
                nvbit_insert_call(instr, "profile_hmma", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr, current_kernel);

                                // Pass managed profile structure pointer
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&current_profile);
            }
        }

        if (profile.has_target_instr) {
            profile.kernel_name = nvbit_get_func_name(ctx, f, mangled);
            kernel_profiles[current_kernel] = profile;
            kernel_info_map[current_kernel] = {ctx, f};
            printf("Kernel %d has %zu HMMA instructions\n",
                   current_kernel, profile.target_instr_indices.size());
        }
    }
}

void select_injection_targets() {
        // Select one kernel per frequency range that has HMMA
    std::vector<int> hmma_kernels;
    for (const auto& kv : kernel_profiles) {
        if (kv.second.has_target_instr) {
            hmma_kernels.push_back(kv.first);
        }
    }

    printf("\nFound %zu kernels with HMMA instructions\n", hmma_kernels.size());
    printf("Generating %d injection targets per selected kernel\n", num_injections_per_kernel);

    for (int freq_range = 0; freq_range < num_frequency_ranges; freq_range++) {
        int range_start = frequency_ranges[freq_range].start;
        int range_end = frequency_ranges[freq_range].end;

        std::vector<int> range_hmma_kernels;
        for (int k : hmma_kernels) {
            if (k >= range_start && k <= range_end) {
                range_hmma_kernels.push_back(k);
            }
        }

        if (!range_hmma_kernels.empty()) {
                        // Randomly select one kernel from this frequency range
            std::uniform_int_distribution<> kernel_dist(0, range_hmma_kernels.size() - 1);
            int selected_kernel = range_hmma_kernels[kernel_dist(gen)];
            targeted_kernels.insert(selected_kernel);

                        // Randomly select MULTIPLE targets from profiled ranges
            auto& profile = kernel_profiles[selected_kernel];

                        // Create distributions for random selection
            std::uniform_int_distribution<> block_x_dist(profile.block_x_min, profile.block_x_max);
            std::uniform_int_distribution<> block_y_dist(profile.block_y_min, profile.block_y_max);
            std::uniform_int_distribution<> block_z_dist(profile.block_z_min, profile.block_z_max);
            std::uniform_int_distribution<> thread_x_dist(profile.thread_x_min, profile.thread_x_max);
            std::uniform_int_distribution<> thread_y_dist(profile.thread_y_min, profile.thread_y_max);
            std::uniform_int_distribution<> thread_z_dist(profile.thread_z_min, profile.thread_z_max);
            std::uniform_int_distribution<> instr_dist(0, profile.target_instr_indices.size() - 1);

                        // Generate multiple random targets
            for (int i = 0; i < num_injections_per_kernel; i++) {
                InjectionTarget target;

                target.target_block_x = block_x_dist(gen);
                target.target_block_y = block_y_dist(gen);
                target.target_block_z = block_z_dist(gen);
                target.target_thread_x = thread_x_dist(gen);
                target.target_thread_y = thread_y_dist(gen);
                target.target_thread_z = thread_z_dist(gen);
                target.target_instr = profile.target_instr_indices[instr_dist(gen)];

                injection_targets[selected_kernel].push_back(target);
            }

            printf("Frequency range %d: Selected kernel %d with %d targets\n",
                   freq_range, selected_kernel, num_injections_per_kernel);
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
                        // Initialize managed profiling structure before kernel launch
            current_profile.block_x_min = current_profile.block_y_min = current_profile.block_z_min = 0xFFFFFFFF;
            current_profile.thread_x_min = current_profile.thread_y_min = current_profile.thread_z_min = 0xFFFFFFFF;
            current_profile.block_x_max = current_profile.block_y_max = current_profile.block_z_max = 0;
            current_profile.thread_x_max = current_profile.thread_y_max = current_profile.thread_z_max = 0;

            instrument_function_if_needed(ctx, func, gridDimX, gridDimY, gridDimZ,
                                        blockDimX, blockDimY, blockDimZ);
            nvbit_enable_instrumented(ctx, func, true);

            kernel_count++;
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());

                        // Calculate grid dimensions to ensure proper params access
            if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel) {
            } else if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
            }

                        // Copy managed profiling data to host kernel profile
            if (current_profile.block_x_min != 0xFFFFFFFF && kernel_profiles.find(kernel_count) != kernel_profiles.end()) {
                KernelProfile& host_prof = kernel_profiles[kernel_count];
                host_prof.block_x_min = current_profile.block_x_min;
                host_prof.block_x_max = current_profile.block_x_max;
                host_prof.block_y_min = current_profile.block_y_min;
                host_prof.block_y_max = current_profile.block_y_max;
                host_prof.block_z_min = current_profile.block_z_min;
                host_prof.block_z_max = current_profile.block_z_max;
                host_prof.thread_x_min = current_profile.thread_x_min;
                host_prof.thread_x_max = current_profile.thread_x_max;
                host_prof.thread_y_min = current_profile.thread_y_min;
                host_prof.thread_y_max = current_profile.thread_y_max;
                host_prof.thread_z_min = current_profile.thread_z_min;
                host_prof.thread_z_max = current_profile.thread_z_max;
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

        // Generate selection for injection mode
    printf("\n=== GENERATING INJECTION TARGETS ===\n");
    select_injection_targets();

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
            fprintf(fp_info, "BLOCKS %u %u %u %u %u %u\n",
                   profile.block_x_min, profile.block_x_max,
                   profile.block_y_min, profile.block_y_max,
                   profile.block_z_min, profile.block_z_max);
            fprintf(fp_info, "THREADS %u %u %u %u %u %u\n",
                   profile.thread_x_min, profile.thread_x_max,
                   profile.thread_y_min, profile.thread_y_max,
                   profile.thread_z_min, profile.thread_z_max);
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

        // Save injection targets to separate file with kernel names
    FILE* fp_targets = fopen("fault_injection_target.txt", "w");
    if (fp_targets) {
        size_t total_targets = 0;
        for (const auto& kv : injection_targets) {
            total_targets += kv.second.size();
        }
        fprintf(fp_targets, "TARGETS %zu\n", total_targets);

        // Write targets grouped by kernel, with kernel name for each group
        for (const auto& kv : injection_targets) {
            int kernel_id = kv.first;
            const auto& targets = kv.second;

            // Write kernel name if available
            if (kernel_profiles.find(kernel_id) != kernel_profiles.end()) {
                fprintf(fp_targets, "KERNEL_NAME %s\n", kernel_profiles[kernel_id].kernel_name.c_str());
            }

            // Write all targets for this kernel
            for (const auto& target : targets) {
                fprintf(fp_targets, "TARGET %d %d %d %d %d %d %d %u\n",
                       kernel_id,
                       target.target_block_x, target.target_block_y, target.target_block_z,
                       target.target_thread_x, target.target_thread_y, target.target_thread_z,
                       target.target_instr);
            }
        }
        fclose(fp_targets);
        printf("Fault injection targets saved to fault_injection_target.txt\n");
    }
}
