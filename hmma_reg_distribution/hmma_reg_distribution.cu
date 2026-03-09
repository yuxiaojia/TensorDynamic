// NVBit tool: hmma_reg_distribution
// Instruments HMMA (Tensor Core) instructions to count how many output
// register values are zero vs non-zero (FP16 low half), reported per kernel
// and as a global total across the application.

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>
#include <string>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

uint32_t kernel_id = 0;

// per-kernel counters in managed memory, written by GPU instrumentation
__managed__ uint64_t zero_regs = 0;
__managed__ uint64_t nonzero_regs = 0;

// accumulated totals across all kernels
uint64_t total_zero_regs = 0;
uint64_t total_nonzero_regs = 0;

uint32_t start_grid_num = 0;
uint32_t end_grid_num = UINT32_MAX;
int verbose = 0;
int active_from_start = 1;
bool mangled = false;
bool active_region = true;

pthread_mutex_t mutex;

bool is_hmma_instruction(Instr *instr) {
    std::string opcode = instr->getOpcode();
    return opcode.find("HMMA") != std::string::npos;
}

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(start_grid_num, "START_GRID_NUM", 0,
                "Beginning of the kernel launch interval to instrument");
    GET_VAR_INT(end_grid_num, "END_GRID_NUM", UINT32_MAX,
                "End of the kernel launch interval to instrument");
    GET_VAR_INT(active_from_start, "ACTIVE_FROM_START", 1,
                "Start from launch 0 or wait for cuProfilerStart/Stop");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1,
                "Print kernel names mangled or not");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");

    if (active_from_start == 0) active_region = false;

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

// track which functions have been instrumented and which contain HMMA
std::unordered_set<CUfunction> already_instrumented;
std::unordered_set<CUfunction> hmma_functions;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);
    related_functions.push_back(func);

    for (auto f : related_functions) {
        if (!already_instrumented.insert(f).second) continue;

        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        if (verbose)
            printf("inspecting %s - num instrs %ld\n",
                   nvbit_get_func_name(ctx, f), instrs.size());

        for (auto i : instrs) {
            if (is_hmma_instruction(i)) {
                hmma_functions.insert(func);
                nvbit_insert_call(i, "count_reg_distribution", IPOINT_AFTER);
                nvbit_add_call_arg_const_val32(i, i->getOperand(0)->u.reg.num);
                nvbit_add_call_arg_const_val64(i, (uint64_t)&zero_regs);
                nvbit_add_call_arg_const_val64(i, (uint64_t)&nonzero_regs);
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
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
        }

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, func);

            if (active_from_start)
                active_region = (kernel_id >= start_grid_num &&
                                 kernel_id < end_grid_num);

            nvbit_enable_instrumented(ctx, func, active_region);

            // reset counters before each kernel launch
            if (hmma_functions.count(func) > 0) {
                zero_regs = 0;
                nonzero_regs = 0;
            }
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());

            // print per-kernel distribution after kernel completes
            if (hmma_functions.count(func) > 0) {
                uint64_t kernel_total = zero_regs + nonzero_regs;
                printf("\nkernel %d - %s\n",
                       kernel_id, nvbit_get_func_name(ctx, func, mangled));
                if (kernel_total > 0) {
                    printf("  HMMA regs: %ld zeros (%.1f%%), %ld non-zeros (%.1f%%)\n",
                           zero_regs, 100.0 * zero_regs / kernel_total,
                           nonzero_regs, 100.0 * nonzero_regs / kernel_total);
                    total_zero_regs += zero_regs;
                    total_nonzero_regs += nonzero_regs;
                }
            }

            kernel_id++;
            pthread_mutex_unlock(&mutex);
        }
    } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
        if (!active_from_start) active_region = true;
    } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
        if (!active_from_start) active_region = false;
    }
}

void nvbit_at_term() {
    uint64_t total = total_zero_regs + total_nonzero_regs;
    if (total > 0)
        printf("\nTotal HMMA registers: %ld zeros (%.1f%%), %ld non-zeros (%.1f%%)\n",
               total_zero_regs, 100.0 * total_zero_regs / total,
               total_nonzero_regs, 100.0 * total_nonzero_regs / total);
}
