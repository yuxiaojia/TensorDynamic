# TensorDynamic — Artifact Evaluation

TensorDynamic is an NVBit-based fault injection tool targeting Tensor Core (HMMA) instructions.
This artifact evaluates fault injection on four DNN models (ResNet20, MobileNetV2, ShuffleNetV2,
YOLOv9) using three backends: PyTorchFI, MRFI, and TensorDynamic.

---

## Requirements

- NVIDIA A100 GPU (or any Ampere/Hopper GPU with Tensor Core support)
- CUDA 12.6.1
- GCC 12.3.0
- Conda

> **Georgia Tech ICE cluster:** load the required versions with:
> ```bash
> module load gcc/12.3.0
> module load cuda/12.6.1
> ```

---

## Quickstart

```bash
# 1. Create and activate conda environment
conda create -n myenv python=3.10 pip -y
conda activate myenv

# 2. Download and extract NVBit 1.7.6
wget https://github.com/NVlabs/NVBit/releases/download/v1.7.6/nvbit-Linux-x86_64-1.7.6.tar.bz2
tar xvfj nvbit-Linux-x86_64-1.7.6.tar.bz2
cd nvbit_release_x86_64

# 3. Clone this repository
git clone <repo_url> TensorDynamic
cd TensorDynamic

# All subsequent commands should be run from the TensorDynamic/ directory.

# 4. Install dependencies (~5-10 min)
pip install -r requirements.txt

# 5. Set up YOLOv9 repo, dataset, and weights (~5 min)
bash setup_yolo.sh
```

---

## Table 3 — Fault Injection Observability (Convolution vs. GEMM)

```bash
nohup bash table3/table3.sh > table3_run.log 2>&1
```

Output: `table3/results/table3_summary.csv`

**Estimated time: ~2–3 h**

---

## Figure 6 — HMMA Register Sparsity

> **Recommended first step** — this is the fastest way to verify your environment is set up correctly before running the longer fault injection sweeps.

```bash
bash figure6/figure6.sh
```

Builds the NVBit profiling tool, runs all four models, parses results, and generates:
`figure6/reg_dist_plots/combined_sparsity_dest_zero_4panel.png`

**Estimated time: ~1 h**

---

## Figure 4 — Fault Injection Accuracy Sweep

Run the full pipeline (sweeps → baselines → parse → plot):

```bash
nohup bash figure4/figure4.sh > figure4_run.log 2>&1
```

Or run each step individually:

```bash
# Step 1: PyTorchFI & MRFI sweep (~2-3 h)
nohup bash figure4/sweep_pytorchfi_mrfi.sh > pytorchfi_run.log 2>&1

# Step 2: TensorDynamic sweep (~13-14 h)
nohup bash figure4/sweep_tensordynamic.sh > tensordynamic_run.log 2>&1

# Step 3: Collect baselines, parse results, generate plot
bash figure4/plot_results.sh
```

Output: `figure4/plots/eight_plots.png` and `figure4/plots/eight_plots.pdf`

---

## Figure 5 — Baseline vs. Parameter Sensitivity

Run the full pipeline (sweep → parse → plot):

```bash
nohup bash figure5/figure5.sh > figure5_run.log 2>&1
```

Or run each step individually:

```bash
# Step 1: Run TensorDynamic sensitivity sweep + PyTorchFI baseline (~8 h)
nohup bash figure5/sweep_baseline.sh > figure5_sweep.log 2>&1

# Step 2: After sweep completes — parse results and generate plot
bash figure5/plot_figure5.sh
```

Output: `figure5/plots/nine_point_plot.png` and `figure5/plots/nine_point_plot.pdf`

---

## Directory Structure

```
TensorDynamic/
  setup_yolo.sh               -- YOLOv9 repo, dataset, and weights setup
  requirements.txt            -- pip dependencies (use: pip install -r requirements.txt)
  examples/
    boundary/                 -- pre-generated boundary files for all four models
    eval_resnet20.py, eval_mobilenet.py, eval_shufflenet.py
    mult_*/bit_*              -- PyTorchFI/MRFI fault injection scripts
    yolo/yolov9/              -- YOLOv9 model and eval scripts
  figure4/
    figure4.sh                -- full pipeline: sweeps + baselines + parse + plot
    sweep_pytorchfi_mrfi.sh   -- PyTorchFI & MRFI sweep
    sweep_tensordynamic.sh    -- TensorDynamic sweep
    plot_results.sh           -- collect baselines, parse results, generate plot
    parse_results.py          -- aggregate CSVs from raw results
    plot_8_panels.py          -- generate 8-panel accuracy figure
    csv/                      -- parsed summary CSVs (generated)
    plots/                    -- output figures (generated)
  figure5/
    figure5.sh                -- full pipeline: sweep + parse + plot
    sweep_baseline.sh         -- PyTorchFI baseline + TensorDynamic sensitivity sweep
    parse_results.py          -- parse raw CSVs into pivot tables (averaged across kernels)
    plot.py                   -- generate 4-panel sensitivity figure
    plot_figure5.sh           -- parse + plot pipeline (run after sweep)
    results_baseline/         -- raw sweep CSVs (generated)
    csv/                      -- parsed pivot CSVs (generated)
    plots/                    -- output figures (generated)
  table3/
    table3.sh                 -- full pipeline: 50 trials + aggregate + CSV
    conv_matrix.py            -- convolution workload
    matrix_mult.py            -- GEMM workload
    compare_cnn.py            -- compute metrics for convolution
    compare_gemm.py           -- compute metrics for GEMM
    results/                  -- per-trial CSVs + table3_summary.csv (generated)
  figure6/
    figure6.sh                -- full pipeline: profile -> parse -> plot
    parse_reg_dist.py         -- parse NVBit output txt -> CSV
    plot_reg_dist.py          -- generate 4-panel sparsity figure
  tensor_dynamic/
    tensor_dynamic.so         -- NVBit fault injection tool
  hmma_reg_distribution/
    hmma_reg_distribution.so  -- NVBit register profiling tool
```
