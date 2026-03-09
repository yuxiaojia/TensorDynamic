import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("plots", exist_ok=True)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 23
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.titlesize'] = 26
plt.rcParams['legend.fontsize'] = 23
plt.rcParams['xtick.labelsize'] = 23
plt.rcParams['ytick.labelsize'] = 23

# Load baselines computed by parse_results.py
_bl = pd.read_csv("csv/baselines.csv").set_index("model")["baseline"]

# Model configurations: (csv filename in csv/, display_name, model key for baseline)
model_configs = [
    ("csv/accuracy_pivot_resnet.csv", "ResNet20",    _bl["resnet20"]),
    ("csv/accuracy_pivot_mobile.csv", "MobileNetv2", _bl["mobilenet"]),
    ("csv/accuracy_pivot_shuff.csv",  "ShuffleNetv2",_bl["shufflenet"]),
    ("csv/mAP50_pivot_yolo.csv",      "YOLOv9",      _bl["yolo"]),
]

# Define colors for lines
line_color = "#2E86AB"

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

panel_labels = ["(a)", "(b)", "(c)", "(d)"]

# Process each model
for idx, (filename, display_name, baseline) in enumerate(model_configs):
    ax = axes[idx]
    df = pd.read_csv(filename)

    # Get the data - each row is a count, each column (except first) is a mult value
    counts = df['count'].values
    mult_columns = [col for col in df.columns if col != 'count']

    # Extract mult values from column names (e.g., "mult=1000" -> 1000)
    mult_values = [int(col.split('=')[1]) for col in mult_columns]

    # Create 9 data points: 3 counts × 3 mult values
    x_points = []
    y_points = []

    for i, count in enumerate(counts):
        for j, mult in enumerate(mult_values):
            # x-axis: combination index (0-8)
            x_points.append(i * 3 + j)
            # y-axis: accuracy/mAP value
            y_points.append(df.iloc[i][mult_columns[j]])

    # Plot with markers
    ax.plot(x_points, y_points,
           marker='o',
           color=line_color,
           linewidth=2.5,
           markersize=8,
           alpha=0.9)

    # Add baseline horizontal line
    # Baseline = PyTorchFI with 500 errors, ×1000 magnitude (reference point)
    is_yolo = (display_name == "YOLOv9")
    label_str = f'PyTorchFI (500, ×1000): {baseline:.2f}' if is_yolo else f'PyTorchFI (500, ×1000): {baseline:.2f}%'
    baseline_line = ax.axhline(y=baseline, color='red', linestyle=':', linewidth=2,
                               label=label_str,
                               alpha=0.7)

    # Annotate baseline value at the right end of the line (non-YOLO only)
    if not is_yolo:
        ax.annotate(f'{baseline:.2f}%',
                    xy=(8.5, baseline),
                    xytext=(0, 5),
                    textcoords='offset points',
                    color='red', fontsize=13, fontweight='bold',
                    ha='right', va='bottom')

    # Set up x-axis labels
    # count represents how many times more errors nvbit needs compared to pytorch (baseline 500)
    # mult represents the error factor (how much bigger the error is, baseline 1000)
    x_labels = []
    count_multipliers = [20, 100, 200]  # 10000/500=20x, 50000/500=100x, 100000/500=200x
    mult_multipliers = [1, 10, 100]      # 1000/1000=1x, 10000/1000=10x, 100000/1000=100x

    for count_mult in count_multipliers:
        for mult_mult in mult_multipliers:
            x_labels.append(f"({count_mult}×, {mult_mult}×)")

    ax.set_xticks(range(9))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontweight='bold', fontsize=15)

    # Title with panel label
    ax.set_title(f"{panel_labels[idx]} {display_name}", fontweight='bold', fontsize=26, pad=15)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    legend = ax.legend(loc='best', framealpha=0.9, edgecolor='gray', fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # Make tick labels bold
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

# Add common X and Y labels
fig.supxlabel('(Error Count Multiplier, Error Magnitude)', fontsize=29, fontweight='bold', y=0.11,  x=0.55)
fig.supylabel('Accuracy (%) / mAP@50', fontsize=29, fontweight='bold', x=0.11)

# Tight layout
plt.tight_layout(rect=[0.08, 0.08, 1, 1])

# Save figure
plt.savefig("plots/nine_point_plot.png", dpi=300, bbox_inches='tight')
plt.savefig("plots/nine_point_plot.pdf", bbox_inches='tight')
print("Saved plots/nine_point_plot.png and plots/nine_point_plot.pdf")
