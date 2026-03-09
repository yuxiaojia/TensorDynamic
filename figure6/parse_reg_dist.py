#!/usr/bin/env python3
"""
Parse hmma_reg_distribution output and write a binned CSV.

Usage:
    python3 parse_reg_dist.py <input.txt> <output.csv>

Input lines of interest (emitted by hmma_reg_distribution):
    kernel N - <name>
      HMMA regs: X zeros (Z%), Y non-zeros (NZ%)

Output CSV columns:
    Bin_Range    -- "0-10%", "10-20%", ..., "90-100%", then "Average"
    Count        -- number of kernel invocations whose zero% falls in that bin;
                    last row holds the actual mean zero% across all invocations
"""

import re
import sys
import csv
from pathlib import Path
from collections import defaultdict

HMMA_PAT = re.compile(
    r"HMMA regs:\s+\d+\s+zeros\s+\(([0-9.]+)%\)"
)

BINS = [f"{i*10}-{(i+1)*10}%" for i in range(10)]


def parse(txt_path: Path) -> list[float]:
    """Return list of zero-% values, one per HMMA kernel invocation."""
    values = []
    for line in txt_path.read_text().splitlines():
        m = HMMA_PAT.search(line)
        if m:
            values.append(float(m.group(1)))
    return values


def bin_values(values: list[float]) -> list[int]:
    """Bin zero-% values into 10 equal-width buckets, return raw invocation count per bin."""
    counts = defaultdict(int)
    for v in values:
        bucket = min(int(v // 10), 9)  # clamp 100% into last bin
        counts[bucket] += 1
    return [counts[i] for i in range(10)]


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.txt> <output.csv>")
        sys.exit(1)

    txt_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])

    values = parse(txt_path)
    if not values:
        print(f"WARNING: no HMMA kernel lines found in {txt_path}")

    bin_counts = bin_values(values)
    average = round(sum(values) / len(values), 2) if values else 0.0

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Bin_Range", "Count"])
        for bin_label, cnt in zip(BINS, bin_counts):
            writer.writerow([bin_label, cnt])
        writer.writerow(["Average", average])

    print(f"Parsed {len(values)} HMMA kernel invocations, avg zero%={average} -> {csv_path}")


if __name__ == "__main__":
    main()
