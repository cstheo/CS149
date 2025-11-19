#!/usr/bin/env python3
"""
Plot Mandelbrot speedup measurements for the CS149 Assignment 1 write-up.

Usage:
    python3 plot_speedup.py --view 1
    python3 plot_speedup.py --view 2 --output view2_speedup.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

# Timing data collected via:
#   ./mandelbrot -v <view> -t <threads>
# which internally runs five iterations and reports the minimum runtimes.
VIEW_DATA = {
    1: [
        (1, 239.255, 238.211, 1.00),
        (2, 237.903, 121.269, 1.96),
        (3, 239.397, 150.145, 1.59),
        (4, 237.575, 99.919, 2.38),
        (5, 237.496, 102.314, 2.32),
        (6, 237.570, 78.154, 3.04),
        (7, 237.434, 76.798, 3.09),
        (8, 237.690, 66.583, 3.57),
        (9, 238.760, 65.750, 3.63),
        (10, 238.676, 59.265, 4.03),
        (11, 239.664, 57.208, 4.19),
        (12, 239.242, 51.313, 4.66),
        (13, 240.597, 48.541, 4.96),
        (14, 242.122, 46.371, 5.22),
        (15, 241.830, 46.121, 5.24),
        (16, 241.462, 44.807, 5.39),
        (17, 241.535, 44.781, 5.39),
        (18, 239.509, 43.486, 5.51),
        (19, 241.885, 41.021, 5.90),
        (20, 242.104, 39.580, 6.12),
        (21, 242.264, 39.977, 6.06),
        (22, 243.100, 41.573, 5.85),
        (23, 243.576, 39.700, 6.14),
        (24, 243.547, 40.023, 6.09),
        (25, 244.158, 39.256, 6.22),
        (26, 243.904, 38.710, 6.30),
        (27, 242.066, 39.512, 6.13),
        (28, 242.293, 40.363, 6.00),
        (29, 242.399, 41.473, 5.84),
        (30, 243.981, 39.360, 6.20),
        (31, 244.191, 38.907, 6.28),
        (32, 248.539, 40.478, 6.14),
    ],
    2: [
        (1, 131.709, 131.690, 1.00),
        (2, 131.332, 80.178, 1.64),
        (3, 132.026, 66.028, 2.00),
        (4, 131.960, 57.075, 2.31),
        (5, 132.799, 49.949, 2.66),
        (6, 132.883, 44.478, 2.99),
        (7, 134.095, 39.126, 3.43),
        (8, 132.869, 36.544, 3.64),
        (9, 132.916, 33.385, 3.98),
        (10, 134.237, 29.938, 4.48),
        (11, 134.745, 30.034, 4.49),
        (12, 134.761, 28.326, 4.76),
        (13, 136.841, 27.169, 5.04),
        (14, 134.340, 24.578, 5.47),
        (15, 135.707, 23.844, 5.69),
        (16, 136.079, 23.931, 5.69),
        (17, 135.539, 25.208, 5.38),
        (18, 135.797, 22.419, 6.06),
        (19, 137.162, 23.851, 5.75),
        (20, 137.123, 22.471, 6.10),
        (21, 135.370, 22.346, 6.06),
        (22, 135.061, 23.314, 5.79),
        (23, 135.810, 22.666, 5.99),
        (24, 136.205, 22.339, 6.10),
        (25, 136.275, 23.016, 5.92),
        (26, 135.989, 22.755, 5.98),
        (27, 136.333, 22.323, 6.11),
        (28, 137.082, 22.313, 6.14),
        (29, 137.249, 23.035, 5.96),
        (30, 136.473, 23.370, 5.84),
        (31, 137.065, 21.993, 6.23),
        (32, 137.096, 22.284, 6.15),
    ],
}


def extract_series(view: int) -> Tuple[List[int], List[float]]:
    data = VIEW_DATA[view]
    threads = [entry[0] for entry in data]
    speedups = [entry[3] for entry in data]
    return threads, speedups


def plot_speedup(view: int, output: Path) -> None:
    threads, speedups = extract_series(view)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(threads, speedups, marker="o", linewidth=2, label=f"View {view}")
    ax.set_xlabel("Thread count")
    ax.set_ylabel("Speedup vs. serial")
    ax.set_title(f"Mandelbrot Speedup (View {view})")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xticks(threads[::2])
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Mandelbrot speedup results.")
    parser.add_argument(
        "--view",
        type=int,
        choices=VIEW_DATA.keys(),
        default=1,
        help="Which view's data to plot (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image filename (defaults to view<view>_speedup.png)",
    )
    args = parser.parse_args()

    output = args.output or Path(f"view{args.view}_speedup.png")
    plot_speedup(args.view, output)


if __name__ == "__main__":
    main()
