#!/usr/bin/env python3
"""
Plot MRtrix connectome matrices.

Usage:
    python3 plot_connectome.py input_matrix.csv output_figure.png "Figure title"
    python3 plot_connectome.py --batch /path/to/subject/output

Examples:
    python3 plot_connectome.py \\
        connectome_matrix_aparc+aseg.csv \\
        connectome_matrix_10M_aparc+aseg.png \\
        "Connectome matrix subject 100206 (aparc+aseg)"

    python3 plot_connectome.py --batch /path/to/100206/output
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_connectome_matrix(csv_path: str) -> np.ndarray:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input matrix does not exist: {csv_path}")
    try:
        mat = np.loadtxt(csv_path, delimiter=",")
    except Exception:
        mat = np.loadtxt(csv_path)  # fallback for whitespace-delimited
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape={mat.shape}: {csv_path}")
    return mat


def infer_metric_from_name(path: str) -> str:
    name = os.path.basename(path)
    if "FA_mean" in name:
        return "FA mean"
    if "MD_mean" in name:
        return "MD mean"
    if "AD_mean" in name:
        return "AD mean"
    if "RD_mean" in name:
        return "RD mean"
    if "SIFT_sum" in name:
        return "SIFT2 sum"
    return "Streamline count"


def infer_parc_from_name(path: str) -> str:
    # Check aparc.a2009s+aseg before aparc+aseg — the latter is a substring of the former.
    name = os.path.basename(path).replace(".csv", "")
    for parc in ("aparc.a2009s+aseg", "aparc+aseg"):
        if parc in name:
            return parc
    m = re.search(r"connectome_matrix(?:_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)?)?_(.+)$", name)
    return m.group(1) if m else "unknown parcellation"


def get_display_limits(mat: np.ndarray, percentile: float = 99.0, symmetric: bool = False):
    """Determine robust display range, ignoring zeros and outliers."""
    arr = np.array(mat, dtype=np.float64, copy=True)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    values = finite[finite != 0]
    if values.size == 0:
        return 0.0, 1.0
    if symmetric:
        vmax = np.nanpercentile(np.abs(values), percentile)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(values))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        return -vmax, vmax
    vmin = 0.0 if np.nanmin(values) >= 0 else np.nanmin(values)
    vmax = np.nanpercentile(values, percentile)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = np.nanmax(values)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def maybe_transform_matrix(mat: np.ndarray, transform: str) -> np.ndarray:
    arr = np.array(mat, dtype=np.float64, copy=True)
    if transform == "none":
        return arr
    if transform == "log1p":
        arr[arr < 0] = 0
        return np.log1p(arr)
    if transform == "sqrt":
        arr[arr < 0] = 0
        return np.sqrt(arr)
    raise ValueError(f"Unknown transform: {transform}")


def choose_default_transform(input_csv: str, mat: np.ndarray) -> str:
    """Use log1p for count/SIFT matrices with large dynamic range; leave FA/MD/AD/RD linear."""
    if infer_metric_from_name(input_csv) in {"FA mean", "MD mean", "AD mean", "RD mean"}:
        return "none"
    finite = mat[np.isfinite(mat)]
    return "log1p" if finite.size > 0 and np.nanmax(finite) > 100 else "none"


def plot_connectome(
    input_csv: str,
    output_png: str,
    title: str = None,
    cmap: str = "viridis",
    dpi: int = 300,
    percentile: float = 99.0,
    transform: str = "auto",
    ignore_diagonal: bool = False,
    show_values: bool = False,
):
    mat = read_connectome_matrix(input_csv)

    if not title:
        title = f"{infer_metric_from_name(input_csv)} connectome ({infer_parc_from_name(input_csv)})"

    if transform == "auto":
        transform = choose_default_transform(input_csv, mat)

    plot_mat = maybe_transform_matrix(mat, transform)

    if ignore_diagonal and plot_mat.shape[0] == plot_mat.shape[1]:
        plot_mat = plot_mat.copy()
        np.fill_diagonal(plot_mat, np.nan)

    symmetric_color = bool(np.nanmin(plot_mat) < 0 and np.nanmax(plot_mat) > 0)
    vmin, vmax = get_display_limits(plot_mat, percentile=percentile, symmetric=symmetric_color)

    n_rows, n_cols = plot_mat.shape
    base_size = max(6.0, min(14.0, max(n_rows, n_cols) / 12.0))
    fig, ax = plt.subplots(figsize=(base_size, base_size * 0.95))

    im = ax.imshow(plot_mat, interpolation="nearest", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Node index")
    ax.set_ylabel("Node index")

    max_dim = max(n_rows, n_cols)
    tick_step = 1 if max_dim <= 20 else 10 if max_dim <= 100 else 25 if max_dim <= 250 else 50
    ax.set_xticks(np.arange(0, n_cols, tick_step))
    ax.set_yticks(np.arange(0, n_rows, tick_step))

    if show_values and n_rows <= 20 and n_cols <= 20:
        for i in range(n_rows):
            for j in range(n_cols):
                val = plot_mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    metric = infer_metric_from_name(input_csv)
    cbar.set_label(f"log1p({metric})" if transform == "log1p" else f"sqrt({metric})" if transform == "sqrt" else metric)

    nonzero = int(np.count_nonzero(np.nan_to_num(mat, nan=0.0)))
    total = int(mat.size)
    fig.text(
        0.5, 0.01,
        f"shape={mat.shape[0]}×{mat.shape[1]}, nonzero={nonzero}/{total} ({nonzero/total:.2%}), "
        f"display={transform}, p{percentile:g} clipping",
        ha="center", va="bottom", fontsize=8,
    )

    os.makedirs(os.path.dirname(os.path.abspath(str(output_png))), exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(str(output_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved plot: {output_png}")


def batch_plot(output_dir: str, dpi: int = 300, percentile: float = 99.0, transform: str = "auto"):
    """Plot all connectome_matrix*.csv files in a subject output directory."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {output_dir}")
    csv_files = sorted(output_dir.glob("connectome_matrix*.csv"))
    if not csv_files:
        print(f"[WARNING] No connectome_matrix*.csv found in: {output_dir}")
        return
    for csv_path in csv_files:
        png_path = csv_path.with_suffix(".png")
        print(f"[INFO] Plotting: {csv_path}  ->  {png_path}")
        plot_connectome(str(csv_path), str(png_path), dpi=dpi, percentile=percentile, transform=transform)


def build_argparser():
    parser = argparse.ArgumentParser(description="Plot MRtrix connectome matrices from CSV files.")
    parser.add_argument("input_csv", nargs="?", help="Input connectome matrix CSV file.")
    parser.add_argument("output_png", nargs="?", help="Output PNG path.")
    parser.add_argument("title", nargs="?", default=None, help="Figure title.")
    parser.add_argument("--batch", metavar="OUTPUT_DIR", help="Plot all connectome_matrix*.csv files in this directory.")
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI (default: 300).")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap (default: viridis).")
    parser.add_argument("--percentile", type=float, default=99.0, help="Upper percentile for display clipping (default: 99).")
    parser.add_argument("--transform", choices=["auto", "none", "log1p", "sqrt"], default="auto", help="Display transform (default: auto).")
    parser.add_argument("--ignore-diagonal", action="store_true", help="Set diagonal to NaN in display.")
    parser.add_argument("--show-values", action="store_true", help="Print values inside cells for small matrices (<=20x20).")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.batch is not None:
        batch_plot(output_dir=args.batch, dpi=args.dpi, percentile=args.percentile, transform=args.transform)
        return

    if args.input_csv is None or args.output_png is None:
        parser.print_help()
        sys.exit(1)

    plot_connectome(
        input_csv=args.input_csv,
        output_png=args.output_png,
        title=args.title,
        cmap=args.cmap,
        dpi=args.dpi,
        percentile=args.percentile,
        transform=args.transform,
        ignore_diagonal=args.ignore_diagonal,
        show_values=args.show_values,
    )


if __name__ == "__main__":
    main()
