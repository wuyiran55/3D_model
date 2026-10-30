"""
===============================================================================
Script Name : plot_percentage_layers_stackbar.py
Project     : Spatial Genome Analysis
Description : Calculate and visualize the distribution of genomic features
              across spatial layers of a reconstructed 3D genome model.

Author      : Yiran Wu
Email       : wuyiran55@outlook.com

Created     : 2026-10-30
Last Updated: 2026-10-30
Version     : 1.0.0
===============================================================================
"""



import argparse
import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
import math
from collections import OrderedDict
from scipy.stats import linregress
import seaborn as sns
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.ticker import StrMethodFormatter, MaxNLocator, FuncFormatter
from typing import Tuple
from typing import Optional, Sequence
# ======= Import config utilities ======= #
sys.path.append(os.path.expanduser("~/05.scripts/plot_figures/"))
from config_loader import load_configs
from plot_style import apply_mpl_style

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "Arial"
cfg = load_configs(
    os.path.expanduser("~/05.scripts/plot_figures/configs/global.yaml"),
    os.path.expanduser("~/05.scripts/plot_figures/configs/plot_defaults.yaml"),

)
apply_mpl_style(cfg)

def read_info(info_file):
    print(f"reading {info_file}:")
    info_df = pd.read_csv(info_file, header=0, sep="\t")
    colnames=info_df.columns
    newcol=[k.replace("##","") for k in colnames ]
    info_df.columns=newcol
    print(info_df.describe())
    return info_df


def search_prefix(input: str | Path, target_dir: str = "00.data") -> Tuple[Path, Path, Path, str]:
    """
    Search upward from input (a file or directory) for the directory named target_dir.
    Return (prefix_root, dir_00data, sample_dir, sample_name).
    - prefix_root: Parent directory of 00.data (usually .../02.results/07.ON_modeling).
    - dir_00data : The 00.data directory.
    - sample_dir : Directory containing the specific sample (00.data/<sample_name>).
    - sample_name: Name of the sample directory.
    """
    p = Path(input).expanduser().resolve()
    if p.is_file():
        p_dir = p.parent
    else:
        p_dir = p

    # Traverse upward to find the directory named target_dir.
    cur = p_dir
    while True:
        if cur.name == target_dir:
            dir_00data = cur
            prefix_root = cur.parent
            # Using b_factors.txt as an example, the sample directory is the
            # parent directory of the file, located under 00.data.
            # (.../00.data/<sample>/b_factors.txt)
            # If input is a file, p.parent is the <sample> directory.
            # If input is the <sample> directory, p_dir is the <sample> directory.
            sample_dir = p.parent if p.is_file() else p_dir
            sample_name = sample_dir.name
            return prefix_root, dir_00data, sample_dir, sample_name
        if cur.parent == cur:
            # Reached the root directory without finding the target directory.
            raise FileNotFoundError(f"{target_dir} not found in path: {p}")
        cur = cur.parent


def create_custom_legend_small(colors, cutoffs, tag="RPKM", output=None):
    print(cutoffs)
    labels = []
    cutoffs = cutoffs[::-1]
    font_size = 5  # Adjust font size according to the figure dimensions.
    if cutoffs[-1] < 0:
        cutoffs[-1] = 0
    for k in range(len(cutoffs) - 1):
        myrange = f"{cutoffs[k+1]:.2f}-{cutoffs[k]:.2f}"
        if k == 0:
            myrange = f"> {cutoffs[k+1]:.2f}"
        labels.append(myrange)
    print(labels)

    # Create legend patches.
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]


    fig = plt.figure(figsize=(0.45, 0.6), dpi=600)

    ax = fig.add_axes([0, 0, 1, 1])  # left, bottom, width, height (0~1)

    legend = ax.legend(
        handles=patches,
        title=tag,
        loc='center',
        fontsize=font_size,
        title_fontsize=font_size,
        handleheight=0.8,
        handlelength=1,
        labelspacing=0.2,
        borderaxespad=0.1,
        handletextpad=0.4,
        frameon=False
    )


    ax.set_axis_off()


    if output is not None:
        print(f"output: {output}")
        plt.savefig(output, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_layers_rp_percentage_clean(df, output, legend=True, legend_output=None, colors=None, ylabel='Percentage of bins'):
    import matplotlib.pyplot as plt
    import numpy as np

    # Required physical dimensions.
    width_in_inches = 0.45
    height_in_inches = 0.6
    dpi = 600  # High resolution.

    # Set all line widths consistently to 0.3 pt.
    spine_width_pt = 0.3
    pt_to_inch = 1 / 72.27
    spine_width_inch = spine_width_pt * pt_to_inch

    fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)

    ax.set_box_aspect(1.33333)

    # Draw a stacked bar chart without labels.
    df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.9)

    # Remove label text.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set tick lengths and line widths.
    ax.set_yticks(np.arange(0, 101, 25))
    ax.tick_params(axis='both', which='both', length=2, width=spine_width_pt)

    # Set border line widths.
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width_pt)

    # Remove the legend.
    ax.get_legend().remove()

    # Disable automatic layout adjustments.
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the figure without bbox_inches='tight' to preserve its original dimensions.
    if output is not None:
        print(f"")
        fig.savefig(output, dpi=dpi, bbox_inches=None, pad_inches=0.0)
    plt.show()
    plt.close(fig)



def plot_layers_rp_percentage_bysample(
    sample: str,
    tag: str = "RPKM",
    key: Optional[str] = None,
    oridf: Optional[pd.DataFrame] = None,
    cutoffs: Optional[Sequence[float]] = None,
    legend: bool = False,
    colors=None,
    wkdir: str | Path = "./",
    legend_tag: str = "RPKM",
    model: Optional[str] = None,
    groups: int = 5
):
    """
    Calculate the percentage of bins within each layer, grouped by quantiles
    of the values in key (five quantile groups by default), and output the
    figure and cutoff values.

    Returns:
        rp_pd   : Percentage table (%) with layers as rows and quantile groups
                  as columns (ordered from high to low, i.e., groups..1).
        oridf   : Original DataFrame with an additional 'rp_quartile' column.
        cutoffs : Binning cutoff values (length groups+1; the last value is
                  the maximum boundary).

    Depends on external functions:
        plot_layers_rp_percentage_clean(df, output, legend, colors)
        create_custom_legend_small(colors, cutoffs, output, tag)

    Required columns:
        oridf['layer'], oridf[key]
    """
    # Directory.
    indir = Path(wkdir)
    indir.mkdir(parents=True, exist_ok=True)

    if key is None:
        key = sample
    if oridf is None or key not in oridf.columns:
        raise ValueError(f"`oridf` is empty or missing column: {key}")
    if "layer" not in oridf.columns:
        raise ValueError("`oridf` is missing column: 'layer'")

    df = oridf.copy()

    # Binning.
    if cutoffs is not None:
        # User-defined cutoffs: ensure they are increasing and have length >= 2.
        newcut = list(map(float, cutoffs))
        if len(newcut) < 2 or any(newcut[i] >= newcut[i+1] for i in range(len(newcut)-1)):
            raise ValueError(f"Invalid cutoffs: {cutoffs}")
        # Extend the final upper boundary to a large value to avoid
        # excluding values due to the right-open interval.
        newcut = copy.deepcopy(newcut)
        newcut[-1] = float("inf")
        df["rp_quartile"] = pd.cut(
            df[key],
            bins=newcut,
            labels=range(1, len(newcut)),  # len(newcut)-1 bins.
            right=False,
            include_lowest=True
        )
        # Restore cutoffs for output (remove infinity and retain
        # the original upper boundary).
        out_cutoffs = list(cutoffs)
    else:
        # Automatic qcut: handle NaN and constant values.
        series = df[key]
        # Temporarily exclude NaN values when calculating bins.
        non_na = series.dropna()
        if non_na.nunique() < groups:
            # If there are too few unique values, fall back to equal-width
            # cuts or reduce the number of groups.
            groups_eff = min(groups, max(1, int(non_na.nunique())))
            if groups_eff == 1:
                # All values are identical: assign them to the same bin.
                df["rp_quartile"] = 1
                out_cutoffs = [float(non_na.min()), float(non_na.max())]
            else:
                intervals = pd.qcut(non_na, groups_eff, duplicates="drop").cat.categories
                # Extract the left and right endpoints.
                out_cutoffs = sorted(set([float(iv.left) for iv in intervals] +
                                         [float(iv.right) for iv in intervals]))
                df.loc[non_na.index, "rp_quartile"] = pd.qcut(
                    non_na, groups_eff, labels=range(1, groups_eff+1), duplicates="drop"
                )
        else:
            intervals = pd.qcut(non_na, groups, duplicates="drop").cat.categories
            out_cutoffs = sorted(set([float(iv.left) for iv in intervals] +
                                     [float(iv.right) for iv in intervals]))
            df.loc[non_na.index, "rp_quartile"] = pd.qcut(
                non_na, groups, labels=range(1, groups+1), duplicates="drop"
            )

    # Percentage table.
    # First calculate the total number of bins in each layer.
    layer_sizes = df.groupby("layer", dropna=False).size().to_dict()

    # Initialize all columns (1..groups) to 0.
    # Note: If automatic binning produces fewer bins than groups,
    # generate columns based on the groups actually present.
    present_quartiles = sorted(df["rp_quartile"].dropna().unique().tolist())
    if not present_quartiles:
        present_quartiles = [1]
    # Ensure the values are integers.
    present_quartiles = [int(x) for x in present_quartiles]

    # Count.
    rp_counts = (
        df.groupby(["layer", "rp_quartile"], dropna=False)
          .size()
          .rename("cnt")
          .reset_index()
    )

    # Generate the percentage table.
    rp_pd = (
        rp_counts
        .pivot(index="layer", columns="rp_quartile", values="cnt")
        .fillna(0.0)
    )
    for q in present_quartiles:
        rp_pd[q] = rp_pd.get(q, 0.0)

    # Convert to percentages.
    rp_pd = rp_pd.div(pd.Series(layer_sizes), axis=0) * 100.0

    # Set column order from high to low (groups .. 1),
    # retaining only columns that exist.
    ordered_cols = [q for q in range(groups, 0, -1) if q in rp_pd.columns]
    rp_pd = rp_pd[ordered_cols]

    # Output filenames.
    model_str = "" if model is None else str(model)
    outfig_clean = indir / f"{sample}_{tag}_percentage_layers_{model_str}_clean.svg"
    legend_output2 = indir / f"{sample}_{tag}_percentage_layers_{model_str}_smalllegend.svg"
    cutoffs_output = indir / f"{sample}_{tag}_percentage_layers_{model_str}.cutoffs"

    # Plot (depends on your external function).
    plot_layers_rp_percentage_clean(
        df=rp_pd, output=str(outfig_clean), legend=legend, colors=colors
    )

    # Legend and cutoffs.
    if colors is not None and out_cutoffs is not None:
        # Keep this line that removes "cn" from the tag
        # used by your legend function.
        create_custom_legend_small(
            colors=colors,
            cutoffs=out_cutoffs,
            output=str(legend_output2),
            tag=tag.replace("cn", "")
        )

    with open(cutoffs_output, "w") as f:
        f.write("\t".join(map(str, out_cutoffs)) + "\n")

    return rp_pd, df, out_cutoffs
