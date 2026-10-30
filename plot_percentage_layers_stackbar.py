"""
===============================================================================
Script Name : plot_spatial_layer_distribution.py
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
    Calculate the percentage distribution of value-based quantile
    groups within each spatial layer.

    By default, values are divided into five quantile groups.

    Returns:
        rp_pd:
            Percentage table (%).
            Rows represent spatial layers.
            Columns represent quantile groups, ordered from
            high to low (groups ... 1).

        oridf:
            Original DataFrame with an additional column:
            'rp_quartile'.

        cutoffs:
            Bin boundaries used for value classification.
            Normally contains groups + 1 boundaries.

    Dependencies:
        plot_layers_rp_percentage_clean(
            df, output, legend, colors
        )

        create_custom_legend_small(
            colors, cutoffs, output, tag
        )

    Required columns:
        oridf['layer']
        oridf[key]
    """

    # Create the output directory.
    indir = Path(wkdir)
    indir.mkdir(
        parents=True,
        exist_ok=True
    )

    if key is None:
        key = sample

    if oridf is None or key not in oridf.columns:
        raise ValueError(
            f"`oridf` is empty or missing the required column: {key}"
        )

    if "layer" not in oridf.columns:
        raise ValueError(
            "`oridf` is missing the required column: 'layer'"
        )

    df = oridf.copy()

    # ==========================================================
    # VALUE BINNING
    # ==========================================================

    if cutoffs is not None:

        # Use user-defined cutoff values.
        # Ensure that cutoffs are strictly increasing and
        # contain at least two boundaries.
        newcut = list(map(float, cutoffs))

        if (
            len(newcut) < 2
            or any(
                newcut[i] >= newcut[i + 1]
                for i in range(len(newcut) - 1)
            )
        ):
            raise ValueError(
                f"Invalid cutoff values: {cutoffs}"
            )

        # Extend the last upper boundary to infinity
        # to avoid excluding values above the specified maximum.
        newcut = copy.deepcopy(newcut)
        newcut[-1] = float("inf")

        df["rp_quartile"] = pd.cut(
            df[key],
            bins=newcut,
            labels=range(1, len(newcut)),
            # Number of bins = len(newcut) - 1.
            right=False,
            include_lowest=True
        )

        # Preserve the original cutoff values without infinity.
        out_cutoffs = list(cutoffs)

    else:

        # Automatically determine quantile bins using qcut.
        # Handle missing values and constant-value distributions.
        series = df[key]

        # Temporarily remove NaN values before calculating bins.
        non_na = series.dropna()

        if non_na.nunique() < groups:

            # If the number of unique values is smaller than
            # the requested number of groups, reduce the
            # effective number of groups.
            groups_eff = min(
                groups,
                max(1, int(non_na.nunique()))
            )

            if groups_eff == 1:

                # All values are identical.
                # Assign all observations to a single bin.
                df["rp_quartile"] = 1

                out_cutoffs = [
                    float(non_na.min()),
                    float(non_na.max())
                ]

            else:

                intervals = pd.qcut(
                    non_na,
                    groups_eff,
                    duplicates="drop"
                ).cat.categories

                # Extract the left and right boundaries
                # of the quantile intervals.
                out_cutoffs = sorted(
                    set(
                        [float(iv.left) for iv in intervals]
                        +
                        [float(iv.right) for iv in intervals]
                    )
                )

                df.loc[
                    non_na.index,
                    "rp_quartile"
                ] = pd.qcut(
                    non_na,
                    groups_eff,
                    labels=range(1, groups_eff + 1),
                    duplicates="drop"
                )

        else:

            intervals = pd.qcut(
                non_na,
                groups,
                duplicates="drop"
            ).cat.categories

            out_cutoffs = sorted(
                set(
                    [float(iv.left) for iv in intervals]
                    +
                    [float(iv.right) for iv in intervals]
                )
            )

            df.loc[
                non_na.index,
                "rp_quartile"
            ] = pd.qcut(
                non_na,
                groups,
                labels=range(1, groups + 1),
                duplicates="drop"
            )

    # ==========================================================
    # CALCULATE PERCENTAGES WITHIN EACH LAYER
    # ==========================================================

    # Calculate the total number of bins in each spatial layer.
    layer_sizes = (
        df.groupby(
            "layer",
            dropna=False
        )
        .size()
        .to_dict()
    )

    # Initialize all quantile groups.
    # If the effective number of bins is smaller than the
    # requested number of groups, use the groups that are
    # actually present in the data.
    present_quartiles = sorted(
        df["rp_quartile"]
        .dropna()
        .unique()
        .tolist()
    )

    if not present_quartiles:
        present_quartiles = [1]

    # Convert quantile identifiers to integers.
    present_quartiles = [
        int(x)
        for x in present_quartiles
    ]

    # Count the number of bins in each
    # layer-by-quantile combination.
    rp_counts = (
        df.groupby(
            ["layer", "rp_quartile"],
            dropna=False
        )
        .size()
        .rename("cnt")
        .reset_index()
    )

    # Create a count table with:
    # rows = spatial layers
    # columns = quantile groups
    rp_pd = (
        rp_counts
        .pivot(
            index="layer",
            columns="rp_quartile",
            values="cnt"
        )
        .fillna(0.0)
    )

    for q in present_quartiles:
        rp_pd[q] = rp_pd.get(q, 0.0)

    # Convert counts to percentages within each layer.
    rp_pd = (
        rp_pd.div(
            pd.Series(layer_sizes),
            axis=0
        )
        * 100.0
    )

    # Order columns from highest to lowest quantile group.
    # Retain only quantile groups present in the table.
    ordered_cols = [
        q
        for q in range(groups, 0, -1)
        if q in rp_pd.columns
    ]

    rp_pd = rp_pd[ordered_cols]

    # ==========================================================
    # OUTPUT FILE PATHS
    # ==========================================================

    model_str = "" if model is None else str(model)

    outfig_clean = (
        indir
        / f"{sample}_{tag}_percentage_layers_{model_str}_clean.svg"
    )

    legend_output2 = (
        indir
        / f"{sample}_{tag}_percentage_layers_{model_str}_smalllegend.svg"
    )

    cutoffs_output = (
        indir
        / f"{sample}_{tag}_percentage_layers_{model_str}.cutoffs"
    )

    # ==========================================================
    # PLOT STACKED PERCENTAGE DISTRIBUTION
    # ==========================================================

    # Generate the stacked percentage plot using
    # the external plotting function.
    plot_layers_rp_percentage_clean(
        df=rp_pd,
        output=str(outfig_clean),
        legend=legend,
        colors=colors
    )

    # ==========================================================
    # GENERATE LEGEND AND SAVE CUTOFF VALUES
    # ==========================================================

    if colors is not None and out_cutoffs is not None:

        # The legend function uses the tag as its title.
        # Remove "cn" from the tag while preserving
        # the original naming convention.
        create_custom_legend_small(
            colors=colors,
            cutoffs=out_cutoffs,
            output=str(legend_output2),
            tag=tag.replace("cn", "")
        )

    # Save cutoff boundaries to a tab-separated text file.
    with open(cutoffs_output, "w") as f:
        f.write(
            "\t".join(map(str, out_cutoffs)) + "\n"
        )

    return rp_pd, df, out_cutoffs
