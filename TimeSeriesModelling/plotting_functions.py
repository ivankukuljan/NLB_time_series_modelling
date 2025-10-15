#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from typing import Dict, Tuple, Iterable, Optional

from utils import get_basedir, load_nlb_npl_data, get_eurostat_time_unit, extract_dataset_name_delay_and_unit, year_locator_for_span


BASE_DIR = get_basedir()


def plot_nlb_and_eurostat_data(
    eurostat_datasets,
    output_path=os.path.join(BASE_DIR, 'data', 'data_plots.png')
):
    """
    Create a grid of subplots with the NLB NPL time series and each Eurostat dataset.
    Each subplot displays a macroeconomic time series from 2008–2022, if available.
    The resulting grid is saved as a PNG to output_path.
    """
    # Total number of plots: 1 for NLB NPL plus one per Eurostat dataset
    num_plots = 1 + len(eurostat_datasets)
    cols = 3  # Number of columns for subplot grid
    # Compute rows needed to fit all subplots
    rows = (num_plots + cols - 1) // cols
    # Create the figure and axes for subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.6, rows * 3.8), squeeze=False)
    axes_flat = axes.ravel()  # Flatten for sequential access

    # ---------- Plot NLB NPL ----------
    # Load NLB dates and NPL values
    nlb_dates, nlb_npl_values = load_nlb_npl_data()
    ax = axes_flat[0]  # First subplot for NLB NPL
    try:
        # Try to convert dates to pandas datetime for better handling
        idx = pd.to_datetime(nlb_dates)
    except Exception:
        # Fallback: keep as list if conversion fails
        idx = nlb_dates.to_list()
    # Plot NLB NPL series
    ax.plot(idx, nlb_npl_values, linewidth=3.5, color='#230078')
    ax.set_title('NLB NPL', color='#230078', pad=8, fontweight='bold')
    # Optionally: set x/y labels (commented)
    # ax.set_xlabel('Date'); ax.set_ylabel('NPL')
    # If the index has more than one date, adjust x-axis ticks for years
    if hasattr(idx, '__len__') and len(idx) > 1:
        loc, fmt = year_locator_for_span(idx)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)

    # ---------- Plot each Eurostat dataset ----------
    plot_i = 1  # Start with subplot 1 (subplot 0 is NLB NPL)
    for key, dataset in eurostat_datasets.items():
        ax = axes_flat[plot_i]

        # Convert 'times' and 'values' to pandas Series
        times = pd.Series(dataset['times'])
        values = pd.Series(dataset['values'])
        ylab = dataset.get('dataset_code', 'Value')  # Y-axis label (dataset code or 'Value')

        # Filter data: only include entries from 2008 to 2022 (by year)
        years = times.astype(str).str[:4].astype(int)
        mask = (years >= 2008) & (years <= 2022)
        times_f = times[mask]
        values_f = values[mask]

        # If no filtered data, write notice and skip plot
        if times_f.empty:
            ax.set_title(f"{key} (no data 2008–2022)", pad=8)
            ax.set_xlabel('Time')
            ax.set_ylabel(ylab)
            plot_i += 1
            continue

        # Choose the frequency: annual ('a') or quarterly ('q')
        freq = get_eurostat_time_unit(str(times_f.iloc[0]))  # Determine time unit

        # Build x-axis values as datetime objects for plotting
        if freq == 'a':
            x_vals = pd.to_datetime(times_f.astype(str), format='%Y')
        else:  # for quarter
            x_vals = pd.PeriodIndex(times_f.astype(str), freq='Q').to_timestamp('Q')

        # Plot the Eurostat time series
        ax.plot(x_vals, values_f.values, linewidth=2.2, color='black')
        ax.set_title(key, pad=8)
        # Optionally: set x/y labels (commented)
        # ax.set_xlabel('Time'); ax.set_ylabel(ylab)

        # Set appropriately spaced year ticks
        loc, fmt = year_locator_for_span(x_vals)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)

        plot_i += 1
        # Exit if we've filled all available axes
        if plot_i >= len(axes_flat):
            break

    # Hide any unused axes in the subplot grid
    for n in range(num_plots, rows * cols):
        fig.delaxes(axes_flat[n])

    # Adjust layout and save the figure
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_df_head_table(
    df: pd.DataFrame,
    n: int = 5,
    caption: str = "Data preview",
    precision: int = 4,
    show_index: bool = True,   # show row names by default
    heatmap: bool = False,     # no background colors
):
    """
    Render a nicely styled HTML table (in Jupyter/Quarto) showing df.head(n),
    with index (row names) visible and no background coloring.
    Returns (head_df, styler).
    """
    head_df = df.head(n).copy()

    # Numeric formatting only for numeric columns
    num_cols = head_df.select_dtypes(include="number").columns.tolist()
    fmt = {col: f"{{:.{precision}f}}" for col in num_cols}

    styler = (
        head_df.style
        .set_caption(caption)
        .format(fmt, na_rep="—")
        .set_table_styles(
            [
                {"selector": "caption",
                 "props": [("caption-side", "top"),
                           ("font-size", "1.05rem"),
                           ("font-weight", "600"),
                           ("padding", "0 0 8px 0")]},
                {"selector": "table",
                 "props": [("border-collapse", "separate"),
                           ("border-spacing", "0"),
                           ("width", "100%"),
                           ("border", "1px solid #e6e6e6"),
                           ("border-radius", "12px"),
                           ("overflow", "hidden"),
                           ("font-family", "system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial")]},
                # Header — no background color
                {"selector": "thead th",
                 "props": [("border-bottom", "1px solid #e6e6e6"),
                           ("padding", "10px 12px"),
                           ("text-align", "center"),
                           ("font-weight", "600")]},
                # Index (row labels) — no background color
                {"selector": "th.row_heading",
                 "props": [("border-right", "1px solid #e6e6e6"),
                           ("font-weight", "600"),
                           ("padding", "10px 12px"),
                           ("text-align", "left")]},
                # Body cells — no background color
                {"selector": "tbody td",
                 "props": [("padding", "10px 12px"),
                           ("text-align", "left"),
                           ("border-bottom", "1px solid #f0f0f0")]},
                {"selector": "tbody tr:hover",
                 "props": [("background", "#fcfcfc")]}
            ]
        )
        .set_properties(**{"min-width": "80px"})
    )

    # Index visibility (keep shown unless user overrides)
    try:
        if not show_index:
            styler = styler.hide(axis="index")
    except Exception:
        if not show_index and hasattr(styler, "hide_index"):
            styler = styler.hide_index()

    # No heatmap by default; only apply if explicitly requested
    if heatmap and num_cols:
        styler = styler.background_gradient(
            cmap="RdYlGn_r",
            subset=num_cols,
            gmap=head_df[num_cols].abs(),
            axis=None
        )

    return head_df, styler





def plot_model_coefficients_table(
    model,
    data: pd.DataFrame,
    target: str = "npl",
    caption: str = "Model Coefficients",
    precision: int = 4,
    heatmap: bool = True,
):
    """
    Create a nicely-styled HTML table of model coefficients that renders well in Jupyter/Quarto.

    Parameters
    ----------
    model : fitted sklearn-like model (must have .coef_)
    data : pd.DataFrame
        Training dataframe; features are all columns except `target`.
    target : str, default "npl"
        Name of the target column to drop from `data`.
    caption : str
        Caption shown above the table.
    precision : int
        Decimal places for coefficients.
    heatmap : bool
        If True, apply a red-blue background gradient by value.

    Returns
    -------
    (df, styler) : (pd.DataFrame, pd.io.formats.style.Styler)
        The raw dataframe and its styled representation (display `styler` in notebooks).
    """
    # Features and coefficients
    X = data.drop(columns=[target])
    coef = np.asarray(model.coef_).ravel()
    if coef.size != X.shape[1]:
        raise ValueError(
            f"Coefficient length ({coef.size}) does not match number of features ({X.shape[1]})."
        )

    df = pd.DataFrame([coef], columns=X.columns, index=["Coefficient"])

    # Formatting
    fmt = {col: f"{{:.{precision}f}}" for col in df.columns}

    styler = (
        df.style
        .set_caption(caption)
        .format(fmt, na_rep="—")
        .set_table_styles(
            [
                {"selector": "caption",
                 "props": [("caption-side", "top"),
                           ("font-size", "1.05rem"),
                           ("font-weight", "600"),
                           ("padding", "0 0 8px 0")]},
                {"selector": "table",
                 "props": [("border-collapse", "separate"),
                           ("border-spacing", "0"),
                           ("width", "100%"),
                           ("border", "1px solid #e6e6e6"),
                           ("border-radius", "12px"),
                           ("overflow", "hidden"),
                           ("font-family", "system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial") ]},
                {"selector": "thead th",
                 "props": [("background", "#fafafa"),
                           ("border-bottom", "1px solid #e6e6e6"),
                           ("padding", "10px 12px"),
                           ("text-align", "center"),
                           ("font-weight", "600")]},
                {"selector": "th.row_heading",
                 "props": [("background", "#fafafa"),
                           ("border-right", "1px solid #e6e6e6"),
                           ("font-weight", "600"),
                           ("padding", "10px 12px"),
                           ("text-align", "left")]},
                {"selector": "tbody td",
                 "props": [("padding", "10px 12px"),
                           ("text-align", "right"),
                           ("border-bottom", "1px solid #f0f0f0")]},
                {"selector": "tbody tr:hover",
                 "props": [("background", "#fcfcfc")]}
            ]
        )
        .set_properties(**{"min-width": "80px"})
    )

    if heatmap:
        # Create a diverging color map: red for negative, white for zero, blue for positive
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "red_white_blue", ["#d73027", "#ffffff", "#4575b4"]
        )
        vmin, vmax = df.values.min(), df.values.max()
        max_abs = max(abs(vmin), abs(vmax))
        styler = styler.background_gradient(cmap=cmap, axis=None, vmin=-max_abs, vmax=max_abs)

    return df, styler


# Plots the NPL values (actual and predicted) for training and two test splits
def plot_npl_predictions(data_train, data_test_1, data_test_2, y_pred_test_1, mse_test_1, r2_test_1, y_pred_test_2, mse_test_2, r2_test_2):
    """
    Plot NPL (Non-Performing Loan) values for training and test sets, together with model predictions
    and their associated MSE and R² metrics for two test splits.

    Parameters
    ----------
    data_train : pd.DataFrame
        Training data. Must contain a 'npl' column and use a datetime index.
    data_test_1 : pd.DataFrame
        First test split. Must contain a 'npl' column and use a datetime index.
    data_test_2 : pd.DataFrame
        Second test split. Must contain a 'npl' column and use a datetime index.
    y_pred_test_1 : np.ndarray or pd.Series
        Predicted NPL values for data_test_1, same order as data_test_1.
    mse_test_1 : float
        Mean squared error for predictions on data_test_1.
    r2_test_1 : float
        R² score for predictions on data_test_1.
    y_pred_test_2 : np.ndarray or pd.Series
        Predicted NPL values for data_test_2, same order as data_test_2.
    mse_test_2 : float
        Mean squared error for predictions on data_test_2.
    r2_test_2 : float
        R² score for predictions on data_test_2.

    Returns
    -------
    None
        Shows and saves plot with training and test sets and predictions.
    """

    # Create figure with two stacked subplots (taller for training/test_1, shorter for test_2)
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'height_ratios': [2, 1]}
    )

    # --- Top subplot: Training set and first test split ---
    ax = axes[0]
    # Plot observed training NPL values
    ax.plot(
        data_train.index,
        data_train['npl'],
        color='black',
        label='train data',
        linewidth=2
    )
    # Plot observed first test split (ground truth)
    ax.scatter(
        data_test_1.index,
        data_test_1['npl'],
        color='green',
        label='test_1 ground truth',
        zorder=3
    )
    # Plot predicted values for first test split
    ax.scatter(
        data_test_1.index,
        y_pred_test_1,
        color='violet',
        label='test_1 predicted',
        marker='x',
        s=70,
        zorder=4
    )
    ax.set_title(
        f"Training and Test 1 data with predictions (MSE: {mse_test_1:.4f}, R²: {r2_test_1:.4f})"
    )
    ax.set_ylabel("NPL")
    ax.legend()
    # ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # --- Bottom subplot: Second test split only ---
    ax2 = axes[1]
    # Plot observed second test split (ground truth)
    ax2.scatter(
        data_test_2.index,
        data_test_2['npl'],
        color='green',
        label='test_2 ground truth',
        zorder=3
    )
    ax2.plot(
    data_test_2.index,
    data_test_2['npl'],
    color='green',
    linewidth=2,
    alpha=0.7,
    label='_nolegend_',  # so it doesn’t add a duplicate legend entry
    zorder=2
    )
    # Plot predicted values for second test split
    ax2.scatter(
        data_test_2.index,
        y_pred_test_2,
        color='violet',
        label='test_2 predicted',
        marker='x',
        s=70,
        zorder=4
    )
    ax2.set_title(
        f"Test 2 data with predictions (MSE: {mse_test_2:.4f}, R²: {r2_test_2:.4f})"
    )
    ax2.set_ylabel("NPL")
    ax2.legend()
    # ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.set_xlabel("date")

    # Finalize layout and save/show the plot
    plt.tight_layout()
    # Save to file, ensure tight bounding box for legend etc.
    plt.savefig(
        os.path.join(BASE_DIR, 'plots', 'npl_predictions.png'),
        dpi=300,
        bbox_inches="tight",   # includes outside artists
        pad_inches=0.1
    )
    plt.show()


def plot_abs_coeff_series_by_dataset(
    model: LinearRegression,
    feature_names: Iterable[str],
    title: Optional[str] = "Absolute model coefficients by dataset (log scale)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Group linear model coefficients into time-delay series per (dataset name, unit),
    then plot the absolute values on a single log-y plot.
    Lines show |coef|; marker direction shows sign at each delay.
    """
    if not hasattr(model, "coef_"):
        raise ValueError("Model must be fit and have a 'coef_' attribute.")

    coefs = np.asarray(model.coef_, dtype=float)
    names = list(feature_names)
    if coefs.shape[0] != len(names):
        raise ValueError(
            f"Length mismatch: {coefs.shape[0]} coefficients vs {len(names)} feature names."
        )

    # (dataset, unit) -> delay -> (abs_val, sign)
    grouped: Dict[Tuple[str, str], Dict[int, Tuple[float, float]]] = defaultdict(dict)
    for name, coef in zip(names, coefs):
        try:
            dataset, delay, unit = extract_dataset_name_delay_and_unit(name)
        except ValueError:
            continue
        grouped[(dataset, unit)][delay] = (float(abs(coef)), float(np.sign(coef)))

    if not grouped:
        raise ValueError("No parsable feature names found to plot.")

    series = []
    for key, mapping in grouped.items():
        delays = sorted(mapping.keys())
        ys = [mapping[d][0] for d in delays]
        signs = [mapping[d][1] for d in delays]
        max_abs = max(ys) if ys else 0.0
        series.append({"key": key, "x": delays, "y": ys, "signs": signs, "max_abs": max_abs})
    series.sort(key=lambda s: s["max_abs"], reverse=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
           

    linestyles = ["-", "--", ":"]
    colors = [d.get("color") for d in plt.rcParams["axes.prop_cycle"]]
    styles = [(c, linestyles[i]) for c in colors for i in range(len(linestyles))]

    # Plot: line for |coef| with a single legend label; scatters for sign markers (no legend)
    for i, s in enumerate(series):
        color, ls = styles[i % len(styles)]
        dataset, unit = s["key"]
        label = f"{dataset} ({unit})"

        # 1) line for abs values (legend label here, once)
        ax.plot(
            s["x"], s["y"],
            linestyle=ls, color=color, linewidth=2, alpha=0.7, label=label,
        )

        # 2) sign markers (no legend entries)
        pos_x = [x for x, sign in zip(s["x"], s["signs"]) if sign > 0]
        pos_y = [y for y, sign in zip(s["y"], s["signs"]) if sign > 0]
        neg_x = [x for x, sign in zip(s["x"], s["signs"]) if sign < 0]
        neg_y = [y for y, sign in zip(s["y"], s["signs"]) if sign < 0]

        if pos_x:
            ax.scatter(
                pos_x, pos_y,
                marker="^", color=color, linewidth=0.5,
                s=55, alpha=0.8, label="_nolegend_",
            )
        if neg_x:
            ax.scatter(
                neg_x, neg_y,
                marker="v", color=color, linewidth=0.5,
                s=55, alpha=0.8, label="_nolegend_",
            )

    # Axes styling
    ax.set_yscale("log")
    ax.set_xlabel("delay in time units")
    ax.set_ylabel("|coefficient| (log scale)")
    if title:
        ax.set_title(title)

    # ax.grid(axis="y", which="both", color="gray", linewidth=0.6, alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Build legend: series lines first, then black triangle sign key at the bottom
    handles, labels = ax.get_legend_handles_labels()
    sign_handles = [
        Line2D([], [], marker="^", linestyle="None", color="black", markersize=7, label="positive"),
        Line2D([], [], marker="v", linestyle="None", color="black", markersize=7, label="negative"),
    ]
    ax.legend(
        handles + sign_handles, 
        labels + ["positive coefficient (▲)", "negative coefficient (▼)"],
        loc="center left", 
        bbox_to_anchor=(1.02, 0.5), 
        fontsize=8, 
        ncol=1, 
        frameon=False,
        title="Variable name (time unit)"
    )

    plt.savefig(os.path.join(BASE_DIR, 'plots', 'model_coefficients.png'),
        dpi=300, bbox_inches="tight", pad_inches=0.1
    )
    plt.show()
    

                