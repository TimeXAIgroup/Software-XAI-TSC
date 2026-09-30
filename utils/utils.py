import json
import logging
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).parent.parent

logger = logging.getLogger()

mpl.colormaps.register(
    LinearSegmentedColormap.from_list(
        "BlueToRed", [(0, 0, 1), (1, 1, 1), (1, 0, 0)], N=256
    ),
    name="BlueToRed",
)
mpl.colormaps.register(
    LinearSegmentedColormap.from_list("WhiteToRed", [(1, 1, 1), (1, 0, 0)], N=256),
    name="WhiteToRed",
)
mpl.colormaps.register(
    LinearSegmentedColormap.from_list("GrayToOrange", ["#d3d3d3", "#ff8c00"], N=256),
    name="GrayToOrange",
)
mpl.colormaps.register(
    LinearSegmentedColormap.from_list("WhiteToOrange", ["#ffffff", "#ff8c00"], N=256),
    name="WhiteToOrange",
)


def make_reproducible(
    seed: int = 2025, use_legacy: bool = False
) -> np.random.RandomState | np.random.Generator:
    """
    Try to make the program as deterministic as possible.

    Uncomment code for each used library.

    Args:
    seed (int):       Integer to use the basic seed, big integers recommended.
    use_legacy (bool):Whether to return a legacy RandomState or a new Generator.

    Returns:
    np.random.RandomState:  A numpy random generator, if use_legacy is True
    np.random.Generator:  A numpy random generator, if use_legacy is False
    """
    # If numpy is used
    np.random.seed(seed)
    logger.info("Setting internal numpy seed to: %s", seed)
    if use_legacy:
        prng = np.random.RandomState(seed)
    else:
        prng = np.random.default_rng(seed)

    # If TensorFlow is used:
    # Source: https://github.com/NVIDIA/framework-determinism
    # logger.info("Setting internal Tensorflow seed to: %s",seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # tf.random.set_seed(seed)

    # If PyTorch is used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG "] = ":4096:8"
    # torch.use_deterministic_algorithms(True)
    logger.info("Setting internal torch seed to: %s", seed)
    torch.manual_seed(seed)

    # If random is used
    # logger.info("Setting internal random seed to: %s",seed)
    # random.seed(seed)

    return prng


def rescale_array(arr, new_min=-1.0, new_max=1.0):
    """
    Linearly rescale a NumPy array to new_min and new_max.

    Parameters
    ----------
    arr : np.ndarray
        Input array (any shape).
    new_min : float
        Desired minimum value of the scaled array.
    new_max : float
        Desired maximum value of the scaled array.

    Returns
    -------
    np.ndarray
        Scaled array with the same shape as input.
    """
    arr = np.asarray(arr, dtype=np.float32)
    old_min = np.min(arr)
    old_max = np.max(arr)

    # Avoid division by zero (constant array)
    if np.isclose(old_max, old_min):
        raise ValueError("Array is constant which would result in division by zero!")

    scaled = (arr - old_min) / (old_max - old_min)  # scale to [0, 1]
    scaled = scaled * (new_max - new_min) + new_min  # shift to [new_min, new_max]
    return scaled


def replace_positive(x, positive=True):
    mask = x > 0 if positive else x < 0
    x_mod = x.copy()
    x_mod[mask] = 0
    return x_mod


def plot_dft_attr_bubbles(
    signal_time,
    signal_freq,
    relevance_time,
    relevance_freq,
    linewidth=1.4,
    bubble_size=10,
    threshold=0.0,
    cmap_boost=0.0,
    title=None,
    save_to=None,
):
    # Blue to red: "BlueToRed", "colorwarm" or "RdBu_r"
    # Red: "WhiteToRed"
    cmap_time = "BlueToRed" if np.min(relevance_time) < 0 else "WhiteToRed"
    cmap_freq = "BlueToRed" if np.min(relevance_freq) < 0 else "WhiteToRed"

    nrows, ncols = 1, 2
    figsize = (10, 2.5)
    fig, ax = plt.subplots(
        nrows, ncols, gridspec_kw={"width_ratios": [3, 1]}, figsize=figsize, dpi=300
    )
    if title is not None:
        fig.suptitle(title)

    x = np.linspace(0, signal_time.shape[-1], signal_time.shape[-1])
    ax[0].plot(x, signal_time, linewidth=linewidth, color="black")
    xticks = np.arange(0, signal_time.shape[-1] + 1, 800)
    xlabels = [f"{x / 8:.0f}" for x in xticks]
    ax[0].set_xticks(xticks, labels=xlabels)
    x = np.where(np.abs(relevance_time) >= threshold)[0]
    y = signal_time[x]
    z = relevance_time[x] + np.sign(relevance_time[x]) * cmap_boost
    ax[0].scatter(
        x,
        y,
        marker="o",
        c=z,
        cmap=cmap_time,
        s=bubble_size,
        zorder=2,
        vmin=np.min(relevance_time),
        vmax=np.max(relevance_time),
    )
    ax[0].set_xlabel("Time [ms]")
    ax[0].set_ylabel("Relative Amplitude")
    # ax[0].set_title("signal in time domain")
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    # ax[0].spines["bottom"].set_visible(False)

    # Plot time domain
    t = np.arange(signal_freq.shape[-1])
    ax[1].plot(t, signal_freq, linewidth=linewidth, color="black")
    xticks = np.arange(0, signal_freq.shape[-1] + 1, 200)
    ax[1].set_xticks(xticks)
    x = np.where(np.abs(relevance_freq) >= threshold)[0]
    y = signal_freq[x]
    z = relevance_freq[x] + np.sign(relevance_freq[x]) * cmap_boost
    ax[1].scatter(
        x,
        y,
        marker="o",
        c=z,
        cmap=cmap_freq,
        s=bubble_size,
        zorder=2,
        vmin=np.min(relevance_freq),
        vmax=np.max(relevance_freq),
    )
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("Relative Amplitude")
    # ax[1].set_title("signal in freq. domain")
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    # ax[1].spines["bottom"].set_visible(False)
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()

    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

    return save_to


def plot_dft_attr_bkg(
    signal_time,
    signal_freq,
    relevance_time,
    relevance_freq,
    gt_freq=None,
    linewidth=1.4,
    threshold=0.0,
    cmap_boost=0.0,
    title=None,
    save_to=None,
):
    # Blue to red: "BlueToRed", "coolwarm" or "RdBu_r"
    # Red: "WhiteToRed"
    time_negative_exists = True if np.min(relevance_time) < 0 else False
    freq_negative_exists = True if np.min(relevance_freq) < 0 else False
    cmap_time = "BlueToRed" if time_negative_exists else "WhiteToRed"
    cmap_freq = "BlueToRed" if freq_negative_exists else "WhiteToRed"

    nrows, ncols = 1, 2
    figsize = (8, 2.5)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        gridspec_kw={"width_ratios": [3, 1]},
        figsize=figsize,
        dpi=300,
    )
    if title is not None:
        fig.suptitle(title)

    t = np.arange(signal_time.shape[-1])
    xticks = np.arange(0, signal_time.shape[-1] + 1, 800)
    xlabels = [f"{x / 8:.0f}" for x in xticks]
    ax[0].set_xticks(xticks, labels=xlabels)
    x = np.where(
        np.abs(relevance_time) >= threshold,
        relevance_time + np.sign(relevance_time) * cmap_boost,
        0,
    )
    im = ax[0].imshow(
        np.expand_dims(x, 0),
        aspect="auto",
        cmap=cmap_time,
        extent=[t[0], t[-1] + 1, np.min(signal_time), np.max(signal_time)],
        alpha=1,
        origin="lower",
    )
    ax[0].plot(t, signal_time, linewidth=linewidth, color="black")
    ax[0].set_xlabel("Time [ms]")
    ax[0].set_ylabel("Relative Signal Amplitude")
    # ax[0].set_title("signal in time domain")
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    # ax[0].spines["bottom"].set_visible(False)

    # Plot frequency domain
    t = np.arange(signal_freq.shape[-1])
    xticks = np.arange(0, signal_freq.shape[-1] + 1, 200)
    ax[1].set_xticks(xticks)
    x = np.where(
        np.abs(relevance_freq) >= threshold,
        relevance_freq + np.sign(relevance_freq) * cmap_boost,
        0,
    )

    im = ax[1].imshow(
        np.expand_dims(x, 0),
        aspect="auto",
        cmap=cmap_freq,
        extent=[t[0], t[-1] + 1, np.min(signal_freq), np.max(signal_freq)],
        alpha=1,
        origin="lower",
    )
    if gt_freq is not None:
        # im = ax[1].imshow(
        #     np.expand_dims(gt_freq, 0),
        #     aspect="auto",
        #     cmap="GrayToOrange",
        #     extent=[t[0], t[-1] + 1, 0, 1],
        #     alpha=1,
        #     origin="lower",
        # )

        # Geometry for the bar in *data y-coordinates*
        y0, y1 = ax[1].get_ylim()
        bar_h = 0.02 * (y1 - y0)
        gap = 0.01 * (y1 - y0)
        bar_y = y0 - gap - bar_h

        # Extend y-limits to make room for the bar
        ax[1].set_ylim(bar_y - 0.01 * (y1 - y0), y1)

        # Draw the confidence strip (1 x N "image") under the plot
        ax[1].imshow(
            np.expand_dims(gt_freq, 0),
            aspect="auto",
            interpolation="nearest",
            cmap="GrayToOrange",
            extent=[t[0], t[-1], bar_y, bar_y + bar_h],
            alpha=1,
        )
        # # Optional label
        # ax[1].text(x.min(), bar_y + bar_height / 2, "GT", va="center", ha="left")
    ax[1].plot(t, signal_freq, linewidth=linewidth, color="black")
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("Relative Frequency Amplitude")
    # ax[1].set_title("signal in freq. domain")
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    # ax[1].spines["bottom"].set_visible(False)
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()

    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

    return save_to


def plot_multi_dft_attr(
    signals_time,
    signals_freq,
    relevances_time,
    relevances_freq,
    gts_freq=None,
    gt_bar=True,
    linewidth=1.4,
    threshold=0.0,
    cmap_boost=0.0,
    attr_as_bkg=True,
    bubble_size=5,
    title=None,
    one_axis_label=True,
    height_per_row=2.5,
    width_per_column=4,
    font_size=20,
    save_to=None,
):
    plt.rcParams.update({"font.size": font_size})
    nrows, ncols = len(signals_time), 2
    figsize = (width_per_column * ncols, height_per_row * nrows)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        gridspec_kw={"width_ratios": [3, 1]},
        figsize=figsize,
        dpi=300,
    )
    if title is not None:
        fig.suptitle(title)

    if one_axis_label:
        # Left shared y-axis label
        fig.text(
            0.00, 0.5, "Relative Signal Amplitude", va="center", rotation="vertical"
        )
        # Right shared y-axis label
        fig.text(
            0.97, 0.5, "Relative Frequency Amplitude", va="center", rotation="vertical"
        )

    for index, (
        signal_time,
        relevance_time,
        signal_freq,
        relevance_freq,
        gt_freq,
    ) in enumerate(
        zip(signals_time, relevances_time, signals_freq, relevances_freq, gts_freq)
    ):
        # Blue to red: "BlueToRed", "coolwarm" or "RdBu_r"
        # Red: "WhiteToRed"
        time_negative_exists = True if np.min(relevance_time) < 0 else False
        freq_negative_exists = True if np.min(relevance_freq) < 0 else False
        cmap_time = "BlueToRed" if time_negative_exists else "WhiteToRed"
        cmap_freq = "BlueToRed" if freq_negative_exists else "WhiteToRed"

        # Only plot X-Axis for last figure
        show_x = False if (index + 1) < nrows else True

        # Plot Time Domain
        t = np.arange(signal_time.shape[-1])
        xticks = np.arange(0, signal_time.shape[-1] + 1, 800)
        xlabels = [f"{x / 8:.0f}" for x in xticks]
        ax[index % 2][0].set_xticks(xticks, labels=xlabels)
        if attr_as_bkg:
            # Attribution as background
            x = np.where(
                np.abs(relevance_time) >= threshold,
                relevance_time + np.sign(relevance_time) * cmap_boost,
                0,
            )
            im = ax[index % 2][0].imshow(
                np.expand_dims(x, 0),
                aspect="auto",
                cmap=cmap_time,
                extent=[t[0], t[-1] + 1, np.min(signal_time), np.max(signal_time)],
                alpha=1,
                origin="lower",
            )
        else:
            # Attribution as bubbles
            x = np.where(np.abs(relevance_time) >= threshold)[0]
            y = signal_time[x]
            z = relevance_time[x] + np.sign(relevance_time[x]) * cmap_boost
            ax[index % 2][0].scatter(
                x,
                y,
                marker="o",
                c=z,
                cmap=cmap_time,
                s=bubble_size,
                zorder=2,
                vmin=np.min(relevance_time),
                vmax=np.max(relevance_time),
            )
            # Correct limits
            # ax[index % 2][0].set_ylim(np.min(signal_time), np.max(signal_time))
        ax[index % 2][0].plot(t, signal_time, linewidth=linewidth, color="black")
        # ax[index%2][0].set_title("signal in time domain")
        ax[index % 2][0].spines["top"].set_visible(False)
        ax[index % 2][0].spines["right"].set_visible(False)
        ax[index % 2][0].set_ylabel(
            "Relative Signal Amplitude" if not one_axis_label else ""
        )
        ax[index % 2][0].set_xlabel("Time [ms]")
        ax[index % 2][0].spines["bottom"].set_visible(True)
        ax[index % 2][0].get_xaxis().set_visible(show_x)
        ax[index % 2][0].margins(x=0)

        # Plot frequency domain
        t = np.arange(signal_freq.shape[-1])
        xticks = np.arange(0, signal_freq.shape[-1] + 1, 200)
        ax[index % 2][1].set_xticks(xticks)
        ax[index % 2][1].plot(t, signal_freq, linewidth=linewidth, color="black")
        if attr_as_bkg:
            # Attribution as background
            x = np.where(
                np.abs(relevance_freq) >= threshold,
                relevance_freq + np.sign(relevance_freq) * cmap_boost,
                0,
            )
            im = ax[index % 2][1].imshow(
                np.expand_dims(x, 0),
                aspect="auto",
                cmap=cmap_freq,
                extent=[t[0], t[-1] + 1, np.min(signal_freq), np.max(signal_freq)],
                alpha=1,
                origin="lower",
            )
        else:
            # Attribution as bubbles
            x = np.where(np.abs(relevance_freq) >= threshold)[0]
            y = signal_freq[x]
            z = relevance_freq[x] + np.sign(relevance_freq[x]) * cmap_boost
            ax[index % 2][1].scatter(
                x,
                y,
                marker="o",
                c=z,
                cmap=cmap_freq,
                s=bubble_size,
                zorder=2,
                vmin=np.min(relevance_freq),
                vmax=np.max(relevance_freq),
            )
            # Correct limits
            # ax[index % 2][1].set_ylim(np.min(signal_freq), np.max(signal_freq))
        if gt_freq is not None:
            # Ground truth as bar or background
            if gt_bar:
                # Ground Truth bar
                # Geometry for the bar in *data y-coordinates*
                if attr_as_bkg:
                    y0, y1 = ax[index % 2][1].get_ylim()
                else:
                    y0, y1 = np.min(signal_freq), np.max(signal_freq)
                bar_h = 0.02 * (y1 - y0)
                gap = 0.01 * (y1 - y0)
                bar_y = y0 - gap - bar_h

                # Extend y-limits to make room for the bar
                ax[index % 2][1].set_ylim(bar_y - 0.01 * (y1 - y0), y1)

                # Draw the confidence strip (1 x N "image") under the plot
                ax[index % 2][1].imshow(
                    np.expand_dims(gt_freq, 0),
                    aspect="auto",
                    interpolation="nearest",
                    cmap="GrayToOrange",
                    extent=[t[0], t[-1], bar_y, bar_y + bar_h],
                    alpha=1,
                )
                # # Optional label
                # ax[index+1].text(x.min(), bar_y + bar_height / 2, "GT", va="center", ha="left")
            else:
                # GT as Background
                y0, y1 = ax[index % 2][1].get_ylim()
                x = np.where(
                    np.abs(relevance_freq) >= threshold,
                    relevance_freq + np.sign(relevance_freq) * cmap_boost,
                    0,
                )
                im = ax[index % 2][1].imshow(
                    np.expand_dims(gt_freq, 0),
                    aspect="auto",
                    cmap="WhiteToOrange",
                    extent=[t[0], t[-1] + 1, np.min(signal_freq), np.max(signal_freq)],
                    alpha=0.3,
                    origin="lower",
                )
                ax[index % 2][1].set_ylim(0, y1)
        ax[index % 2][1].set_ylabel(
            "Relative Frequency Amplitude" if not one_axis_label else ""
        )
        ax[index % 2][1].set_xlabel("Frequency [Hz]")
        # ax[index % 2][1].set_title("signal in freq. domain")
        ax[index % 2][1].spines["top"].set_visible(False)
        ax[index % 2][1].spines["left"].set_visible(False)
        ax[index % 2][1].yaxis.set_label_position("right")
        ax[index % 2][1].yaxis.tick_right()
        ax[index % 2][1].spines["bottom"].set_visible(True)
        ax[index % 2][1].get_xaxis().set_visible(show_x)
        ax[index % 2][1].margins(x=0)

    if one_axis_label:
        plt.tight_layout(rect=[0.01, 0, 0.97, 1])
    else:
        plt.tight_layout()

    if save_to is not None:
        # plt.savefig(save_to, bbox_inches="tight", dpi=300)
        plt.savefig(save_to, dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

    return save_to


def plot_time_attr_bkg(
    signal_time,
    relevance_time,
    linewidth=1.4,
    threshold=0.0,
    cmap_boost=0.0,
    save_to=None,
):
    # Blue to red: "BlueToRed", "coolwarm" or "RdBu_r"
    # Red: "WhiteToRed"
    time_negative_exists = True if np.min(relevance_time) < 0 else False
    cmap_time = "BlueToRed" if time_negative_exists else "WhiteToRed"

    figsize = (6, 2.5)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    t = np.arange(signal_time.shape[-1])
    xticks = np.arange(0, signal_time.shape[-1] + 1, 800)
    xlabels = [f"{x / 8:.0f}" for x in xticks]
    ax.set_xticks(xticks, labels=xlabels)
    x = np.where(
        np.abs(relevance_time) >= threshold,
        relevance_time + np.sign(relevance_time) * cmap_boost,
        0,
    )
    im = ax.imshow(
        np.expand_dims(x, 0),
        aspect="auto",
        cmap=cmap_time,
        extent=[t[0], t[-1] + 1, np.min(signal_time), np.max(signal_time)],
        alpha=1,
        origin="lower",
    )
    ax.plot(t, signal_time, linewidth=linewidth, color="black")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Relative Signal Amplitude")
    # ax.set_title("signal in time domain")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)

    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

    return save_to


def plot_two_time_attr_bkg(
    signals_time: list,
    relevances_time: list,
    linewidth=1.4,
    threshold=0.0,
    cmap_boost=0.0,
    save_to=None,
):
    figsize = (6, 5)
    fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=300)

    for index, (signal_time, relevance_time) in enumerate(
        zip(signals_time, relevances_time)
    ):
        # Blue to red: "BlueToRed", "coolwarm" or "RdBu_r"
        # Red: "WhiteToRed"
        time_negative_exists = True if np.min(relevance_time) < 0 else False
        cmap_time = "BlueToRed" if time_negative_exists else "WhiteToRed"
        t = np.arange(signal_time.shape[-1])
        xticks = np.arange(0, signal_time.shape[-1] + 1, 800)
        xlabels = [f"{x / 8:.0f}" for x in xticks]
        ax[index].set_xticks(xticks, labels=xlabels)
        x = np.where(
            np.abs(relevance_time) >= threshold,
            relevance_time + np.sign(relevance_time) * cmap_boost,
            0,
        )
        im = ax[index].imshow(
            np.expand_dims(x, 0),
            aspect="auto",
            cmap=cmap_time,
            extent=[t[0], t[-1] + 1, np.min(signal_time), np.max(signal_time)],
            alpha=1,
            origin="lower",
        )
        ax[index].plot(t, signal_time, linewidth=linewidth, color="black")
        ax[index].set_xlabel("Time [ms]")
        ax[index].set_ylabel("Relative Signal Amplitude")
        # ax[index].set_title("signal in time domain")
        ax[index].spines["top"].set_visible(False)
        ax[index].spines["right"].set_visible(False)
        # ax[index].spines["bottom"].set_visible(False)

    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

    return save_to


def plot_two_time_attr_bubbles(
    signals_time: list,
    relevances_time: list,
    linewidth=1.4,
    threshold=0.0,
    cmap_boost=0.0,
    bubble_size=5,
    one_axis_label=False,
    save_to=None,
):
    figsize = (6, 5)
    nrows, ncols = 2, 1
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, dpi=300)

    if one_axis_label:
        # Left shared y-axis label
        fig.text(
            0.00, 0.5, "Relative Signal Amplitude", va="center", rotation="vertical"
        )

    for index, (signal_time, relevance_time) in enumerate(
        zip(signals_time, relevances_time)
    ):
        # Blue to red: "BlueToRed", "coolwarm" or "RdBu_r"
        # Red: "WhiteToRed"
        time_negative_exists = True if np.min(relevance_time) < 0 else False
        cmap_time = "BlueToRed" if time_negative_exists else "WhiteToRed"
        last_plot = (index + 1) == nrows

        # Plot time domain
        t = np.arange(signal_time.shape[-1])
        xticks = np.arange(0, signal_time.shape[-1] + 1, 800)
        xlabels = [f"{x / 8:.0f}" for x in xticks]
        ax[index].set_xticks(xticks, labels=xlabels)
        x = np.where(np.abs(relevance_time) >= threshold)[0]
        y = signal_time[x]
        z = relevance_time[x] + np.sign(relevance_time[x]) * cmap_boost
        ax[index].scatter(
            x,
            y,
            marker="o",
            c=z,
            cmap=cmap_time,
            s=bubble_size,
            zorder=2,
            vmin=np.min(relevance_time),
            vmax=np.max(relevance_time),
        )
        ax[index].plot(t, signal_time, linewidth=linewidth, color="black")
        ax[index].set_xlabel("Time [ms]")
        # ax[index].set_ylabel("Relative Signal Amplitude")
        # ax[index].set_title("signal in time domain")
        ax[index].spines["top"].set_visible(False)
        ax[index].spines["right"].set_visible(False)
        # ax[index].spines["bottom"].set_visible(False)
        ax[index].get_xaxis().set_visible(last_plot)
        ax[index].margins(x=0)

    if one_axis_label:
        plt.tight_layout(rect=[0.01, 0, 1, 1])
    else:
        plt.tight_layout()

    if save_to is not None:
        # plt.savefig(save_to, bbox_inches="tight", dpi=300)
        plt.savefig(save_to, dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

    return save_to


def plot_freq_attr_bkg(
    signal_freq,
    relevance_freq,
    linewidth=1.4,
    threshold=0.0,
    cmap_boost=0.0,
    save_to=None,
):
    # Blue to red: "BlueToRed", "coolwarm" or "RdBu_r"
    # Red: "WhiteToRed"
    freq_negative_exists = True if np.min(relevance_freq) < 0 else False
    cmap_freq = "BlueToRed" if freq_negative_exists else "WhiteToRed"

    figsize = (2, 2.5)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    # Plot frequency domain
    t = np.arange(signal_freq.shape[-1])
    xticks = np.arange(0, signal_freq.shape[-1] + 1, 200)
    ax.set_xticks(xticks)
    x = np.where(
        np.abs(relevance_freq) >= threshold,
        relevance_freq + np.sign(relevance_freq) * cmap_boost,
        0,
    )
    im = ax.imshow(
        np.expand_dims(x, 0),
        aspect="auto",
        cmap=cmap_freq,
        extent=[t[0], t[-1] + 1, np.min(signal_freq), np.max(signal_freq)],
        alpha=1,
        origin="lower",
    )
    ax.plot(t, signal_freq, linewidth=linewidth, color="black")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Relative Frequency Amplitude")
    # ax.set_title("signal in freq. domain")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

    return save_to


def mask_to_intervals(mask):
    """
    mask: 1D array-like of bool or {0,1}
    returns list of (start, width) in index units
    """
    m = np.asarray(mask).astype(bool)
    if m.size == 0:
        return []

    # transitions: False->True start, True->False end
    edges = np.diff(m.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]
