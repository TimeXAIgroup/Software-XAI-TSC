import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.audiomnist_scripts import generate_DFT_example
from utils.logging import add_filehandler, setup_logging
from utils.utils import make_reproducible

logger = logging.getLogger()

ROOT = Path(__file__).parent

GCB_MODEL_PATH = ROOT / "models" / "gcb" / "GCB2020_epoch_20.pt"
GCB_DATA_PATH = ROOT / "models" / "gcb" / "test_data"


def load_figure_sample():
    test_data = np.load(GCB_DATA_PATH / "test_data.npy").astype(np.float32)
    test_labels = np.load(GCB_DATA_PATH / "test_labels.npy")
    index_sample = 28
    mask = test_labels == 1
    test_rbbb_labels = test_labels[mask]
    test_rbbb_data = test_data[mask]
    figure_sample = test_rbbb_data[index_sample]
    figure_label = test_rbbb_labels[index_sample]
    return test_data, test_labels, figure_sample, figure_label


def run_experiments(out_path, out_figures, prng, device):
    from experiments.compare_methods import (
        deconv_comp,
        fa_comp,
        ig_comp,
        occ_comp,
        quantify_ig_comp,
        sal_comp,
    )
    from experiments.compare_metrics import compare_metrics

    out_experiments = out_path / "experiments"
    test_data, test_labels, figure_sample, figure_label = load_figure_sample()

    # Metrics comparison
    out_path_metrics = out_experiments / "metrics_comp"
    out_path_metrics.mkdir(parents=True, exist_ok=True)
    compare_metrics(
        GCB_MODEL_PATH, figure_sample, figure_label, prng, device, out_path_metrics
    )

    # Method comparisons
    out_path_methods = out_experiments / "method_comp"
    out_path_methods.mkdir(parents=True, exist_ok=True)
    add_filehandler(logger, out_path_methods / "methods_experiment.log")

    out_path_ig = out_path_methods / "integrated_gradients_comp"
    out_path_ig.mkdir(parents=True, exist_ok=True)
    ig_comp(
        GCB_MODEL_PATH,
        figure_sample,
        figure_label,
        out_path_ig,
        device,
        figures_path=out_figures / "ig_comparisons",
    )
    quantify_ig_comp(
        GCB_MODEL_PATH,
        test_data,
        test_labels,
        len(test_data),
        device,
        prng,
        out_path=out_path_ig,
    )

    out_path_sal = out_path_methods / "saliency_comp"
    out_path_sal.mkdir(parents=True, exist_ok=True)
    sal_comp(GCB_MODEL_PATH, figure_sample, figure_label, out_path_sal, device)

    out_path_deconv = out_path_methods / "deconv_comp"
    out_path_deconv.mkdir(parents=True, exist_ok=True)
    deconv_comp(GCB_MODEL_PATH, figure_sample, figure_label, out_path_deconv, device)

    out_path_occ = out_path_methods / "occlusion_comp"
    out_path_occ.mkdir(parents=True, exist_ok=True)
    occ_comp(GCB_MODEL_PATH, figure_sample, figure_label, out_path_occ, device)

    out_path_fa = out_path_methods / "feature_ablation_comp"
    out_path_fa.mkdir(parents=True, exist_ok=True)
    fa_comp(GCB_MODEL_PATH, figure_sample, figure_label, out_path_fa, device)


if __name__ == "__main__":
    out_path = ROOT / "output"
    out_figures = out_path / "figures"
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "experiments").mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    file_path = Path(__file__)
    setup_logging(logger, (out_path / (file_path.name + "_log")).with_suffix(".txt"))
    logger.info("This logfile was made from file:\n%s", file_path)
    prng = make_reproducible(0xDEADBEEF, use_legacy=True)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    run_experiments(out_path, out_figures, prng, device)

    # Paper figures not tied to a method comparison
    generate_DFT_example(out_figures, device)
    from experiments.compare_methods import generate_rbbb_fig_one, generate_rbbb_m_GT

    _, _, figure_sample, _ = load_figure_sample()
    generate_rbbb_m_GT(figure_sample, out_figures)
    generate_rbbb_fig_one(figure_sample, out_figures)
