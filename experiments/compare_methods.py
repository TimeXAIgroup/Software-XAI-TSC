import logging
from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import (
    Deconvolution,
    FeatureAblation,
    IntegratedGradients,
    Occlusion,
    Saliency,
)
from quantus.functions.explanation_func import (
    generate_captum_explanation,
    generate_zennit_explanation,
)
from scipy.stats import spearmanr
from tint.attr import FeatureAblation as tintFeatureAblation
from tint.attr import Occlusion as tintOcclusion
from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_PTY import (
    IntegratedGradients as tsiIntegratedGradients,
)
from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_PTY import NoiseTunnel
from zennit.attribution import Gradient as zGradient
from zennit.attribution import IntegratedGradients as zIntegratedGradients
from zennit.attribution import Occlusion as zOcclusion
from zennit.attribution import SmoothGrad as zSmoothGrad
from zennit.attribution import occlude_independent
from zennit.composites import DeconvNet, EpsilonPlusFlat

from models.models import GCB2020
from utils.logging import add_filehandler
from utils.utils import rescale_array
from utils.viz import plot_ecg

logger = logging.getLogger()


def attr_target(output: torch.Tensor) -> torch.Tensor:
    """Returns an one-hot encoded vector of the predicted class in the output."""
    # output: (B, C)
    B, C = output.shape
    # Get target class (most probable)
    pred = output.argmax(dim=1)
    one_hot_class = F.one_hot(pred, num_classes=C).to(dtype=output.dtype)

    # One-hot for each sample's target class; dtype must match output dtype
    # weights = F.one_hot(torch.tensor(y), num_classes=C).to(dtype=output.dtype)

    return one_hot_class


def occ_target(output: torch.Tensor, static_output=None) -> torch.Tensor:
    """Returns the difference between the models original output vs the perturbed output."""
    if static_output is not None:
        occ_target.static_output = static_output
    # output: (B, C)
    B, C = output.shape
    # Get target class (most probable)
    pred = output.argmax(dim=1)
    # Set not predicted logit 0
    not_pred = output.argmin(dim=1).item()
    output[:, not_pred] = 0
    # Get difference to original input
    if occ_target.static_output is not None:
        ret = occ_target.static_output - output
    else:
        ret = output
    return ret


def load_model(model_class, model_path, eval=True, device="cpu", **kwargs):
    # Reload model to ensure same environment
    model = model_class(**kwargs).to(device)
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)
    )
    if eval:
        model.eval()
    return model


def deconv_comp(model_path, sample, label, out_path, device):
    # Parameter:
    input = torch.tensor(sample, device=device).unsqueeze(0)
    target = (
        label.item()
    )  # F.one_hot(torch.tensor(label, device=device), num_classes=2)

    # Reload model to ensure same environment
    model = load_model(GCB2020, model_path, True, device, in_channels=12)

    # Captum
    logger.info("Generating Captum Deconvolution attribution...")
    # Reload model weights from disk
    c_input = input.detach().clone().requires_grad_(True)
    c_method = Deconvolution(model)
    c_attr = c_method.attribute(inputs=c_input, target=target)

    # Zennit
    logger.info("Generating Zennit Deconvolution attribution...")
    # Reload model to ensure same environment
    model = load_model(GCB2020, model_path, True, device, in_channels=12)

    z_input = input.detach().clone().requires_grad_(True)
    with zGradient(model=model, composite=DeconvNet()) as attributor:
        out, z_attr = attributor(input=z_input, attr_output=attr_target)

    comparisons = [{"captum": c_attr.squeeze(), "zennit": z_attr.squeeze()}]

    # Write results
    for comp in comparisons:
        for name, attr in comp.items():
            # Plot comparison
            plot_ecg(
                ecg=sample,
                explanation=scale_rel_pos(attr.detach().cpu().numpy()),
                title=name,
                shape_switch=False,
                save_to=out_path / f"{name}_deconvolution.pdf",
            )
        attr = list(comp.values())
        names = list(comp.keys())
        measurements = {
            "max abs diff": (attr[0] - attr[1]).abs().max().item(),
            "mean abs diff": (attr[0] - attr[1]).abs().mean().item(),
            "L2 diff": torch.linalg.norm((attr[0] - attr[1]).reshape(-1)).item(),
            "allclose(rtol=1e-4,atol=1e-6)": torch.allclose(
                attr[0], attr[1], rtol=1e-4, atol=1e-6
            ),
        }
        df = pd.DataFrame([measurements])
        df.to_csv(out_path / f"comp_{names[0]}_{names[1]}.csv", index=False)
        pass


def occ_comp(model_path, sample, label, out_path, device):
    # Compare Occlusion: Results between Captum and Zennit based explanations will be different, if the stride is
    # different than the window size. Captum stops at the end, while a Zennit implementation "overflows" at the back to
    # the start. (variables win_T & str_T)
    input = torch.tensor(sample, device=device).unsqueeze(0)
    target = torch.tensor(label, device=device)
    max_len = input.shape[-1]
    win_T = max_len // 10
    str_T = max_len // 10
    occ_window = (input.shape[-2], win_T)  # (C,T)
    occ_stride = (input.shape[-2], str_T)  # (C,T)
    baseline = torch.zeros_like(input)
    baseline_fn = torch.zeros_like

    comp_occ = {}
    # Captum
    logger.info("Generating Captum Occlusion attribution...")
    # Reload model to ensure same environment
    model = load_model(GCB2020, model_path, True, device, in_channels=12)
    c_input = input.detach().clone().requires_grad_(True)
    c_occ = Occlusion(model)
    c_occ_attr = c_occ.attribute(
        inputs=c_input,
        target=target,
        sliding_window_shapes=occ_window,
        strides=occ_stride,
        baselines=baseline,
    )  # (1,1,T)

    # Tint
    logger.info("Generating time_interpret Occlusion attribution...")
    # Reload model to ensure same environment
    model = load_model(GCB2020, model_path, True, device, in_channels=12)
    tint_input = input.detach().clone().requires_grad_(True)
    tint_occ = tintOcclusion(model)
    tint_occ_attr = tint_occ.attribute(
        inputs=tint_input,
        target=target,
        baselines=baseline,
        sliding_window_shapes=occ_window,
        strides=occ_stride,
    )  # (1,T,1)

    # Zennit
    logger.info("Generating Zennit Occlusion attribution...")
    # Reload model to ensure same environment
    model = load_model(GCB2020, model_path, True, device, in_channels=12)
    z_setup_input = input.detach().clone().requires_grad_(True)
    # Invert occlusion mask to match other methods
    occlusion_fn = partial(
        occlude_independent,
        fill_fn=torch.zeros_like,  # replace occluded region with 0
        invert=True,  # occlude *inside* mask
    )
    # Occlude with respect to the input
    output = model(z_setup_input)
    not_pred = output.argmin(dim=1).item()
    output[:, not_pred] = 0
    occ_target(output, static_output=output)
    z_input = input.detach().clone().requires_grad_(True)
    with zOcclusion(
        model=model,
        composite=None,
        window=win_T,  # occ_window,  # win_T,
        stride=str_T,  # occ_stride,  # str_T,
        occlusion_fn=occlusion_fn,
    ) as attributor:
        out, z_occ_attr = attributor(
            input=z_input, attr_output=occ_target
        )  # target_logit)  # attr_target)
    pass
    initial_out = out[:, target]
    initial_out_mask = initial_out * 2 * torch.ones(z_occ_attr.shape[-1], device=device)
    pass

    entries = {
        "Captum": c_occ_attr.squeeze(),
        "Zennit": z_occ_attr.squeeze(),
        "Time_Interpret": tint_occ_attr.squeeze(),
    }

    comparisons = [dict(combo) for combo in combinations(entries.items(), 2)]
    # Write results
    for comp in comparisons:
        for name, attr in comp.items():
            # Plot comparison
            plot_ecg(
                ecg=sample,
                explanation=scale_rel_pos(attr.detach().cpu().numpy()),
                title=name,
                shape_switch=False,
                save_to=out_path / f"{name}_occlusion.pdf",
            )
        attr = list(comp.values())
        names = list(comp.keys())
        measurements = {
            "max abs diff": (attr[0] - attr[1]).abs().max().item(),
            "mean abs diff": (attr[0] - attr[1]).abs().mean().item(),
            "L2 diff": torch.linalg.norm((attr[0] - attr[1]).reshape(-1)).item(),
            "allclose(rtol=1e-4,atol=1e-6)": torch.allclose(
                attr[0], attr[1], rtol=1e-4, atol=1e-6
            ),
        }
        df = pd.DataFrame([measurements])
        df.to_csv(out_path / f"comp_{names[0]}_{names[1]}.csv", index=False)


def fa_comp(model_path, sample, label, out_path, device):
    # Parameter:
    input = torch.tensor(sample, device=device).unsqueeze(0)
    target = torch.tensor(label, device=device)
    baseline = torch.zeros_like(input)

    comp_fa = {}
    # Captum
    logger.info("Generating Captum Feature Ablation attribution...")
    # Reload model to ensure same environment
    model = load_model(GCB2020, model_path, True, device, in_channels=12)
    c_input = input.detach().clone().requires_grad_(True)
    c_fa_method = FeatureAblation(model)
    c_fa_attr = c_fa_method.attribute(inputs=c_input, target=target, baselines=baseline)

    # Tint (=time_interpret)
    logger.info("Generating time_interpret Feature Ablation attribution...")
    # Reload model to ensure same environment
    model = load_model(GCB2020, model_path, True, device, in_channels=12)
    tint_input = input.detach().clone().requires_grad_(True)
    tint_fa = tintFeatureAblation(model)
    tint_fa_attr = tint_fa.attribute(
        inputs=tint_input, target=target, baselines=baseline
    )  # (1,T,1)
    comp_fa.update({f"captum_time_interpret_fa": (c_fa_attr, tint_fa_attr)})

    entries = {"Captum": c_fa_attr.squeeze(), "Time_Interpret": tint_fa_attr.squeeze()}

    comparisons = [dict(combo) for combo in combinations(entries.items(), 2)]
    # Write results
    for comp in comparisons:
        for name, attr in comp.items():
            # Plot comparison
            plot_ecg(
                ecg=sample,
                explanation=scale_rel_pos(attr.detach().cpu().numpy()),
                title=name,
                shape_switch=False,
                save_to=out_path / f"{name}_feature_attribution.pdf",
            )
        attr = list(comp.values())
        names = list(comp.keys())
        measurements = {
            "max abs diff": (attr[0] - attr[1]).abs().max().item(),
            "mean abs diff": (attr[0] - attr[1]).abs().mean().item(),
            "L2 diff": torch.linalg.norm((attr[0] - attr[1]).reshape(-1)).item(),
            "allclose(rtol=1e-4,atol=1e-6)": torch.allclose(
                attr[0], attr[1], rtol=1e-4, atol=1e-6
            ),
        }
        df = pd.DataFrame([measurements])
        df.to_csv(out_path / f"comp_{names[0]}_{names[1]}.csv", index=False)


def sal_comp(model_path, sample, label, out_path, device):
    # Parameter:
    input = torch.tensor(sample, device=device).unsqueeze(0)
    target = torch.tensor(label, device=device)

    # Reload model to ensure same environment
    model = load_model(GCB2020, model_path, True, device, in_channels=12)

    # Captum
    logger.info("Generating Captum Saliency attribution...")
    # Reload model weights from disk
    c_input = input.detach().clone().requires_grad_(True)
    c_s_method = Saliency(model)
    c_s_attr = c_s_method.attribute(inputs=c_input, target=target, abs=False)

    # Zennit
    logger.info("Generating Zennit Gradient attribution...")
    # Reload model to ensure same environment
    model = load_model(GCB2020, model_path, True, device, in_channels=12)
    z_input = input.detach().clone().requires_grad_(True)
    with zGradient(model=model, composite=None) as attributor:
        out, z_g_attr = attributor(input=z_input, attr_output=attr_target)

    entries = {"Captum": c_s_attr.squeeze(), "Zennit": z_g_attr.squeeze()}

    comparisons = [dict(combo) for combo in combinations(entries.items(), 2)]
    # Write results
    for comp in comparisons:
        for name, attr in comp.items():
            # Plot comparison
            plot_ecg(
                ecg=sample,
                explanation=scale_rel_pos(attr.detach().cpu().numpy()),
                title=name,
                shape_switch=False,
                save_to=out_path / f"{name}_saliency.pdf",
            )
        attr = list(comp.values())
        names = list(comp.keys())
        comparison = {
            "max abs diff": (attr[0] - attr[1]).abs().max().item(),
            "mean abs diff": (attr[0] - attr[1]).abs().mean().item(),
            "L2 diff": torch.linalg.norm((attr[0] - attr[1]).reshape(-1)).item(),
            "allclose(rtol=1e-4,atol=1e-6)": torch.allclose(
                attr[0], attr[1], rtol=1e-4, atol=1e-6
            ),
        }
        df = pd.DataFrame([comparison])
        df.to_csv(out_path / f"comp_{names[0]}_{names[1]}.csv", index=False)


def ig_comp(model_path, sample, label, out_path, device, figures_path=None):
    # Parameter:
    baseline_fn = torch.zeros_like
    n_steps_list = [10, 50]

    # Prepare vars
    comparison_plotted = False
    target = torch.tensor(label)
    input = torch.tensor(sample, device=device).unsqueeze(dim=0)
    # Clone inputs
    z_input = input.detach().clone()
    q_z_input = input.detach().clone()
    # Plot
    file_type = "pdf"
    plot_path = out_path / "all_plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    for n_steps in n_steps_list:
        comparison_plotted = False
        # Zennit based methods first:
        # ----------------------------------------------
        # Calculate Quantus IG (Zennit):
        # Reload model to ensure same environment
        model = load_model(GCB2020, model_path, True, device, in_channels=12)
        q_z_ig_rel = generate_zennit_explanation(
            model=model,
            inputs=q_z_input,
            targets=target,  # F.one_hot(target, num_classes=2),
            device=device,
            attributor=zIntegratedGradients,
            baseline=baseline_fn(q_z_input),
            reduce_axes=(),
            attributor_kwargs={
                "n_iter": n_steps,
                "baseline_fn": torch.zeros_like,
                "composite": None,
            },
        )
        q_z_ig_rel = scale_rel_pos(q_z_ig_rel)
        name = "quantus_zennit"
        plot_ecg(
            ecg=sample,
            explanation=q_z_ig_rel.squeeze(),
            title=name,
            shape_switch=False,
            save_to=plot_path / f"{name}_{n_steps}.{file_type}",
        )
        logger.info(
            "Calculated Quantus (Zennit) relevance for the figure sample with %s steps.",
            n_steps,
        )
        # Calculate Zennit/SignXAI2 IG:
        # Reload model to ensure same environment
        model = load_model(GCB2020, model_path, True, device, in_channels=12)
        with zIntegratedGradients(
            model=model, n_iter=n_steps, baseline_fn=torch.zeros_like, composite=None
        ) as attributor:
            out, z_ig_rel = attributor(input=z_input, attr_output=attr_target)
        z_ig_rel = scale_rel_pos(z_ig_rel)
        name = "signxai2_zennit"
        plot_ecg(
            ecg=sample,
            explanation=z_ig_rel.squeeze(),
            title=name,
            shape_switch=False,
            save_to=plot_path / f"{name}_{n_steps}.{file_type}",
        )
        logger.info(
            "Calculated SIGNXAI2 (Zennit) relevance for the figure sample with %s steps.",
            n_steps,
        )

        # Captum based Integrated gradients with different integral approx. methods:
        # --------------------------------------------------------------------------
        for method in [
            "riemann_trapezoid",
            "riemann_right",
            "riemann_left",
            "riemann_middle",
            "gausslegendre",
        ]:
            # Prepare separate cloned inputs
            tsi_input = input.detach().clone()
            q_c_input = input.detach().clone()

            # Calculate TSInterpret IG:
            # Reload model to ensure same environment
            model = load_model(GCB2020, model_path, True, device, in_channels=12)
            tsi_ig = tsiIntegratedGradients(model)
            tsi_ig_rel = tsi_ig.attribute(
                inputs=tsi_input,
                baselines=baseline_fn(tsi_input),
                target=target,
                n_steps=n_steps,
                method=method,
            )
            tsi_ig_rel = scale_rel_pos(tsi_ig_rel)
            name = f"tsi_{method}"
            plot_ecg(
                ecg=sample,
                explanation=tsi_ig_rel.squeeze(),
                title=name,
                shape_switch=False,
                save_to=plot_path / f"{name}_{n_steps}.{file_type}",
            )
            logger.info(
                "Calculated TSInterpret relevance for the figure sample with integral approx. method %s and %s steps.",
                method,
                n_steps,
            )

            # Calculate Quantus IG (captum, 10 steps fixed):
            if n_steps == 50:
                logger.info(
                    "Unable to run Quantus (Captum) with 50 steps (Parameter not configurable)."
                )
            else:
                # Reload model to ensure same environment
                model = load_model(GCB2020, model_path, True, device, in_channels=12)
                q_c_ig_rel = generate_captum_explanation(
                    model=model,
                    inputs=q_c_input,
                    targets=target,
                    device=device,
                    method="IntegratedGradients",
                    baseline=baseline_fn(q_c_input),
                    reduce_axes=(),
                    # xai_lib_kwargs={"baselines": baseline_fn(q_c_input), "n_steps": n_steps},
                )
                q_c_ig_rel = scale_rel_pos(q_c_ig_rel)
                name = f"quantus_captum_{method}"
                plot_ecg(
                    ecg=sample,
                    explanation=q_c_ig_rel.squeeze(),
                    title=name,
                    shape_switch=False,
                    save_to=plot_path / f"{name}_{n_steps}.{file_type}",
                )
                logger.info(
                    "Calculated Quantus (Captum) relevance for the figure sample with integral approx. method %s and %s steps.",
                    method,
                    n_steps,
                )
            # Only plot comparison once, and only for 50 steps (paper figure)
            if not comparison_plotted and n_steps == 50:
                comparison_plotted = True
                names = [f"tsi_{method}_{n_steps}", f"signxai2_(zennit)_{n_steps}"]
                plot_ig_comp_parts(
                    sample,
                    relevances=[tsi_ig_rel, z_ig_rel],
                    names=names,
                    out_path=out_path / "figure_plots",
                )
                if figures_path is not None:
                    figures_path.mkdir(parents=True, exist_ok=True)
                    plot_ig_comp_parts(
                        sample,
                        relevances=[tsi_ig_rel, z_ig_rel],
                        names=names,
                        out_path=figures_path,
                    )
                rel_one = (
                    tsi_ig_rel
                    if isinstance(tsi_ig_rel, torch.Tensor)
                    else torch.tensor(tsi_ig_rel)
                )
                rel_two = (
                    z_ig_rel
                    if isinstance(z_ig_rel, torch.Tensor)
                    else torch.tensor(z_ig_rel)
                )
                comparison = {
                    "max abs diff": (rel_one - rel_two).abs().max().item(),
                    "mean abs diff": (rel_one - rel_two).abs().mean().item(),
                    "L2 diff": torch.linalg.norm(
                        (rel_one - rel_two).reshape(-1)
                    ).item(),
                    "allclose(rtol=1e-4,atol=1e-6)": torch.allclose(
                        rel_one, rel_two, rtol=1e-4, atol=1e-6
                    ),
                }
                df = pd.DataFrame([comparison])
                df.to_csv(
                    out_path
                    / f"comp_tsi_{method}_{n_steps}_signxai2_(zennit)_{n_steps}.csv",
                    index=False,
                )


def quantify_ig_comp(model_path, test_data, test_labels, n, device, prng, out_path):
    add_filehandler(logger, out_path / "quantify_ig_comp.log")
    logger.info("Quantifying IG differences arcoss %s samples", n)

    # draw n random samples of same index
    if n < len(test_data):
        if prng is None:
            indices = np.random.choice(len(test_data), size=n, replace=False)
        else:
            indices = prng.choice(len(test_data), size=n, replace=False)
        data = test_data[indices]
        labels = test_labels[indices]
    else:
        data = test_data
        labels = test_labels

    # Set IG variables
    n_steps = 50
    baseline_fn = torch.zeros_like

    # load model and generate explainers
    model = load_model(GCB2020, model_path, True, device, in_channels=12)
    tsi_ig = tsiIntegratedGradients(model)
    model = load_model(GCB2020, model_path, True, device, in_channels=12)
    with zIntegratedGradients(
        model=model, n_iter=n_steps, baseline_fn=baseline_fn, composite=None
    ) as attributor:
        # Attribute for all samples
        tsi_ig_attr = []
        z_ig_attr = []
        class_healthy_count = 0
        class_rbbb_count = 0
        for index, (sample, label) in enumerate(zip(data, labels)):
            # set up variables
            if label:
                class_rbbb_count += 1
                logger.info(
                    "Sample %s/%s, class %s count %s",
                    index,
                    n,
                    label,
                    class_rbbb_count,
                )
            else:
                class_healthy_count += 1
                logger.info(
                    "Sample %s/%s, class %s count %s",
                    index,
                    n,
                    label,
                    class_healthy_count,
                )
            target = torch.tensor(label)
            tsi_input = (
                torch.tensor(sample, device=device)
                .unsqueeze(dim=0)
                .requires_grad_(True)
            )
            z_input = tsi_input.detach().clone().requires_grad_(True)

            # calculate TSI IG
            tsi_ig_rel = tsi_ig.attribute(
                inputs=tsi_input,
                baselines=baseline_fn(tsi_input),
                target=target,
                n_steps=n_steps,
            )
            tsi_ig_attr.append(tsi_ig_rel)

            # calculate Zennit IG
            out, z_ig_rel = attributor(
                input=z_input,
                attr_output=F.one_hot(target, num_classes=2).unsqueeze(0).to(device),
            )
            z_ig_attr.append(z_ig_rel)
        logger.info("Test dataset contains %s samples of class 0", class_healthy_count)
        logger.info("Test dataset contains %s samples of class 1", class_rbbb_count)
    result = compare_attributions(tsi_ig_attr, z_ig_attr)
    logger.info("Result across all %s samples:", n)
    for k, v in result.items():
        logger.info("%s: %s", k, v)
    return result


def compare_attributions(attr_a, attr_b, top_k_frac=0.05):
    """
    attr_a, attr_b: arrays of shape (N, ...) — N instances, any attribution shape.
    Returns mean Spearman correlation and mean top-k Jaccard across instances.
    """
    (
        spearmans,
        jaccards,
    ) = [], []
    for a, b in zip(attr_a, attr_b):
        a_flat = np.asarray(a.detach().cpu()).ravel()
        b_flat = np.asarray(b.detach().cpu()).ravel()

        # Spearman on absolute relevance
        # rho, _ = spearmanr(np.abs(a_flat), np.abs(b_flat))
        # Spearman on signed relevance
        rho, _ = spearmanr(a_flat, b_flat)
        spearmans.append(rho)

        # Top-k Jaccard
        k = max(1, int(top_k_frac * a_flat.size))
        top_a = set(np.argpartition(-np.abs(a_flat), k)[:k])
        top_b = set(np.argpartition(-np.abs(b_flat), k)[:k])
        jaccards.append(len(top_a & top_b) / len(top_a | top_b))

    spearmans = np.array(spearmans)
    spearman_with_agreement = (spearmans > 0.5).sum()
    jaccards = np.array(jaccards)

    return {
        "spearman_mean": spearmans.mean(),
        "spearman_median": np.median(spearmans),
        "spearman_above_0.5": spearman_with_agreement,
        "spearman_std": spearmans.std(),
        "jaccard_mean": jaccards.mean(),
        "jaccard_std": jaccards.std(),
    }


def plot_ig_comp_parts(sample, relevances, names, out_path):
    # Prepare comparison
    leads = [6]  # [6, 11]
    start = 500
    stop = 1800
    file_type = "pdf"
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / f"ig_rel_comp_rbbb_{names[0]}_{names[1]}.{file_type}"
    plot_ig_comp(rescale_array(sample), relevances, leads, start, stop, path)
    logger.info("Saved IG relevances comparison for the figure sample under:\n%s", path)

    # Plot three beats
    lead = 6
    start = 500
    stop = 1550
    save_path = out_path / f"three_beats_{names[0]}.{file_type}"
    generate_rbbb_part(sample, relevances[0], lead, start, stop, save_path)
    save_path = out_path / f"three_beats_{names[1]}.{file_type}"
    generate_rbbb_part(sample, relevances[1], lead, start, stop, save_path)


def scale_rel_pos(relevance):
    if isinstance(relevance, torch.Tensor):
        relevance = relevance.detach().cpu().numpy().squeeze()
    relevance[relevance < 0] = 0
    relevance = rescale_array(relevance, new_min=0, new_max=1)
    relevance = np.nan_to_num(relevance, nan=0)
    return relevance


def plot_ig_comp(sample, relevances, leads, start, stop, save_to):
    # Settings
    linewidth = 1.2

    # Leads
    lead_index = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    # Scale signals, keep specified leads
    lead_index = [lead_index[x] for x in leads] * 2
    lead_indices = np.array(leads)
    ecg = sample[..., lead_indices, start:stop]

    # Create and scale positive only relevances
    relevance_one = relevances[0][..., lead_indices, start:stop]
    relevance_two = relevances[1][..., lead_indices, start:stop]

    if len(ecg.shape) == 1:
        ecg = np.expand_dims(ecg, 0)
        relevance_one = np.expand_dims(relevance_one, 0)
        relevance_two = np.expand_dims(relevance_two, 0)
    plot_ecg(
        np.concatenate([ecg, ecg]),
        np.concatenate([relevance_one, relevance_two]),
        title="",
        lead_index=lead_index,
        columns=1,
        line_width=linewidth,
        shape_switch=False,
        save_to=save_to,
        add_middle_sep=True,
        omit_write_speed=True,
    )


def generate_rbbb_part(figure_sample, relevance, lead, start, stop, save_path):
    linewidth = 0.8

    ecg = figure_sample[lead, start:stop]
    rel = relevance[lead, start:stop]

    if len(ecg.shape) == 1:
        ecg = np.expand_dims(ecg, 0)
        rel = np.expand_dims(rel, 0)

    plot_ecg(
        ecg=ecg,
        explanation=rel,
        title="",
        lead_index=["V1"],
        columns=1,
        row_height=6,
        line_width=linewidth,
        shape_switch=False,
        save_to=save_path,
        omit_write_speed=True,
        move_title=True,
        additional_y_shift=-0.45,
    )


def generate_rbbb_m_GT(figure_sample, save_path):
    # Cut out a single heartbeat
    lead = 6
    start = 500
    stop = 1550
    linewidth = 0.8
    # figure_sample = figure_sample[6, :]
    figure_sample = figure_sample[6, start:stop]

    gt = np.zeros_like(figure_sample)
    # gt[169:218] = 1

    plot_ecg(
        ecg=np.expand_dims(figure_sample, 0),
        explanation=np.expand_dims(gt, 0),
        title="",
        lead_index=[""],
        columns=1,
        line_width=linewidth,
        shape_switch=False,
        save_to=save_path / "rbbb_gt.svg",
        omit_write_speed=True,
        gt=True,
        move_title=True,
        show_grid=False,
    )


def generate_rbbb_fig_one(figure_sample, save_path):
    # Cut out a single heartbeat
    start = 200
    stop = 3200
    linewidth = 0.8

    figure_sample = figure_sample[..., start:stop]
    gt = np.zeros_like(figure_sample)

    plot_ecg(
        ecg=figure_sample,
        explanation=gt,
        title="",
        columns=2,
        line_width=linewidth,
        shape_switch=False,
        save_to=save_path / "rbbb_fig_one.svg",
        omit_write_speed=True,
        show_grid=True,
    )
