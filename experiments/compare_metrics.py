import logging
import math

import numpy as np
import pandas as pd
import quantus as q
import torch

from models.models import GCB2020
from utils.logging import add_filehandler

logger = logging.getLogger()


def compare_metrics(
    model_ckpt_path, figure_sample, figure_label, prng, device, out_path
):
    model = GCB2020(in_channels=12).to(device)
    model.load_state_dict(
        torch.load(model_ckpt_path, weights_only=True, map_location=device)
    )
    model.eval()

    # Add filehandler for metrics log
    add_filehandler(logger, out_path / "metrics_experiment.log")

    x_np = figure_sample
    if len(x_np.shape) == 2:
        x_np = np.expand_dims(x_np, axis=0)
    x = torch.tensor(x_np, device=device)
    target = torch.tensor(figure_label, device=device)
    # Expects (B,C,T) -> Batch,Channels,Timesteps
    with torch.no_grad():
        out = model(x)
        target = out.argmax(dim=1).item()
    assert target == figure_label

    # Seed once for reproducibility
    random_state_seed = prng.randint(2**32 - 1)
    np.random.seed(random_state_seed)
    logger.info(
        "Set numpy random seed to %s, sampled from initial seed.", random_state_seed
    )

    # Use default gradient
    from captum.attr import Saliency

    saliency = Saliency(model)
    attr = saliency.attribute(x, target)

    # Save global RNG state
    state = np.random.get_state()
    # ROC-AUC: Quantus, tint
    from tint.metrics.white_box import roc_auc

    # Set GT for all attr values above 0.4
    attr_np = attr.detach().cpu().numpy().squeeze()
    attr_np_norm = attr_np - attr_np.min()
    attr_np_norm = attr_np_norm / attr_np.max()
    mask = attr_np_norm >= attr_np_norm.max() - 0.6
    gt = np.zeros_like(attr_np_norm)
    gt[mask] = 1

    # Add block of GT
    x_len = x_np.shape[-1]
    start = x_len // 2 - math.ceil(x_len * 0.1)
    stop = x_len // 2 + math.ceil(x_len * 0.1)
    gt[:, start:stop] = 1
    gt = gt[np.newaxis]

    results = []
    for lead in range(x_np.shape[1]):
        # Reset to same random state
        np.random.set_state(state)
        logger.info("Resetting to saved random state with seed: %s", random_state_seed)
        q_auc = q.AUC()
        q_auc_res = q_auc(
            model=None,  # model,
            x_batch=x_np[np.newaxis, :, lead, :],
            y_batch=np.array(target)[np.newaxis],
            a_batch=attr_np_norm[np.newaxis, np.newaxis, lead, :],
            s_batch=gt[np.newaxis, :, lead, :],
            # explain_func=saliency_wrapper,
            # explain_func_kwargs={"target": target},
            device=device,
        )
        logger.info("Quantus lead %s roc_auc: %s", lead, q_auc_res[0])

        np.random.set_state(state)
        logger.info("Resetting to saved random state with seed: %s", random_state_seed)
        tint_auc_res = roc_auc(
            attributions=torch.tensor(attr_np_norm[np.newaxis, lead, :]),
            true_attributions=torch.tensor(gt[:, lead, :]),
        )
        logger.info("time_interpret lead %s roc_auc: %s", lead, tint_auc_res)
        results.append(
            {"lead": lead, "quantus_auc": q_auc_res[0], "tint_auc": float(tint_auc_res)}
        )

    pd.DataFrame(results).to_csv(out_path / "comp_auc.csv", index=False)
