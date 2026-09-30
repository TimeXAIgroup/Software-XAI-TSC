import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from signxai2.dftutils import DFTLRP
from zennit.attribution import Gradient
from zennit.composites import EpsilonAlpha2Beta1

from models.models import AudioNet
from utils.utils import (
    plot_multi_dft_attr,
    rescale_array,
)

_ROOT = Path(__file__).parent.parent
AUDIOMNIST_MODEL_DIR_PATH = _ROOT / "models" / "audionet"
AUDIOMNIST_SAMPLE_FEMALE = (
    _ROOT / "models" / "audionet" / "samples" / "sample_female.pt"
)
AUDIOMNIST_SAMPLE_MALE = _ROOT / "models" / "audionet" / "samples" / "sample_male.pt"

logger = logging.getLogger()


def generate_DFT_example(out_path, device):
    out_path = out_path / "audio_mnist"
    out_path.mkdir(parents=True, exist_ok=True)

    model = AudioNet(output_size=2).to(device)
    model.load_state_dict(
        torch.load(
            AUDIOMNIST_MODEL_DIR_PATH / "audionet_gender_digit9.pt",
            weights_only=True,
            map_location=device,
        )
    )
    model.eval()

    samples_dict = {}
    for key, path in [
        ("female", AUDIOMNIST_SAMPLE_FEMALE),
        ("male", AUDIOMNIST_SAMPLE_MALE),
    ]:
        sample = torch.load(path, weights_only=False)
        data = sample["signal"].unsqueeze(0).to(device)
        target = torch.tensor(sample["label"], device=device)
        assert model(data).argmax(dim=1) == target, f"Model misclassifies {path.name}"
        signal_time = data.detach().clone().cpu().numpy()
        results = {"signal_time": signal_time, "target": target.item()}
        results = calc_eps2a1b(model, target, data, results, device)
        samples_dict[key] = results

    female_sample = (400, 950, samples_dict["female"])
    male_sample = (85, 635, samples_dict["male"])
    plot_figure_4(out_path, female_sample, male_sample)


def calc_eps2a1b(model, target, data, results, device):
    # compute relevance values in time domain with lrp (zennit)
    signal_time = data.detach().clone().cpu().numpy()
    z_input = (
        torch.tensor(signal_time, dtype=torch.float32).to(device).requires_grad_(True)
    )
    with Gradient(model=model, composite=EpsilonAlpha2Beta1()) as attributor:
        _, rel = attributor(
            input=z_input,
            attr_output=lambda o: (
                o * F.one_hot(target.view(-1), num_classes=o.shape[1]).to(dtype=o.dtype)
            ),
        )
    relevance_time = rel.detach().cpu().numpy()
    results["relevance_time"] = relevance_time

    # compute relevance values in frequency domain
    dftlrp = DFTLRP(
        data.shape[-1],
        leverage_symmetry=True,
        precision=32,
        create_stdft=False,
        create_inverse=False,
        cuda=False,
    )

    signal_freq = np.empty(
        (data.shape[-2], data.shape[-1] // 2 + 1), dtype=np.complex128
    )
    relevance_freq = np.empty((data.shape[-2], data.shape[-1] // 2 + 1))
    signal_freq, relevance_freq = dftlrp.dft_lrp(
        relevance_time,
        signal_time,
        real=False,
        short_time=False,
    )
    results["signal_freq"] = signal_freq
    results["relevance_freq"] = relevance_freq
    results["target"] = target
    return results


def plot_figure_4(out_path, female_sample, male_sample):
    # Figure settings
    boost = 0.0
    threshold = 0.1
    max_freq = 800
    linewidth = 0.4
    ftype = "pdf"
    font_size = 8
    name = f"eps2a1b_pos_bubbles_female_male"
    save_to = out_path / "Figure_4"
    save_to.mkdir(parents=True, exist_ok=True)
    save_to = save_to / (name + f".{ftype}")

    signals_time = []
    signals_freq = []
    relevances_time = []
    relevances_time_pos = []
    relevances_freq = []
    relevances_freq_pos = []
    gts_freq = []
    # Get female data
    data_female = female_sample[2]
    target = data_female["target"]
    f_signal_time = data_female["signal_time"]
    f_signal_freq = data_female["signal_freq"]
    f_relevance_time = data_female["relevance_time"]
    f_relevance_freq = data_female["relevance_freq"]
    f_start, f_stop = female_sample[0] * 8, female_sample[1] * 8
    if target:
        target_str = "female"
        gt_freq = np.zeros(max_freq)
        gt_freq[165:180] = np.linspace(0, 1, 15)
        gt_freq[180:240] = 1
        gt_freq[240:255] = np.linspace(1, 0, 15)
    else:
        target_str = "male"
        gt_freq = np.zeros(max_freq)
        gt_freq[90:103] = np.linspace(0, 1, 13)
        gt_freq[103:143] = 1
        gt_freq[143:155] = np.linspace(1, 0, 12)
    gts_freq.append(gt_freq)

    # scale signals
    f_signal_time = rescale_array(f_signal_time.squeeze(), new_min=-1, new_max=1)[
        f_start:f_stop
    ]
    signals_time.append(f_signal_time)
    f_signal_freq = rescale_array(
        np.abs(f_signal_freq.squeeze())[:max_freq], new_min=0, new_max=1
    )
    signals_freq.append(f_signal_freq)
    f_relevance_time = f_relevance_time.squeeze()[f_start:f_stop]
    f_relevance_freq = f_relevance_freq.squeeze()[:max_freq]

    # create and scale pos only
    f_relevance_time_pos = np.array(f_relevance_time, copy=True)
    f_relevance_time_pos[f_relevance_time_pos < 0] = 0
    f_relevance_time_pos = rescale_array(f_relevance_time_pos, new_min=0, new_max=1)
    f_relevance_time_pos = np.nan_to_num(f_relevance_time_pos, nan=0)
    relevances_time_pos.append(f_relevance_time_pos)
    f_relevance_freq_pos = np.array(f_relevance_freq, copy=True)
    f_relevance_freq_pos[f_relevance_freq_pos < 0] = 0
    f_relevance_freq_pos = rescale_array(f_relevance_freq_pos, new_min=0, new_max=1)
    f_relevance_freq_pos = np.nan_to_num(f_relevance_freq_pos, nan=0)
    relevances_freq_pos.append(f_relevance_freq_pos)

    # Get male data
    data_male = male_sample[2]
    target = data_male["target"]
    m_signal_time = data_male["signal_time"]
    m_signal_freq = data_male["signal_freq"]
    m_relevance_time = data_male["relevance_time"]
    m_relevance_freq = data_male["relevance_freq"]
    m_start, m_stop = male_sample[0] * 8, male_sample[1] * 8
    if target:
        target_str = "female"
        gt_freq = np.zeros(max_freq)
        gt_freq[165:180] = np.linspace(0, 1, 15)
        gt_freq[180:240] = 1
        gt_freq[240:255] = np.linspace(1, 0, 15)
    else:
        target_str = "male"
        gt_freq = np.zeros(max_freq)
        gt_freq[90:103] = np.linspace(0, 1, 13)
        gt_freq[103:143] = 1
        gt_freq[143:155] = np.linspace(1, 0, 12)
    gts_freq.append(gt_freq)

    # Scale signals
    m_signal_time = rescale_array(m_signal_time.squeeze(), new_min=-1, new_max=1)[
        m_start:m_stop
    ]
    signals_time.append(m_signal_time)
    m_signal_freq = rescale_array(
        np.abs(m_signal_freq.squeeze())[:max_freq], new_min=0, new_max=1
    )
    signals_freq.append(m_signal_freq)
    m_relevance_time = m_relevance_time.squeeze()[m_start:m_stop]
    m_relevance_freq = m_relevance_freq.squeeze()[:max_freq]

    # Create and scale positive only relevances
    m_relevance_time_pos = np.array(m_relevance_time, copy=True)
    m_relevance_time_pos[m_relevance_time_pos < 0] = 0
    m_relevance_time_pos = rescale_array(m_relevance_time_pos, new_min=0, new_max=1)
    m_relevance_time_pos = np.nan_to_num(m_relevance_time_pos, nan=0)
    relevances_time_pos.append(m_relevance_time_pos)

    m_relevance_freq_pos = np.array(m_relevance_freq, copy=True)
    m_relevance_freq_pos[m_relevance_freq_pos < 0] = 0
    m_relevance_freq_pos = rescale_array(m_relevance_freq_pos, new_min=0, new_max=1)
    m_relevance_freq_pos = np.nan_to_num(m_relevance_freq_pos, nan=0)
    relevances_freq_pos.append(m_relevance_freq_pos)

    plot_multi_dft_attr(
        signals_time,
        signals_freq,
        relevances_time_pos,
        relevances_freq_pos,
        gts_freq,
        gt_bar=False,
        linewidth=linewidth,
        threshold=threshold,
        cmap_boost=boost,
        attr_as_bkg=False,
        # title=name,
        height_per_row=1.5,
        font_size=font_size,
        save_to=save_to,
    )
    logger.info("Saved Figure 4 as %s", save_to)
