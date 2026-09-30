# Software Frameworks for Explainable AI in Time Series Classification: A Systematic Review
*Louis Peter, Nils Gumpfer, Jana Fischer, Christin Seifert and Jennifer Hannig*

This repository holds supplementary material to the publication "Software Frameworks for Explainable AI in Time Series Classification: A Systematic Review" by Peter et al. (2026).

A preprint of the paper is available [here](https://doi.org/10.48550/arXiv.2608.21449). If this is useful for your own work, please consider citing our paper:
```bibtex
 @article{Peter2026,
      title={Software Frameworks for Explainable AI in Time Series Classification: A Systematic Review}, 
      author={Louis Peter and Nils Gumpfer and Jana Fischer and Christin Seifert and Jennifer Hannig},
      year={2026},
      eprint={2608.21449},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.21449}, 
      doi = {https://doi.org/10.48550/arXiv.2608.21449}
}
```

The paper has been accepted for presentation and publication in the post-workshop proceedings of the 8th International Workshop on eXplainable Knowledge Discovery in Data Mining (XKDD 2026), co-located with ECML PKDD 2026. The publication information of the accepted version will be updated here, as soon as it is available.

---

## Reproducing Results
**Requirements:**
- Python 3.11
- Dependencies are listed in `pyproject.toml`.

**Commands:**
```bash
pip install .
python run_all.py
```

## Pre-trained model weights
Both pre-trained models are included in the repository:
```
models/gcb/GCB2020_epoch_20.pt             # ECG classifier (12-lead, RBBB vs. healthy)
models/audionet/audionet_gender_digit9.pt  # AudioNet (gender classification, digit 9)
```


## Project structure
```
├── supplementary.pdf                   # supplementary material of the paper
├── run_all.py                          # entry point - runs all experiments
├── pyproject.toml                      # pip dependencies
├── experiments/
│   ├── compare_methods.py              # per-method attribution comparisons
│   ├── compare_metrics.py              # faithfulness metric comparisons
│   └── audiomnist_scripts.py           # AudioMNIST DFT-LRP experiments
├── utils/
│   ├── logging.py                      # logging helpers
│   ├── utils.py                        # general utilities
│   └── viz.py                          # visualisation helpers
├── models/
│   ├── models.py                       # model class definitions (GCB2020, AudioNet)
│   ├── gcb/
│   │   ├── GCB2020_epoch_20.pt         # pre-trained GCB2020 weights
│   │   └── test_data/                  # ECG test split (443 samples, 12-lead, 5000 timesteps)
│   └── audionet/
│       ├── audionet_gender_digit9.pt   # pre-trained AudioNet weights
│       ├── split_9.json                # train/val/test speaker split for digit 9
│       └── samples/
│           ├── sample_female.pt        # pre-computed input tensor, female speaker
│           └── sample_male.pt          # pre-computed input tensor, male speaker
└── output/
    ├── experiments/                    # per-experiment CSVs, logs, and intermediate outputs
    └── figures/                        # components for publication figures
```