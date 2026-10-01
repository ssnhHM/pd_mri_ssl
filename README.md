# Self-supervised structural MRI for predicting putaminal dopamine transporter binding in Parkinson's disease.

## Abstract

Whether self-supervised structural MRI representations can be leveraged to predict variation in putaminal dopamine transporter (DAT) binding in Parkinson’s disease (PD) remains uncertain. Using 3.0 T MRI from 343 participants in the Parkinson’s Progression Markers Initiative, encoders with four single-channel and five multichannel configurations were pre-trained. By means of nested cross-validation, frozen representations predicted the lowest putamen specific binding ratio (SBR) derived from dopamine transporter single-photon emission computed tomography (DAT-SPECT) in 126 participants with PD. The E5 configuration, which combined T2-weighted (T2w) MRI with the T1-weighted/T2-weighted ratio (T1w/T2w ratio), yielded the highest mean MRI-only $R^2$ [0.228; Spearman correlation, $\rho=0.389$; mean absolute error (MAE) = 0.168; root mean squared error (RMSE) = 0.235]. The corresponding atlas-based regional-mean model achieved an $R^2$ of 0.031; the difference was not significant after multiple comparisons correction ($q$=0.058). Two configurations provided predictive information beyond the covariates of age, sex and scanner after correction (both $q$=0.043). The E5 configuration retained an $R^2$ of 0.188 following SBR residualization, and T2w channel shuffling induced the largest mean $R^2$ loss. These findings support independent validation of combined T2w and T1w/T2w ratio representations for DAT-binding prediction in PD.

## Model and inputs

The four-block 3D CNN produces 256-dimensional embeddings. Training uses
VICReg ([Bardes et al., 2022](https://arxiv.org/abs/2105.04906)), AdamW and
200 epochs; full settings are in `configs/`. The best
checkpoint is selected by SSL validation loss from epoch 50 onward.

Only the common 3.0 T T1w/T2w cohort is supported, including single-channel
configurations. Missing or non-3T acquisition metadata are excluded; a
channel-specific cohort mismatch is reported rather than silently intersected.

| Input | Path column |
|---|---|
| T1w | `t1_crop_path` |
| T2w | `t2_crop_path` |
| T1 log-Jacobian (T1J) | `t1j_crop_path` |
| T1w/T2w ratio | `ratio_crop_path` |

T1J uses the Camacho-derived log-Jacobian map
([Camacho et al., 2024](https://doi.org/10.1038/s41531-024-00647-9)); ratio uses
the Du2019-derived ratio map ([Du et al., 2019](https://doi.org/10.1002/ana.25376)).

| Configuration | Channels, in model input order |
|---|---|
| E1 | T1J, ratio, T2w |
| E2 | T1J, ratio, T2w, T1w |
| E3 | T1w, T2w, ratio |
| E4 | T1J, ratio |
| E5 | T2w, ratio |
| E6 | T1J |
| E7 | ratio |
| E8 | T1w |
| E9 | T2w |

## Prerequisites

- Miniconda or Anaconda.
- A CUDA-capable NVIDIA GPU for training with the supplied environment.
- Preprocessed MRI volumes and a private participant-level split manifest.

## Step 1: Create the environment

Run these commands from the repository directory:

```bash
conda env create -f environment.yml
conda activate pd-mri-ssl
```

## Step 2: Run the analysis

```bash
MANIFEST=/path/to/publication_manifest.csv \
DEVICE=cuda:0 bash run_example.sh
```

The script runs E1-E9 across seeds 3407-3411 and folds 1-5, then extracts
`checkpoint_best` embeddings and fits frozen probes. Add
`CONFIG=configs/e8_t1_only.yml` to run one configuration, or set
`PHASE=train`, `extract` or `nested` to run one stage. Complete training runs
are reused unless `FORCE=1`. Set `OUT=results/my_3t_run` to keep all stages
in a separate directory. `TRAIN_ROOT`, `EMBEDDING_ROOT` and `PROBING_ROOT`
can still override individual stage paths.
`PHASE=all` runs all stages, not all field strengths. Nested-probe defaults
(five inner folds, the Ridge grid and random seed) are defined in
`scripts/frozen_probe.py`; call that script directly to change them.

## References

1. Camacho, M. et al. (2024). [Exploiting macro- and micro-structural brain changes for improved Parkinson's disease classification from MRI data](https://doi.org/10.1038/s41531-024-00647-9).
2. Du, G. et al. (2019). [Magnetic resonance T1w/T2w ratio: A parsimonious marker for Parkinson disease](https://doi.org/10.1002/ana.25376).
3. Bardes, A., Ponce, J. & LeCun, Y. (2022). [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906). ICLR 2022.
