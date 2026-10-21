# SERNet

PyTorch implementation of the Siamese Evidential Regression Network (SERNet) for cuffless blood pressure estimation and uncertainty quantification.

## Repository contents

- [`src/sernet_model.py`](src/sernet_model.py) — core network modules and `SERNet` model.
- [`docs/model_structure.md`](docs/model_structure.md) — architecture and parameter table. Each module links to its implementation below.
- [`experiments/`](experiments/) — supplementary experiment planning and a results template. Numerical experiment results can be added when the corresponding logs or tables are available.

The manuscript PDF and source workbook are not included in this repository.

## Model overview

SERNet utilizes a feature-modulated spatio-temporal encoder (FMSTE) to extract highly representative features of the PPG signals and their derivatives. By introducing an evidential regression head to parameterize the higher-order probability distribution of the predicted BP, it replaces traditional point estimation, thereby achieving uncertainty quantification. Furthermore, we designed a Siamese path with a shared encoder to predict the relative changes in BP values through the feature differences between the current input and a reference input, eliminating the impact of intra-subject variability. Finally, an output BP value with higher confidence is obtained through a confidence-driven gating mechanism.

See the [linked architecture table](docs/model_structure.md) for module details and direct code links.

## Inputs

`SERNet.forward` expects:

- `x`: current signal tensor `[batch, 3, length]` (PPG, VPG, APG channels).
- `demos`: demographic features `[batch, 5]`.
- `extra_feats`: additional features `[batch, 34]`.

For the Siamese path, pass `x_ref`, `demos_ref`, `extra_feats_ref`, and `label_ref` as well. `label_ref` is the reference blood pressure vector `[batch, 2]`. Without `x_ref`, the model returns the absolute prediction and evidential parameters.

## Requirements

This repository contains the model definition only.

## Notes

- 
- 
- 
