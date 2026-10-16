# SERNet

PyTorch implementation of the Siamese Evidential Regression Network (SERNet) for cuffless blood pressure estimation and uncertainty quantification.

## Repository contents

- [`src/sernet_model.py`](src/sernet_model.py) — core network modules and `SERNet` model.
- [`docs/model_structure.md`](docs/model_structure.md) — architecture and parameter table. Each module links to its implementation below.
- [`experiments/`](experiments/) — supplementary experiment planning and a results template. Numerical experiment results can be added when the corresponding logs or tables are available.

The manuscript PDF and source workbook are not included in this repository.

## Model overview

SERNet uses a shared Feature-Modulated Spatio-Temporal Encoder (FMSTE), an absolute evidential regression head, and a Siamese differential evidential head. When reference samples and their labels are supplied, a confidence-driven gate combines absolute and differential predictions.

See the [linked architecture table](docs/model_structure.md) for module details and direct code links.

## Inputs

`SERNet.forward` expects:

- `x`: current signal tensor `[batch, 3, length]` (PPG, VPG, APG channels).
- `demos`: demographic features `[batch, 5]`.
- `extra_feats`: additional features `[batch, 34]`.

For the Siamese path, pass `x_ref`, `demos_ref`, `extra_feats_ref`, and `label_ref` as well. `label_ref` is the reference blood pressure vector `[batch, 2]`. Without `x_ref`, the model returns the absolute prediction and evidential parameters.

## Requirements

```bash
python -m pip install -r requirements.txt
```

This repository contains the model definition only. Dataset loading, preprocessing, training configuration, and evaluation scripts are not included in the supplied materials.

## Notes

- The model expects signals of a length compatible with its pooling layers and the dimensions documented in the source workbook (the example architecture uses length 420).
- No dataset, trained weights, or numerical supplementary experiment results are published here.
- The model file keeps the complete `SERNet(nn.Module)` implementation as the exported `SERNet` class.
