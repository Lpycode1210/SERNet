# SERNet model structure

The table below transcribes the supplied architecture workbook. Click a module name to jump directly to its implementation in [`src/sernet_model.py`](../src/sernet_model.py).

| Module | Layer setting | Output shape | Parameters | Code |
|---|---|---:|---:|---|
| Input | PPG, VPG, APG waveform, demographic vector (age, gender, height, weight, BMI), and 34 additional PPG-derived features | `[B, 3, 420]`, `[B, 5]`, `[B, 34]` | 0 | — |
| CNN: convolution block | Four `Conv1d(3×3) + BN + ReLU` layers, channels `3→32→32→64→64`, with two `MaxPool(2)` layers | `[B, 64, 105]` | 22,368 | [CNNModule](../src/sernet_model.py#L173-L232) |
| CNN: residual blocks | Two 3-convolution residual blocks; first has `1×1 + BN` shortcut (`64→128`), second identity shortcut; final max pool | `[B, 128, 52]` | 280,320 | [BasicBlock](../src/sernet_model.py#L131-L171), [CNNModule](../src/sernet_model.py#L201-L232) |
| KFM-LSTM: KAN-FiLM | Two KAN mappings (`39→128`, Fourier grid `k=5`), identity-style bias initialization; `X ← s₁·X + s₂` | `[B, 128, 52]` | 110,080 | [KANLinear](../src/sernet_model.py#L42-L72), [KANFiLMLayer](../src/sernet_model.py#L74-L105), [KFMLSTM](../src/sernet_model.py#L294-L342) |
| KFM-LSTM: sequence encoder | Smooth absolute transform `x·tanh(x)`; LSTM bottleneck `128→64→128`; final hidden state | `[B, 128]` | 148,224 | [KFMLSTM](../src/sernet_model.py#L294-L342) |
| XLSTM: temporal features | BiLSTM (`3→64` per direction) and temporal attention (SiLU then softmax over time) | `[B, 420, 128]` at attention | 34,944 | [TemporalAttention](../src/sernet_model.py#L234-L248), [XLSTMModule](../src/sernet_model.py#L344-L393) |
| XLSTM: sequence encoder | LSTM `128→256→128`; final time-step representation | `[B, 128]` | 591,360 | [XLSTMModule](../src/sernet_model.py#L344-L393) |
| Fusion: branch attention | Spatial attention for both branch vectors; LayerNorm; stack branches | `[B, 2, 128]` | 257 | [SpatialAttention](../src/sernet_model.py#L250-L268), [FusionModule](../src/sernet_model.py#L395-L449) |
| Fusion: cross attention | Cross-attention over two branch vectors, then concatenate | `[B, 256]` | 128 | [CrossAttention](../src/sernet_model.py#L270-L292), [FusionModule](../src/sernet_model.py#L395-L449) |
| Fusion: guidance modulation | `Linear(39→256) + BN + ReLU + Dropout`; `tanh(H_f(G)) ⊙ V₄ + V₄` | `[B, 256]` | 10,752 | [FusionModule](../src/sernet_model.py#L395-L449) |
| Absolute evidential head | MLP `256→128`, then evidential layer `128→8` for two targets | `[B, 2, 4]` | 33,928 | [EvidentialHead](../src/sernet_model.py#L11-L40), [SERNet](../src/sernet_model.py#L511-L540) |
| Siamese differential head | `[F₁−F₂, F₁⊙F₂]`; `512→256` + LayerNorm + SiLU; evidential layer `256→8` | `[B, 2, 4]` | 133,896 | [SERNet](../src/sernet_model.py#L541-L604) |
| Confidence gating Output| Evidential uncertainty and confidence gate combine absolute and reference-adjusted differential estimates | `[B, 2]` | 0 | [SERNet](../src/sernet_model.py#L606-L626) |
| **Total** |  |  | **1,366,257 (≈1.36M)** |  |

`B` is batch size. In the table, `[B, 420, 128]` describes the attended temporal sequence before the XLSTM module selects its final time step.
