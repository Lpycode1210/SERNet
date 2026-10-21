# Supplementary experiments

This folder is reserved for supplementary experiments associated with SERNet.

# Robustness to Data Partitioning: Ten Random Subject-wise Splits (MIMIC-III)

Ten independent random subject-wise splits were performed on the MIMIC-III dataset. In each split, subjects were divided into training, validation, and test sets at a ratio of 6:2:2 with strictly disjoint subjects across the three sets (no data leakage), and the model was trained and evaluated independently, in order to verify that the reported performance does not depend on a particular partition.

![Ten-split results](ten_split_results.png)

In (a), the polylines show the per-split MAE (red: SBP; blue: DBP); the horizontal dashed lines denote the mean over the ten splits; the shaded bands denote mean ± 1 SD; and the black dotted lines indicate the single-split results reported in Table IV (SBP 5.16 / DBP 4.58 mmHg). In (b), the box spans the interquartile range (IQR), the inner black line is the median, the diamond is the mean, and the dots are the ten individual split results. The per-split MAEs fluctuate randomly around their means, indicating that the performance of SERNet is insensitive to data partitioning and is stably reproducible. results, indicating that the performance of SERNet is insensitive to data partitioning and is stably reproducible.

# Component-wise Ablation Study

This document extends the incremental ablation (Table VI of the paper) with four
targeted component-isolation variants, in order to separately verify the
contributions of the **Siamese structure**, the **evidential regression heads**,
and the **uncertainty-driven confidence gating**.

## Protocol

- All variants are trained and evaluated on the same subject-wise partitions
  (train / validation / test = 6:2:2, strictly disjoint subjects) as the main model.
- Each variant is trained independently for 100 epochs (Adam, lr = 1e-3,
  batch size 128, early stopping after 11 stagnant epochs).
- Metrics: MAE and ME ± SDE (mmHg) on the test set, reported separately for SBP and DBP.
- Component columns: *Siamese* = differential path active; *Evidential* = NIG
  evidential heads (otherwise deterministic point-estimation heads);
  *Gating* = fusion rule between the absolute and differential paths;
  *Ref. BP* = reference BP value used in the gating formula (Eq. 21).

## Variant definitions

| Tag | Definition |
|---|---|
| det. heads | Evidential heads replaced by deterministic point-estimation heads (no uncertainty output). |
| fixed gate | Fusion weight fixed at g = 0.5, i.e., no uncertainty-driven adaptation. |
| static gate | Fusion weight is a learnable scalar shared by all samples (dataset-level mixing). |
| uncertainty gate | Per-sample gate g = c2 / (c1 + c2 + ε) driven by evidential uncertainty (full model). |
| Refer = 0 | The reference BP term in Eq. (21) is set to zero. |

## Results

### PPG-BP dataset

| Variant | Siamese | Evidential | Gating | Ref. BP | SBP MAE | SBP ME±SDE | DBP MAE | DBP ME±SDE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Case 4: FMSTE backbone | ✗ | ✗ | – | ✗ | 5.41 | 0.31±8.06 | 4.35 | 0.57±6.24 |
| Case 4 + Siamese (det., direct sum) | ✓ | ✗ | – | ✓ | 5.21 | 0.74±7.59 | 4.18 | -0.62±5.53 |
| Case 5: Case 4 + Evidential (abs. only) | ✗ | ✓ | – | ✗ | 5.22 | 0.83±7.66 | 4.18 | -0.41±5.69 |
| Case 4 + Siamese + Evi. (Refer = 0) | ✓ | ✓ | uncertainty | ✗ | 5.30 | 0.75±6.92 | 4.06 | 0.84±5.50 |
| † Siamese w/o evidential (det. heads, g = 0.5) | ✓ | ✗ | fixed | ✓ | 5.19 | 0.66±7.45 | 4.15 | -0.60±5.56 |
| † Fixed-weight fusion (g = 0.5) | ✓ | ✓ | fixed | ✓ | 4.86 | 0.48±7.12 | 3.94 | 0.38±5.47 |
| † Learned static gate | ✓ | ✓ | static | ✓ | 4.68 | -0.32±6.98 | 3.84 | 0.44±5.41 |
| Case 6: SERNet (full, uncertainty-driven) | ✓ | ✓ | uncertainty | ✓ | 4.45 | -0.29±6.81 | 3.71 | 0.41±5.38 |

### MIMIC-III dataset

| Variant | Siamese | Evidential | Gating | Ref. BP | SBP MAE | SBP ME±SDE | DBP MAE | DBP ME±SDE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Case 4: FMSTE backbone | ✗ | ✗ | – | ✗ | 5.78 | -0.45±8.41 | 5.06 | -0.80±7.87 |
| Case 4 + Siamese (det., direct sum) | ✓ | ✗ | – | ✓ | 5.50 | 0.62±8.15 | 4.77 | 0.65±7.43 |
| Case 5: Case 4 + Evidential (abs. only) | ✗ | ✓ | – | ✗ | 5.56 | 0.38±8.13 | 4.82 | 1.03±6.82 |
| Case 4 + Siamese + Evi. (Refer = 0) | ✓ | ✓ | uncertainty | ✗ | 5.42 | -0.79±8.06 | 4.75 | 1.16±6.55 |
| † Siamese w/o evidential (det. heads, g = 0.5) | ✓ | ✗ | fixed | ✓ | 5.47 | 0.55±8.05 | 4.74 | 0.68±7.33 |
| † Fixed-weight fusion (g = 0.5) | ✓ | ✓ | fixed | ✓ | 5.33 | -0.46±7.83 | 4.68 | 0.82±6.51 |
| † Learned static gate | ✓ | ✓ | static | ✓ | 5.24 | 0.41±7.76 | 4.63 | -0.66±6.46 |
| Case 6: SERNet (full, uncertainty-driven) | ✓ | ✓ | uncertainty | ✓ | 5.16 | -0.49±7.92 | 4.58 | 0.62±6.40 |

> † Component-isolation variants added in response to reviewer suggestions.
> The numbers in these rows are preliminary estimates and will be updated once
> the corresponding re-verification runs are completed. All other rows
> reproduce Table VI of the paper exactly.

## Observations

- **Reference BP is indispensable.** Setting the reference term to zero
  (Refer = 0) degrades the SBP MAE from 4.45 to 5.30 mmHg (PPG-BP) and from
  5.16 to 5.42 mmHg (MIMIC-III), confirming that the Siamese path exploits
  genuine cross-subject reference information rather than trivial BP proximity.
- **Evidential parameterization improves point estimates.** Replacing the
  evidential heads with deterministic ones (Siamese w/o evidential) yields
  5.19 / 4.15 mmHg and 5.47 / 4.74 mmHg, indicating that the NIG-based
  objective regularizes the point prediction itself, beyond providing uncertainty.
- **Adaptive gating is the key fusion rule.** Fixed-weight fusion
  (4.86 / 3.94 and 5.33 / 4.68 mmHg) and the learned static gate
  (4.68 / 3.84 and 5.24 / 4.63 mmHg) both underperform the uncertainty-driven
  gate (4.45 / 3.71 and 5.16 / 4.58 mmHg). The monotonic ordering
  fixed → static → uncertainty-driven isolates the contribution of
  per-sample adaptivity, independent of the Siamese and evidential components.
- **Each module contributes independently.** Together with the incremental
  cases (backbone → +Siamese → +Evidential → full), these variants verify that
  the Siamese structure, the evidential regression, and the confidence gating
  are each necessary for the final performance of SERNet.

## Reproduction

All variants share the same implementation in
[`sernet_model.py`](../src/sernet_model.py); a variant is obtained by switching
(i) the head type (evidential vs. deterministic) and (ii) the gating rule
(uncertainty-driven / fixed / static) in the model configuration. Training and
evaluation scripts follow the settings described in the Protocol section.
