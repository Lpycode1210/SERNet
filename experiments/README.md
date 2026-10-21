# Supplementary experiments

This folder is reserved for supplementary experiments associated with SERNet.

# Robustness to Data Partitioning: Ten Random Subject-wise Splits (MIMIC-III)

Ten independent random subject-wise splits were performed on the MIMIC-III dataset. In each split, subjects were divided into training, validation, and test sets at a ratio of 6:2:2 with strictly disjoint subjects across the three sets (no data leakage), and the model was trained and evaluated independently, in order to verify that the reported performance does not depend on a particular partition.

![Ten-split results](ten_split_results.png)

In (a), the polylines show the per-split MAE (red: SBP; blue: DBP); the horizontal dashed lines denote the mean over the ten splits; the shaded bands denote mean ± 1 SD; and the black dotted lines indicate the single-split results reported in Table IV (SBP 5.16 / DBP 4.58 mmHg). In (b), the box spans the interquartile range (IQR), the inner black line is the median, the diamond is the mean, and the dots are the ten individual split results. The per-split MAEs fluctuate randomly around their means, indicating that the performance of SERNet is insensitive to data partitioning and is stably reproducible. results, indicating that the performance of SERNet is insensitive to data partitioning and is stably reproducible.

# Component-wise Ablation Study

This document reports the incremental ablation (Cases 1–6, as in Table VI of the paper) together with additional component-isolation variants (marked †) that separately verify the Siamese structure, the evidential regression heads, and the uncertainty-driven confidence gating. All variants use the same subject-wise partitions (6:2:2, disjoint subjects) and training protocol as the main model.

## PPG-BP dataset

| Variant | SBP MAE | SBP ME±SDE | DBP MAE | DBP ME±SDE |
|---|:---:|:---:|:---:|:---:|
| Case 1: CNN-LSTM | 6.76 | -2.25±9.22 | 5.47 | -1.76±7.85 |
| Case 2: CNN-LSTM+XLSTM | 6.10 | 0.43±8.60 | 5.11 | -1.09±7.13 |
| Case 3: CNN-LSTM+XLSTM+Fusion | 5.83 | -0.36±8.11 | 4.84 | -0.85±6.91 |
| Case 4: CNN+KFM-LSTM+XLSTM+Fusion | 5.41 | 0.31±8.06 | 4.35 | 0.57±6.24 |
| Case 4+Siamese (det., direct sum) | 5.21 | 0.74±7.59 | 4.18 | -0.62±5.53 |
| Case 5: Case 4+Evidential (abs. only) | 5.22 | 0.83±7.66 | 4.18 | -0.41±5.69 |
| Case 4+Siamese+Evidential (Refer=0) | 5.30 | 0.75±6.92 | 4.06 | 0.84±5.50 |
| † Siamese w/o evidential (det. heads, g=0.5) | 5.19 | 0.66±7.45 | 4.15 | -0.60±5.56 |
| † Fixed-weight fusion (g=0.5) | 4.86 | 0.48±7.12 | 3.94 | 0.38±5.47 |
| † Learned static gate | 4.68 | -0.32±6.98 | 3.84 | 0.44±5.41 |
| Case 6: SERNet (full) | 4.45 | -0.29±6.81 | 3.71 | 0.41±5.38 |

## MIMIC-III dataset

| Variant | SBP MAE | SBP ME±SDE | DBP MAE | DBP ME±SDE |
|---|:---:|:---:|:---:|:---:|
| Case 1: CNN-LSTM | 8.32 | -2.16±9.78 | 6.75 | 1.21±8.53 |
| Case 2: CNN-LSTM+XLSTM | 7.46 | -1.68±9.12 | 5.94 | -0.91±8.69 |
| Case 3: CNN-LSTM+XLSTM+Fusion | 6.45 | 0.96±9.30 | 5.53 | 0.92±8.22 |
| Case 4: CNN+KFM-LSTM+XLSTM+Fusion | 5.78 | -0.45±8.41 | 5.06 | -0.80±7.87 |
| Case 4+Siamese (det., direct sum) | 5.50 | 0.62±8.15 | 4.77 | 0.65±7.43 |
| Case 5: Case 4+Evidential (abs. only) | 5.56 | 0.38±8.13 | 4.82 | 1.03±6.82 |
| Case 4+Siamese+Evidential (Refer=0) | 5.42 | -0.79±8.06 | 4.75 | 1.16±6.55 |
| † Siamese w/o evidential (det. heads, g=0.5) | 5.47 | 0.55±8.05 | 4.74 | 0.68±7.33 |
| † Fixed-weight fusion (g=0.5) | 5.33 | -0.46±7.83 | 4.68 | 0.82±6.51 |
| † Learned static gate | 5.24 | 0.41±7.76 | 4.63 | -0.66±6.46 |
| Case 6: SERNet (full) | 5.16 | -0.49±7.92 | 4.58 | 0.62±6.40 |

> † Component-isolation variants added in response to reviewer suggestions; values are preliminary and will be updated after re-verification. Abbreviations: det. = deterministic point-estimation heads (no uncertainty); g = fusion gate weight; Refer=0 = reference BP set to zero in the gating formula.
