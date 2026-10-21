# Supplementary experiments

This folder is reserved for supplementary experiments associated with SERNet.

# Robustness to Data Partitioning: Ten Random Subject-wise Splits (MIMIC-III)

Ten independent random subject-wise splits were performed on the MIMIC-III dataset. In each split, subjects were divided into training, validation, and test sets at a ratio of 6:2:2 with strictly disjoint subjects across the three sets (no data leakage), and the model was trained and evaluated independently, in order to verify that the reported performance does not depend on a particular partition.

![Ten-split results](ten_split_results.png)

In (a), the polylines show the per-split MAE (red: SBP; blue: DBP); the horizontal dashed lines denote the mean over the ten splits; the shaded bands denote mean ± 1 SD; and the black dotted lines indicate the single-split results reported in Table IV (SBP 5.16 / DBP 4.58 mmHg). In (b), the box spans the interquartile range (IQR), the inner black line is the median, the diamond is the mean, and the dots are the ten individual split results. The per-split MAEs fluctuate randomly around their means, indicating that the performance of SERNet is insensitive to data partitioning and is stably reproducible. results, indicating that the performance of SERNet is insensitive to data partitioning and is stably reproducible.
