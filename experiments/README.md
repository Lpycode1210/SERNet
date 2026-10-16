# Supplementary experiments

This folder is reserved for supplementary experiments associated with SERNet.

The files supplied for this repository contained the model definition and an architecture/parameter table, but no supplementary experiment result tables, logs, dataset split information, or training configuration. Therefore, no numerical results are asserted here. Use [`RESULTS_TEMPLATE.md`](RESULTS_TEMPLATE.md) to record results once the source experiment outputs are available.

Suggested supplementary analyses, if they were performed in the study:

1. Component ablation for CNN, KAN-FiLM, XLSTM, fusion, and confidence gating.
2. Absolute-only versus Siamese differential prediction.
3. Uncertainty quality and calibration for SBP and DBP.
4. Robustness across subjects, signal lengths, or noise conditions.

For each experiment, include the data split, preprocessing, training configuration, metrics, number of runs, and uncertainty/error summaries. Do not add sensitive patient-level records.
