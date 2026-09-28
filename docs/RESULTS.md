# Included result data

- image_scores.csv: dataset, seed, category, relative sample, binary label, dsb_fusion_full, dsb_fusion_roi and dsb_fr. Scores are raw recorded values, not display-normalized heatmaps.
- paired_image_auroc.csv: 108 matched dataset/category/seed rows for SubspaceAD and DSB-FR. Legacy baseline values are retained at their saved precision.
- paired_bootstrap.csv: recorded paired hierarchical bootstrap with 100,000 resamples, seed 20260902; intervals are in AUROC units, multiply by 100 for percentage points.
- rank_matched_pairs.csv / rank_matched_summary.csv: the three-development-dataset rank-control evidence, not a four-dataset experiment.
- fr_weight_sensitivity.csv: post-hoc regional-score weight sweep; it did not replace the frozen equal weights.

source_manifest.json records the copied files and original hashes. score_provenance.json records the exact per-query sources and selected columns. Only final DSB columns were exported from the historical multi-method logs. In particular, the early SC-MLSB bootstrap is different and is deliberately not used as the paper's +2.37-point evidence.

Aggregation first averages each category/seed's metric within its dataset, then gives four datasets equal weight. Do not pool every image from every dataset into a single AUROC or weight the four dataset means by image count.

The bundled data supports the main image-level comparison and selected controls. It is not a complete archive of every external-method prediction, every ablation, or full-resolution pixel map. Large pixel maps, model weights and dataset images are excluded.
