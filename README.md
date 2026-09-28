# DSB-FR — private working release

Reproducibility materials for **One-Shot Image-Level Industrial Anomaly Detection via Dual-Depth Subspaces and Full–Region Fusion**, by Yanan Yang and Shuhan Ren.

This is a small private author-review upload prepared on 2026-09-28, not yet a complete public release. The organized source tree is in **DSB-FR-source.zip**. Extract it and read DSB-FR/README.md for settings, dependencies, dataset layout and commands. Source directories can be expanded here in a later update.

## Included

- Paper-only DSB-FR runner, frozen helpers, configuration and reproduction documentation.
- 15,258 recorded query/seed scores and 108 category/seed paired image-AUROC rows.
- Bootstrap intervals, selected rank/fusion controls and analysis scripts.
- Source hashes, provenance and validation details.

## Results and validation

Equal-dataset mean image AUROC: SubspaceAD 0.859074; DSB-FR 0.882728. Gain: **2.37 percentage points**, paired 95% interval **[0.68, 4.27]**. This is an average, not a gain on every dataset. BTAD decreases; pixel AUROC decreases on all four datasets. FR changes the image score, not the pixel map.

The included score data reproduce the main gain, FR gain (1.59 points), rank-control gain (0.517 points), and 100,000-resample confidence interval. The new runner completed hazelnut seed 42 with AUROC 1.0; this is not a full benchmark rerun.

## Quick use after extraction

Install a compatible PyTorch/torchvision pair and then, inside DSB-FR:

    python -m pip install -r requirements.txt
    python tools/analyze_results.py
    python tools/run_dsb_fr.py --dataset /path/to/mvtec_anomaly_detection --categories hazelnut --seeds 42 --outdir outputs/hazelnut

The runner downloads the pinned DINOv2 checkpoint unless --model-path and --offline are supplied. Data images and weights are not included.

## Before public release

Review upstream redistribution permissions and choose a license for author-owned work. The historical driver includes exploratory methods: **dsb_fr** is the paper method, not **mlsb_support_cal_blend_fr**. External baselines and all pixel metrics have not been rerun for this package.

No raw dataset images, weights, signatures, submission files or credentials are included. A private repository is not a public deposit: do not describe it as publicly available until access and release are finalized.
