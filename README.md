# DSB-FR

Reproducibility materials for **One-Shot Image-Level Industrial Anomaly Detection via Dual-Depth Subspaces and Full–Region Fusion**, by Yanan Yang and Shuhan Ren.

DSB-FR fits two PCA subspaces to features from one normal reference image and combines full-image and intensity-derived region scores. This is an initial public release of the code, configuration files, and recorded image-level results. It does not assert journal acceptance or provide a paper DOI.

## What is included

- A paper-only inference/evaluation entry point: tools/run_dsb_fr.py.
- The recorded numerical primitives and historical research driver, with provenance hashes.
- Frozen settings in configs/paper.json, environment notes, and dataset-layout instructions.
- Per-query DSB scores, 108 category/seed paired AUROC results, recorded bootstrap intervals, and structural-control summaries.
- A script to recompute the main image-level gains directly from the included scores.

Dataset images, model weights, submission PDFs, author signatures and private correspondence are not included. Obtain third-party assets separately under their original terms.

## Main findings

| Dataset | SubspaceAD image AUROC | DSB-FR image AUROC |
|---|---:|---:|
| MVTec-AD | 0.919422 | 0.959384 |
| BTAD | 0.953867 | 0.945291 |
| VisA | 0.900500 | 0.906501 |
| MPDD | 0.662506 | 0.719736 |
| Equal-dataset mean | 0.859074 | 0.882728 |

The equal-dataset gain is **2.37 percentage points**, with a paired hierarchical 95% interval of **[0.68, 4.27]**. It is not an improvement on every dataset: BTAD decreases. Fixed-map full–region scoring contributes 1.59 points over full-image scoring; the rank-matched joint-PCA comparison gives 0.517 points over the three development datasets. Pixel AUROC decreases on all four datasets. Selected example heatmaps do not establish average localization gains.

## Installation

Use Python 3.10 and an isolated environment. The observed GPU environment used torch 2.5.1+cu118 and torchvision 0.20.1+cu118. Install the compatible PyTorch pair for your platform first, then run:

    python -m pip install -r requirements.txt

The minimal runner does not need anomalib. The unmodified historical support_mixture_subspace_pilot.py does import anomalib, including a private AUPRO API; it used local anomalib 2.4.3.dev0 and is provided for provenance, not as the recommended installation entry point. See docs/REPRODUCIBILITY.md for environment and numerical limitations.

## Verify the released results without GPU or datasets

    python tools/analyze_results.py
    python tools/analyze_results.py --bootstrap 100000

These commands recompute image AUROC/AP from results/image_scores.csv, check all category/seed AUROCs against the frozen paired table, and calculate the equal-dataset and rank-matched effects. The second command additionally resamples paired categories and seeds using seed 20260902. Outputs go to outputs/ and are not tracked.

## Run DSB-FR

Download and arrange the dataset as described in docs/DATASETS.md. For example:

    python tools/run_dsb_fr.py --dataset /path/to/mvtec_anomaly_detection --categories hazelnut --seeds 42 123 999 --outdir outputs/hazelnut

The first run obtains the specific DINOv2 checkpoint revision recorded in configs/paper.json. For a pre-downloaded snapshot, supply --model-path /path/to/snapshot --offline. For a small smoke test add --debug-limit 2; never report that subset as benchmark performance. Add --save-maps to save full pixel maps for inspection. The entry point evaluates image AUROC and average precision; it does not silently substitute a different pixel-AUPRO protocol.

For each category, paths in train/good are sorted and shuffled with a dedicated Python RNG seeded by the support seed. Only the first real normal image is used. Thirty rotation views are added, except for transistor (original alone). The user-selected normal hazelnut 000 image used in the graphical abstract is an illustrative case, not a replacement for benchmark support sampling.

## Method

1. Freeze DINOv2-L with registers; resize RGB inputs to 256×256 without center crop.
2. Average the last six hidden states for group A and the previous six for group B; discard CLS/register tokens.
3. Fit a centered PCA at 95% explained variance independently per group, without whitening.
4. Compute squared orthogonal residuals, apply log1p, support median/MAD calibration with the documented fallback, and clip negative calibrated values to zero.
5. Average the two maps, bilinearly resize the 18×18 grid to 256×256, and smooth with a 3×3 Gaussian kernel, sigma 4.
6. Derive a query region using grayscale Otsu thresholding, border-based polarity selection and area fallbacks. Compute the mean of the largest 1% responses over the full image and region; average the two scores.

FR changes the image score, not the anomaly map. Query labels and ground-truth masks are used only for evaluation.

## Scope and release notes

MVTec-AD, BTAD and VisA informed development. MPDD was evaluated after configuration freezing. The equal-dataset bootstrap excludes configuration-selection uncertainty. External baselines used different encoders/preprocessing, so their comparison is not backbone-controlled.

Read docs/VALIDATION.md for actual tests, docs/RESULTS.md for schemas and provenance, and docs/THIRD_PARTY_NOTICES.md for third-party attribution and licensing notes. The historical driver contains many exploratory MLSB/router variants; only the dsb_fr column corresponds to the paper's DSB-FR. Do not replace it with mlsb_support_cal_blend_fr.

The initial public release is available at https://github.com/rshh-11/DSB-FR.
