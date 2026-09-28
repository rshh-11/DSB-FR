# Reproducibility details

Paper settings are centralized in configs/paper.json. The Hugging Face encoder is facebook/dinov2-with-registers-large, revision e4c89a4e05589de9b3e188688a303d0f3c04d0f3. This is the actual cached snapshot also recorded in the manuscript, not a floating latest version.

Observed inference environment: Python 3.10; torch 2.5.1+cu118; torchvision 0.20.1+cu118; transformers 5.8.1; numpy 1.26.4; opencv-python 4.8.1.78; Pillow 12.2.0; scikit-learn 1.7.2; tqdm 4.67.3. These are observed installed versions, not a claim that every combination of OS/driver has been tested or that this is a complete historical lockfile for every experiment.

Features and residual reconstruction use float32. PCA covariance/eigendecomposition uses float64. Whitening and additional group-level normalization are disabled. Augmentations use torchvision RandomRotation(degrees=(0,345)) defaults, including interpolation/fill conventions. The PCA helper selects CUDA when available; the new launcher explicitly forces PCA to CPU when --device cpu is requested.

The Otsu mask uses 5×5 Gaussian-blurred grayscale input. Bright and dark masks are compared by border occupancy (including duplicated corner pixels); ties choose bright. Area below 2% or above 98% gives a full mask. Nearest-neighbor mapping to 18×18 repeats the area check; the mask is enlarged for scoring. Empty selections fall back to full-image pooling. Top 1% uses max(1,floor(0.01*N)) pixels, so region/full pooling need not average the same number of pixels.

The new paper-only launcher reuses recorded extraction/PCA/calibration/scoring functions and omits exploratory mixture computations. Its validation is separate from the historical results. Deterministic seeds do not guarantee cross-platform bitwise equality. Full reruns of 108 category/seed pairs and all external baselines have not been performed as part of repository packaging.

Pixel evaluation in the paper used all test images for raw-map AUROC, but anomalous images only, per-image min–max normalization and FPR<=0.3 for AUPRO. The minimal runner saves maps on request but does not implement a replacement AUPRO metric. The historical driver retains that implementation and requires its compatible anomalib development environment.

No training data augmentation is an additional independently acquired normal reference. No defect annotations are used in fitting, calibration or region inference. Development feedback and post-hoc analyses must remain distinguished from frozen validation.
