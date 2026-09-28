"""Support-conditioned mixture of local subspace banks pilot.

The script tests whether support patch tokens should be modeled by several
local PCA banks instead of one category-level PCA bank.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from anomalib.metrics.aupro import _AUPRO as TM_AUPRO
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from multilayer_subspace_pilot import (
    DEFAULT_BLOCKS,
    BlockSpec,
    MetricRow,
    MultiLayerExtractor,
    fit_pca,
    image_foreground_mask,
    image_score,
    image_score_masked,
    parse_blocks,
    pca_energy,
    post_process_map,
    reference_images,
    robust_log_z,
    set_seed,
    test_paths,
    train_paths,
    write_outputs,
)


DEFAULT_CATEGORIES = ["cable", "capsule", "hazelnut", "screw", "toothbrush"]


@dataclass
class MixtureBank:
    centers: np.ndarray
    pca_params: list[dict]
    component_means: np.ndarray
    component_train_energy: list[np.ndarray]
    component_reliability: np.ndarray
    global_pca: dict
    train_global_energy: np.ndarray
    train_hard_energy: np.ndarray
    train_soft_energy: np.ndarray
    train_anchor_energy: np.ndarray
    train_proto_center_energy: np.ndarray
    temperature: float
    router_temperature: float
    component_sizes: list[int]
    pseudo_router: dict[str, np.ndarray] | None


@dataclass
class SupportValGate:
    trust: float
    n_fit: int
    n_val: int
    dsb_median: float
    hard_median: float
    dsb_q90: float
    hard_q90: float
    median_ratio: float
    q90_ratio: float
    reason: str


@dataclass
class SupportSelfValSelector:
    selected_method: str
    reason: str
    n_fit: int
    n_val: int
    method_aurocs: dict[str, float]
    method_margins: dict[str, float]


@dataclass
class FeatureAdapterState:
    mean: np.ndarray
    std: np.ndarray
    down_weight: np.ndarray
    up_weight: np.ndarray
    strength: float
    rank: int
    normal_energy_before: float
    normal_energy_after: float
    pseudo_energy_before: float
    pseudo_energy_after: float
    delta_rms: float


def ground_truth_mask(
    dataset_root: Path,
    category: str,
    test_path: Path,
    image_res: int,
) -> np.ndarray:
    """Load masks from MVTec/BTAD-style or flattened VisA-style layouts."""
    if test_path.parent.name.lower() in {"good", "ok", "normal"}:
        return np.zeros((image_res, image_res), dtype=np.uint8)
    gt_root = dataset_root / category / "ground_truth"
    candidates = [
        gt_root / test_path.parent.name / f"{test_path.stem}_mask.png",
        gt_root / test_path.parent.name / f"{test_path.stem}.png",
        gt_root / f"{test_path.stem}_mask.png",
        gt_root / f"{test_path.stem}.png",
    ]
    mask_path = next((path for path in candidates if path.exists()), None)
    if mask_path is None:
        raise FileNotFoundError(f"No ground-truth mask found for {test_path}")
    mask = Image.open(mask_path).convert("L").resize(
        (image_res, image_res), Image.Resampling.NEAREST
    )
    return (np.asarray(mask) > 0).astype(np.uint8)


def normalized_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def pixel_metric_row(
    seed: int,
    category: str,
    method: str,
    gt_maps: list[np.ndarray],
    pred_maps: list[np.ndarray],
    device: str,
) -> dict[str, object]:
    gt_flat = np.concatenate([mask.reshape(-1) for mask in gt_maps]).astype(np.uint8)
    pred_flat = np.concatenate([pred.reshape(-1) for pred in pred_maps]).astype(np.float32)
    pixel_auroc = float(roc_auc_score(gt_flat, pred_flat))
    anomalous = [idx for idx, mask in enumerate(gt_maps) if np.any(mask)]
    au_pro = float("nan")
    if anomalous:
        preds = torch.from_numpy(
            np.stack([normalized_map(pred_maps[idx]) for idx in anomalous])
        )
        targets = torch.from_numpy(np.stack([gt_maps[idx] for idx in anomalous]))
        metric_device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        metric = TM_AUPRO(fpr_limit=0.3).to(metric_device)
        au_pro = float(metric(preds.to(metric_device), targets.to(metric_device)).cpu().item())
    return {
        "seed": seed,
        "category": category,
        "method": method,
        "pixel_auroc": pixel_auroc,
        "au_pro": au_pro,
        "n_images": len(gt_maps),
        "n_anomalous": len(anomalous),
    }


class LowRankResidualAdapter(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        rank: int,
        strength: float,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, -1))
        self.register_buffer("std", std.reshape(1, -1))
        self.strength = float(strength)
        self.down = torch.nn.Linear(feature_dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, feature_dim, bias=False)
        torch.nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / torch.clamp(self.std, min=1e-6)
        delta = self.up(torch.tanh(self.down(z)))
        return x + self.strength * delta


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def grid_coordinates(grid_size: tuple[int, int], n_images: int) -> np.ndarray:
    h, w = grid_size
    ys, xs = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([ys.reshape(-1), xs.reshape(-1)], axis=1)
    return np.tile(coords, (n_images, 1))


def assignment_features(
    features: np.ndarray,
    coords: np.ndarray,
    coord_weight: float,
) -> np.ndarray:
    feat = l2_normalize(features.astype(np.float32, copy=False))
    if coord_weight <= 0:
        return feat.astype(np.float32)
    return np.concatenate(
        [feat, float(coord_weight) * coords.astype(np.float32, copy=False)],
        axis=1,
    ).astype(np.float32)


def collect_spatial_features(
    extractor: MultiLayerExtractor,
    images: list[Image.Image],
    batch_size: int,
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {block.name: [] for block in extractor.blocks}
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        extracted = extractor.extract(batch)
        for name, arr in extracted.items():
            parts[name].append(arr)
    return {name: np.concatenate(chunks, axis=0) for name, chunks in parts.items()}


def torch_pca_energy(
    features: torch.Tensor,
    mu: torch.Tensor,
    components: torch.Tensor,
    k: int,
) -> torch.Tensor:
    c = components[:, :k]
    centered = features - mu.reshape(1, -1)
    residual = centered - (centered @ c) @ c.T
    return torch.sum(residual * residual, dim=1)


def synthesize_adapter_pseudo_features(
    features: np.ndarray,
    coords: np.ndarray,
    sample_idx: np.ndarray,
    rng: np.random.Generator,
    pca_ref: dict,
    normal_energy_ref: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    base = features[sample_idx].astype(np.float32, copy=True)
    n_items = base.shape[0]
    donor_candidates = rng.integers(
        0,
        features.shape[0],
        size=(n_items, max(int(args.adapter_donor_candidates), 1)),
    )
    base_coords = coords[sample_idx]
    candidate_coords = coords[donor_candidates]
    coord_dist = np.sum((candidate_coords - base_coords[:, None, :]) ** 2, axis=2)
    donor_idx = donor_candidates[np.arange(n_items), np.argmax(coord_dist, axis=1)]
    donor = features[donor_idx].astype(np.float32, copy=False)

    mix_min = min(float(args.adapter_pseudo_mix_min), float(args.adapter_pseudo_mix_max))
    mix_max = max(float(args.adapter_pseudo_mix_min), float(args.adapter_pseudo_mix_max))
    lam = rng.uniform(mix_min, mix_max, size=(n_items, 1)).astype(np.float32)
    if args.adapter_pseudo_mode == "interpolate":
        pseudo = base + lam * (donor - base)
    elif args.adapter_pseudo_mode == "extrapolate":
        pseudo = base + lam * (base - donor)
    else:
        raise ValueError(f"Unknown adapter pseudo mode: {args.adapter_pseudo_mode}")

    noise_std = float(args.adapter_pseudo_noise_std)
    if noise_std > 0:
        scale = np.std(features, axis=0, keepdims=True).astype(np.float32)
        noise = rng.normal(0.0, noise_std, size=pseudo.shape).astype(np.float32)
        pseudo = pseudo + noise * np.maximum(scale, 1e-6)

    target = float(
        np.percentile(normal_energy_ref, float(args.adapter_margin_percentile))
        * float(args.adapter_margin_scale)
    )
    if args.adapter_enforce_pseudo_margin and target > 0:
        pseudo = enforce_pseudo_residual_margin(pseudo, pca_ref, target, rng)
    return pseudo.astype(np.float32)


def pca_residual_vectors(features: np.ndarray, pca_params: dict) -> np.ndarray:
    mu = np.asarray(pca_params["mu"], dtype=np.float32)
    comps = np.asarray(pca_params["components"], dtype=np.float32)
    k = int(pca_params["k"])
    c = comps[:, :k]
    centered = features.astype(np.float32, copy=False) - mu
    projected = centered @ c
    residual = centered - projected @ c.T
    return residual.astype(np.float32)


def enforce_pseudo_residual_margin(
    pseudo: np.ndarray,
    pca_ref: dict,
    target_energy: float,
    rng: np.random.Generator,
) -> np.ndarray:
    energy = pca_energy(pseudo, pca_ref)
    low = energy < target_energy
    if not np.any(low):
        return pseudo.astype(np.float32)

    adjusted = pseudo.astype(np.float32, copy=True)
    residual = pca_residual_vectors(adjusted[low], pca_ref)
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    fallback = rng.normal(0.0, 1.0, size=residual.shape).astype(np.float32)
    fallback = pca_residual_vectors(
        np.asarray(pca_ref["mu"], dtype=np.float32).reshape(1, -1) + fallback,
        pca_ref,
    )
    fallback_norms = np.linalg.norm(fallback, axis=1, keepdims=True)
    unit = np.where(
        norms > 1e-6,
        residual / np.maximum(norms, 1e-6),
        fallback / np.maximum(fallback_norms, 1e-6),
    )
    lift = np.sqrt(np.maximum(target_energy - energy[low], 0.0)).reshape(-1, 1)
    jitter = rng.uniform(1.0, 1.35, size=(lift.shape[0], 1)).astype(np.float32)
    adjusted[low] = adjusted[low] + unit * lift.astype(np.float32) * jitter
    return adjusted.astype(np.float32)


def fit_feature_adapter(
    features_spatial: np.ndarray,
    grid_size: tuple[int, int],
    seed: int,
    args: argparse.Namespace,
) -> FeatureAdapterState:
    features = features_spatial.reshape(-1, features_spatial.shape[-1]).astype(np.float32)
    feature_dim = features.shape[1]
    rank = max(1, min(int(args.adapter_rank), feature_dim))
    rng = np.random.default_rng(seed)
    n_fit = min(int(args.adapter_max_tokens), features.shape[0])
    sample_idx = rng.choice(features.shape[0], size=n_fit, replace=False)
    coords = grid_coordinates(grid_size, features_spatial.shape[0])
    normal = features[sample_idx]
    pca_ref = fit_pca(normal, float(args.adapter_pca_ev))
    normal_energy_before_arr = pca_energy(normal, pca_ref)
    pseudo = synthesize_adapter_pseudo_features(
        features,
        coords,
        sample_idx,
        rng,
        pca_ref,
        normal_energy_before_arr,
        args,
    )
    pseudo_energy_before_arr = pca_energy(pseudo, pca_ref)
    normal_energy_before = float(np.median(normal_energy_before_arr))
    pseudo_energy_before = float(np.median(pseudo_energy_before_arr))

    if int(args.adapter_steps) <= 0:
        mean = np.mean(normal, axis=0).astype(np.float32)
        std = np.std(normal, axis=0).astype(np.float32)
        return FeatureAdapterState(
            mean=mean,
            std=np.maximum(std, 1e-6),
            down_weight=np.zeros((rank, feature_dim), dtype=np.float32),
            up_weight=np.zeros((feature_dim, rank), dtype=np.float32),
            strength=float(args.adapter_strength),
            rank=rank,
            normal_energy_before=normal_energy_before,
            normal_energy_after=normal_energy_before,
            pseudo_energy_before=pseudo_energy_before,
            pseudo_energy_after=pseudo_energy_before,
            delta_rms=0.0,
        )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)
    mean_np = np.mean(normal, axis=0).astype(np.float32)
    std_np = np.maximum(np.std(normal, axis=0).astype(np.float32), 1e-6)
    mean_t = torch.from_numpy(mean_np).to(torch_device)
    std_t = torch.from_numpy(std_np).to(torch_device)
    model = LowRankResidualAdapter(
        feature_dim,
        rank,
        float(args.adapter_strength),
        mean_t,
        std_t,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.adapter_lr),
        weight_decay=float(args.adapter_weight_decay),
    )

    normal_t = torch.from_numpy(normal).to(torch_device)
    pseudo_t = torch.from_numpy(pseudo).to(torch_device)
    mu_t = torch.from_numpy(np.asarray(pca_ref["mu"], dtype=np.float32)).to(torch_device)
    comps_t = torch.from_numpy(np.asarray(pca_ref["components"], dtype=np.float32)).to(torch_device)
    k = int(pca_ref["k"])
    margin_ref = np.percentile(
        normal_energy_before_arr,
        float(args.adapter_margin_percentile),
    )
    margin_log = float(np.log1p(max(margin_ref, 1e-12) * float(args.adapter_margin_scale)))
    batch_tokens = min(max(int(args.adapter_batch_tokens), 1), n_fit)
    feature_var = float(np.mean(np.var(normal, axis=0)) + 1e-12)
    temp = max(float(args.adapter_loss_temp), 1e-6)

    for _ in range(int(args.adapter_steps)):
        batch_idx = torch.randint(0, n_fit, (batch_tokens,), device=torch_device)
        x_normal = normal_t[batch_idx]
        x_pseudo = pseudo_t[batch_idx]
        y_normal = model(x_normal)
        y_pseudo = model(x_pseudo)
        normal_energy = torch_pca_energy(y_normal, mu_t, comps_t, k)
        pseudo_energy = torch_pca_energy(y_pseudo, mu_t, comps_t, k)
        normal_loss = torch.log1p(normal_energy).mean()
        pseudo_loss = torch.nn.functional.softplus(
            (margin_log - torch.log1p(pseudo_energy)) / temp
        ).mean()
        delta = y_normal - x_normal
        reg_loss = torch.mean(delta * delta) / feature_var
        loss = (
            float(args.adapter_normal_weight) * normal_loss
            + float(args.adapter_pseudo_weight) * pseudo_loss
            + float(args.adapter_reg_weight) * reg_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.adapter_grad_clip))
        optimizer.step()

    with torch.inference_mode():
        normal_after = model(normal_t)
        pseudo_after = model(pseudo_t)
        normal_energy_after_arr = (
            torch_pca_energy(normal_after, mu_t, comps_t, k).detach().cpu().numpy()
        )
        pseudo_energy_after_arr = (
            torch_pca_energy(pseudo_after, mu_t, comps_t, k).detach().cpu().numpy()
        )
        delta_rms = float(
            torch.sqrt(torch.mean((normal_after - normal_t) ** 2)).detach().cpu().item()
        )

    return FeatureAdapterState(
        mean=mean_np,
        std=std_np,
        down_weight=model.down.weight.detach().cpu().numpy().astype(np.float32),
        up_weight=model.up.weight.detach().cpu().numpy().astype(np.float32),
        strength=float(args.adapter_strength),
        rank=rank,
        normal_energy_before=normal_energy_before,
        normal_energy_after=float(np.median(normal_energy_after_arr)),
        pseudo_energy_before=pseudo_energy_before,
        pseudo_energy_after=float(np.median(pseudo_energy_after_arr)),
        delta_rms=delta_rms,
    )


def apply_feature_adapter_flat(
    features: np.ndarray,
    adapter: FeatureAdapterState,
) -> np.ndarray:
    x = features.astype(np.float32, copy=False)
    z = (x - adapter.mean.reshape(1, -1)) / np.maximum(adapter.std.reshape(1, -1), 1e-6)
    hidden = np.tanh(z @ adapter.down_weight.T)
    delta = hidden @ adapter.up_weight.T
    return (x + adapter.strength * delta).astype(np.float32)


def apply_feature_adapters_spatial(
    spatial: dict[str, np.ndarray],
    adapters: dict[str, FeatureAdapterState],
) -> dict[str, np.ndarray]:
    adapted: dict[str, np.ndarray] = {}
    for name, arr in spatial.items():
        flat = arr.reshape(-1, arr.shape[-1])
        adapted[name] = apply_feature_adapter_flat(flat, adapters[name]).reshape(arr.shape)
    return adapted


def component_residuals(features: np.ndarray, pca_params: list[dict]) -> np.ndarray:
    parts = [pca_energy(features, params) for params in pca_params]
    return np.stack(parts, axis=1).astype(np.float32)


def pca_energy_with_mean(
    features: np.ndarray,
    pca_params: dict,
    mean: np.ndarray,
) -> np.ndarray:
    comps = np.asarray(pca_params["components"], dtype=np.float32)
    k = int(pca_params["k"])
    c = comps[:, :k]
    x = features.astype(np.float32, copy=False)
    centered = x - mean.astype(np.float32, copy=False)
    projected = centered @ c
    recon = projected @ c.T
    residual = centered - recon
    return np.sum(residual * residual, axis=1).astype(np.float32)


def assignment_distances(
    features: np.ndarray,
    coords: np.ndarray,
    centers: np.ndarray,
    coord_weight: float,
) -> np.ndarray:
    x = assignment_features(features, coords, coord_weight)
    diff = x[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2).astype(np.float32)


def soft_weights(distances: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-12)
    shifted = distances - np.min(distances, axis=1, keepdims=True)
    logits = -shifted / temp
    logits = np.clip(logits, -60.0, 0.0)
    weights = np.exp(logits)
    return weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)


def softmin_pair(a: np.ndarray, b: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-12)
    return (-temp * np.logaddexp(-np.asarray(a, dtype=np.float32) / temp, -np.asarray(b, dtype=np.float32) / temp)).astype(
        np.float32
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))).astype(np.float32)


def mixture_energy(
    features: np.ndarray,
    coords: np.ndarray,
    bank: MixtureBank,
    coord_weight: float,
    mode: str,
) -> np.ndarray:
    distances = assignment_distances(features, coords, bank.centers, coord_weight)
    residuals = component_residuals(features, bank.pca_params)
    if mode == "hard":
        idx = np.argmin(distances, axis=1)
        return residuals[np.arange(residuals.shape[0]), idx].astype(np.float32)
    if mode == "soft":
        weights = soft_weights(distances, bank.temperature)
        return np.sum(weights * residuals, axis=1).astype(np.float32)
    raise ValueError(f"Unknown mixture mode: {mode}")


def proto_center_energy(
    features: np.ndarray,
    coords: np.ndarray,
    bank: MixtureBank,
    coord_weight: float,
) -> np.ndarray:
    distances = assignment_distances(features, coords, bank.centers, coord_weight)
    labels = np.argmin(distances, axis=1)
    energy = np.zeros(features.shape[0], dtype=np.float32)
    for idx in range(bank.component_means.shape[0]):
        mask = labels == idx
        if not np.any(mask):
            continue
        energy[mask] = pca_energy_with_mean(
            features[mask],
            bank.global_pca,
            bank.component_means[idx],
        )
    return energy


def support_component_reliability(
    labels: np.ndarray,
    train_global_energy: np.ndarray,
    component_train_energy: list[np.ndarray],
    component_sizes: list[int],
    args: argparse.Namespace,
) -> np.ndarray:
    percentile = float(args.support_rel_percentile)
    values: list[float] = []
    for idx, size in enumerate(component_sizes):
        if size <= 0:
            values.append(0.0)
            continue
        mask = labels == idx
        global_part = train_global_energy[mask]
        local_part = component_train_energy[idx]
        if global_part.size == 0 or local_part.size == 0:
            values.append(0.0)
            continue
        global_q = float(np.percentile(global_part, percentile))
        local_q = float(np.percentile(local_part, percentile))
        gain = np.log((global_q + 1e-12) / (local_q + 1e-12))
        gain_rel = float(sigmoid(np.asarray([gain / max(args.support_rel_gain_scale, 1e-12)]))[0])
        size_rel = float(size / (size + max(args.min_cluster_tokens, 1)))
        values.append(float(np.clip(size_rel * gain_rel, 0.0, 1.0)))
    return np.asarray(values, dtype=np.float32)


def reliability_router_weights(
    features: np.ndarray,
    coords: np.ndarray,
    bank: MixtureBank,
    coord_weight: float,
    assignment_power: float,
) -> np.ndarray:
    distances = assignment_distances(features, coords, bank.centers, coord_weight)
    idx = np.argmin(distances, axis=1)
    n_components = distances.shape[1]
    if n_components == 1:
        confidence = np.ones(distances.shape[0], dtype=np.float32)
    else:
        weights = soft_weights(distances, bank.temperature)
        max_weight = np.max(weights, axis=1)
        chance = 1.0 / float(n_components)
        confidence = np.clip((max_weight - chance) / max(1.0 - chance, 1e-12), 0.0, 1.0).astype(
            np.float32
        )
        confidence = np.power(confidence, max(float(assignment_power), 1e-12)).astype(np.float32)
    return (bank.component_reliability[idx] * confidence).astype(np.float32)


def reliability_routed_z(
    dsb_z: np.ndarray,
    hard_z: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return (dsb_z + weights.astype(np.float32) * (hard_z - dsb_z)).astype(np.float32)


def router_signal_matrix(
    dsb_z: np.ndarray,
    hard_z: np.ndarray,
    soft_z: np.ndarray,
    anchor_z: np.ndarray,
    distances: np.ndarray,
    coords: np.ndarray,
    bank: MixtureBank,
) -> np.ndarray:
    idx = np.argmin(distances, axis=1)
    nearest = distances[np.arange(distances.shape[0]), idx]
    n_components = distances.shape[1]
    if n_components == 1:
        confidence = np.ones(distances.shape[0], dtype=np.float32)
    else:
        weights = soft_weights(distances, bank.temperature)
        max_weight = np.max(weights, axis=1)
        chance = 1.0 / float(n_components)
        confidence = np.clip((max_weight - chance) / max(1.0 - chance, 1e-12), 0.0, 1.0)

    dist_norm = np.log1p(nearest / max(float(bank.temperature), 1e-12))
    comp_rel = bank.component_reliability[idx]
    return np.stack(
        [
            np.clip(dsb_z, 0.0, 12.0),
            np.clip(hard_z, 0.0, 12.0),
            np.clip(soft_z, 0.0, 12.0),
            np.clip(anchor_z, 0.0, 12.0),
            np.clip(hard_z - dsb_z, -6.0, 6.0),
            np.clip(soft_z - dsb_z, -6.0, 6.0),
            np.clip(anchor_z - dsb_z, -6.0, 6.0),
            confidence.astype(np.float32),
            dist_norm.astype(np.float32),
            comp_rel.astype(np.float32),
            coords[:, 0].astype(np.float32),
            coords[:, 1].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def pseudo_router_input_from_features(
    features: np.ndarray,
    coords: np.ndarray,
    bank: MixtureBank,
    coord_weight: float,
) -> np.ndarray:
    global_energy = pca_energy(features, bank.global_pca)
    hard_energy = mixture_energy(features, coords, bank, coord_weight, "hard")
    soft_energy = mixture_energy(features, coords, bank, coord_weight, "soft")
    anchor_energy = softmin_pair(global_energy, hard_energy, bank.router_temperature)
    distances = assignment_distances(features, coords, bank.centers, coord_weight)
    return router_signal_matrix(
        robust_log_z(global_energy, bank.train_global_energy),
        robust_log_z(hard_energy, bank.train_hard_energy),
        robust_log_z(soft_energy, bank.train_soft_energy),
        robust_log_z(anchor_energy, bank.train_anchor_energy),
        distances,
        coords,
        bank,
    )


def pseudo_router_alpha(
    router: dict[str, np.ndarray] | None,
    signal: np.ndarray,
    alpha_power: float,
) -> np.ndarray:
    if router is None:
        return np.zeros(signal.shape[0], dtype=np.float32)
    x = (signal - router["mean"]) / router["std"]
    logits = x @ router["coef"] + float(router["intercept"][0])
    alpha = sigmoid(logits)
    power = max(float(alpha_power), 1e-12)
    if abs(power - 1.0) > 1e-12:
        alpha = np.power(alpha, power).astype(np.float32)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def synthesize_pseudo_features(
    features: np.ndarray,
    coords: np.ndarray,
    bank: MixtureBank,
    coord_weight: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_tokens = features.shape[0]
    sample_n = min(int(args.pseudo_router_max_tokens), n_tokens)
    if sample_n < 32:
        return None

    sample_idx = rng.choice(n_tokens, size=sample_n, replace=False)
    base = features[sample_idx].astype(np.float32, copy=True)
    base_coords = coords[sample_idx].astype(np.float32, copy=True)

    all_distances = assignment_distances(features, coords, bank.centers, coord_weight)
    all_labels = np.argmin(all_distances, axis=1)
    base_labels = all_labels[sample_idx]
    donor_idx = rng.integers(0, n_tokens, size=sample_n)
    if np.unique(all_labels).size > 1:
        for _ in range(8):
            same = all_labels[donor_idx] == base_labels
            if not np.any(same):
                break
            donor_idx[same] = rng.integers(0, n_tokens, size=int(np.sum(same)))

    donor = features[donor_idx].astype(np.float32, copy=False)
    mix_min = min(float(args.pseudo_router_mix_min), float(args.pseudo_router_mix_max))
    mix_max = max(float(args.pseudo_router_mix_min), float(args.pseudo_router_mix_max))
    lam = rng.uniform(mix_min, mix_max, size=(sample_n, 1)).astype(np.float32)
    pseudo = (base + lam * (donor - base)).astype(np.float32)
    return base, base_coords, pseudo


def fit_pseudo_router(
    features: np.ndarray,
    coords: np.ndarray,
    bank: MixtureBank,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, np.ndarray] | None:
    if not args.enable_pseudo_router:
        return None

    rng = np.random.default_rng(seed + 104729)
    pseudo_pack = synthesize_pseudo_features(features, coords, bank, args.coord_weight, rng, args)
    if pseudo_pack is None:
        return None
    normal_features, normal_coords, pseudo_features = pseudo_pack

    x_normal = pseudo_router_input_from_features(normal_features, normal_coords, bank, args.coord_weight)
    x_pseudo = pseudo_router_input_from_features(pseudo_features, normal_coords, bank, args.coord_weight)
    x = np.concatenate([x_normal, x_pseudo], axis=0)
    y = np.concatenate(
        [
            np.zeros(x_normal.shape[0], dtype=np.uint8),
            np.ones(x_pseudo.shape[0], dtype=np.uint8),
        ],
        axis=0,
    )
    finite = np.all(np.isfinite(x), axis=1)
    x = x[finite]
    y = y[finite]
    if x.shape[0] < 64 or np.unique(y).size < 2:
        return None

    mean = np.mean(x, axis=0).astype(np.float32)
    std = np.std(x, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    x_scaled = (x - mean) / std

    model = LogisticRegression(
        C=float(args.pseudo_router_c),
        class_weight="balanced",
        max_iter=int(args.pseudo_router_max_iter),
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(x_scaled, y)
    router = {
        "mean": mean,
        "std": std,
        "coef": model.coef_[0].astype(np.float32),
        "intercept": model.intercept_.astype(np.float32),
    }
    alpha_normal = pseudo_router_alpha(router, x_normal, args.pseudo_router_alpha_power)
    alpha_pseudo = pseudo_router_alpha(router, x_pseudo, args.pseudo_router_alpha_power)
    router["normal_alpha_mean"] = np.asarray([float(np.mean(alpha_normal))], dtype=np.float32)
    router["pseudo_alpha_mean"] = np.asarray([float(np.mean(alpha_pseudo))], dtype=np.float32)
    return router


def component_calibrated_residuals(
    features: np.ndarray,
    bank: MixtureBank,
) -> np.ndarray:
    residuals = component_residuals(features, bank.pca_params)
    calibrated = [
        robust_log_z(residuals[:, idx], bank.component_train_energy[idx])
        for idx in range(residuals.shape[1])
    ]
    return np.stack(calibrated, axis=1).astype(np.float32)


def calibrated_mixture_z(
    features: np.ndarray,
    coords: np.ndarray,
    bank: MixtureBank,
    coord_weight: float,
    mode: str,
) -> np.ndarray:
    distances = assignment_distances(features, coords, bank.centers, coord_weight)
    residual_z = component_calibrated_residuals(features, bank)
    if mode == "hard":
        idx = np.argmin(distances, axis=1)
        return residual_z[np.arange(residual_z.shape[0]), idx].astype(np.float32)
    if mode == "soft":
        weights = soft_weights(distances, bank.temperature)
        return np.sum(weights * residual_z, axis=1).astype(np.float32)
    raise ValueError(f"Unknown calibrated mixture mode: {mode}")


def fit_mixture_bank(
    features_spatial: np.ndarray,
    grid_size: tuple[int, int],
    seed: int,
    args: argparse.Namespace,
) -> MixtureBank:
    n_images = features_spatial.shape[0]
    features = features_spatial.reshape(-1, features_spatial.shape[-1]).astype(np.float32)
    coords = grid_coordinates(grid_size, n_images)
    global_pca = fit_pca(features, args.global_pca_ev)
    train_global_energy = pca_energy(features, global_pca)

    if n_images < args.min_ref_images_for_mixture:
        assign = assignment_features(features, coords, args.coord_weight)
        centers = np.mean(assign, axis=0, keepdims=True)
        return MixtureBank(
            centers=centers.astype(np.float32),
            pca_params=[global_pca],
            component_means=np.asarray([global_pca["mu"]], dtype=np.float32),
            component_train_energy=[train_global_energy],
            component_reliability=np.ones(1, dtype=np.float32),
            global_pca=global_pca,
            train_global_energy=train_global_energy,
            train_hard_energy=train_global_energy,
            train_soft_energy=train_global_energy,
            train_anchor_energy=train_global_energy,
            train_proto_center_energy=train_global_energy,
            temperature=1.0,
            router_temperature=1.0,
            component_sizes=[int(features.shape[0])],
            pseudo_router=None,
        )

    n_clusters = max(1, min(args.n_components, features.shape[0] // max(args.min_cluster_tokens, 2)))
    n_clusters = min(n_clusters, features.shape[0])
    assign = assignment_features(features, coords, args.coord_weight)
    if n_clusters == 1:
        labels = np.zeros(features.shape[0], dtype=np.int32)
        centers = np.mean(assign, axis=0, keepdims=True)
    else:
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(args.kmeans_batch_size, max(features.shape[0], n_clusters)),
            n_init=10,
            max_iter=args.kmeans_max_iter,
            reassignment_ratio=0.0,
        )
        labels = clusterer.fit_predict(assign).astype(np.int32)
        centers = clusterer.cluster_centers_.astype(np.float32)

    pca_params: list[dict] = []
    component_means: list[np.ndarray] = []
    component_train_energy: list[np.ndarray] = []
    component_sizes: list[int] = []
    for idx in range(n_clusters):
        mask = labels == idx
        component = features[mask]
        component_sizes.append(int(component.shape[0]))
        if component.shape[0] > 0:
            component_means.append(np.mean(component, axis=0).astype(np.float32))
        else:
            component_means.append(np.asarray(global_pca["mu"], dtype=np.float32))
        if n_clusters == 1:
            pca_params.append(global_pca)
        elif component.shape[0] >= args.min_cluster_tokens:
            pca_params.append(fit_pca(component, args.local_pca_ev))
        else:
            pca_params.append(global_pca)
        if component.shape[0] > 0:
            component_train_energy.append(pca_energy(component, pca_params[-1]))
        else:
            component_train_energy.append(train_global_energy)

    component_reliability = support_component_reliability(
        labels,
        train_global_energy,
        component_train_energy,
        component_sizes,
        args,
    )

    distances = assignment_distances(features, coords, centers, args.coord_weight)
    nearest = np.min(distances, axis=1)
    temperature = float(np.median(nearest) * args.assignment_temp_scale)
    if temperature < 1e-6:
        temperature = float(np.percentile(distances, 25) * args.assignment_temp_scale)
    temperature = max(temperature, 1e-6)

    bank = MixtureBank(
        centers=centers,
        pca_params=pca_params,
        component_means=np.stack(component_means, axis=0).astype(np.float32),
        component_train_energy=component_train_energy,
        component_reliability=component_reliability,
        global_pca=global_pca,
        train_global_energy=train_global_energy,
        train_hard_energy=np.zeros_like(train_global_energy),
        train_soft_energy=np.zeros_like(train_global_energy),
        train_anchor_energy=np.zeros_like(train_global_energy),
        train_proto_center_energy=np.zeros_like(train_global_energy),
        temperature=temperature,
        router_temperature=1.0,
        component_sizes=component_sizes,
        pseudo_router=None,
    )
    bank.train_hard_energy = mixture_energy(features, coords, bank, args.coord_weight, "hard")
    bank.train_soft_energy = mixture_energy(features, coords, bank, args.coord_weight, "soft")
    bank.train_proto_center_energy = proto_center_energy(features, coords, bank, args.coord_weight)
    router_temperature = float(np.median(np.abs(bank.train_global_energy - bank.train_hard_energy)) * args.router_temp_scale)
    if not np.isfinite(router_temperature) or router_temperature < 1e-6:
        router_temperature = 1.0
    bank.router_temperature = router_temperature
    bank.train_anchor_energy = softmin_pair(
        bank.train_global_energy,
        bank.train_hard_energy,
        bank.router_temperature,
    )
    bank.pseudo_router = fit_pseudo_router(features, coords, bank, seed, args)
    return bank


def percentile_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(float(denominator), 1e-12))


def support_gate_from_scores(
    dsb_scores: list[float],
    hard_scores: list[float],
    n_fit: int,
    args: argparse.Namespace,
) -> SupportValGate:
    dsb_arr = np.asarray(dsb_scores, dtype=np.float64)
    hard_arr = np.asarray(hard_scores, dtype=np.float64)
    dsb_median = float(np.median(dsb_arr))
    hard_median = float(np.median(hard_arr))
    dsb_q90 = float(np.percentile(dsb_arr, 90))
    hard_q90 = float(np.percentile(hard_arr, 90))
    median_ratio = percentile_ratio(hard_median, dsb_median)
    q90_ratio = percentile_ratio(hard_q90, dsb_q90)

    trust = 1.0
    reason = "trusted"
    if median_ratio > args.support_val_median_ratio_max:
        trust = 0.0
        reason = "median_ratio_high"
    if q90_ratio > args.support_val_q90_ratio_max:
        trust = 0.0
        reason = "q90_ratio_high" if reason == "trusted" else f"{reason}+q90_ratio_high"

    return SupportValGate(
        trust=trust,
        n_fit=n_fit,
        n_val=len(dsb_scores),
        dsb_median=dsb_median,
        hard_median=hard_median,
        dsb_q90=dsb_q90,
        hard_q90=hard_q90,
        median_ratio=median_ratio,
        q90_ratio=q90_ratio,
        reason=reason,
    )


def fused_validation_maps(
    spatial: dict[str, np.ndarray],
    banks: dict[str, MixtureBank],
    block_names: list[str],
    grid_size: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    n_images = next(iter(spatial.values())).shape[0]
    coords = grid_coordinates(grid_size, n_images)
    block_z: dict[str, list[np.ndarray]] = {
        "dsb": [],
        "hard": [],
        "anchor": [],
        "proto_center": [],
    }
    if args.enable_pseudo_router:
        block_z["prouter_hard"] = []
        block_z["prouter_anchor"] = []
        block_z["prouter_boost"] = []

    for name in block_names:
        features = spatial[name].reshape(-1, spatial[name].shape[-1]).astype(np.float32)
        bank = banks[name]
        global_energy = pca_energy(features, bank.global_pca)
        hard_energy = mixture_energy(features, coords, bank, args.coord_weight, "hard")
        soft_energy = mixture_energy(features, coords, bank, args.coord_weight, "soft")
        anchor_energy = softmin_pair(global_energy, hard_energy, bank.router_temperature)
        proto_energy = proto_center_energy(features, coords, bank, args.coord_weight)
        dsb_z = robust_log_z(global_energy, bank.train_global_energy)
        hard_z = robust_log_z(hard_energy, bank.train_hard_energy)
        soft_z = robust_log_z(soft_energy, bank.train_soft_energy)
        anchor_z = robust_log_z(anchor_energy, bank.train_anchor_energy)
        proto_z = robust_log_z(proto_energy, bank.train_proto_center_energy)
        block_z["dsb"].append(dsb_z)
        block_z["hard"].append(hard_z)
        block_z["anchor"].append(anchor_z)
        block_z["proto_center"].append(proto_z)

        if args.enable_pseudo_router:
            distances = assignment_distances(features, coords, bank.centers, args.coord_weight)
            router_signal = router_signal_matrix(
                dsb_z,
                hard_z,
                soft_z,
                anchor_z,
                distances,
                coords,
                bank,
            )
            alpha = pseudo_router_alpha(
                bank.pseudo_router,
                router_signal,
                args.pseudo_router_alpha_power,
            )
            block_z["prouter_hard"].append(reliability_routed_z(dsb_z, hard_z, alpha))
            block_z["prouter_anchor"].append(reliability_routed_z(dsb_z, anchor_z, alpha))
            block_z["prouter_boost"].append(
                (dsb_z + alpha * np.maximum(hard_z - dsb_z, 0.0)).astype(np.float32)
            )

    return {
        key: np.mean(np.stack(values, axis=0), axis=0)
        for key, values in block_z.items()
    }


def raw_fr_scores_from_maps(
    fused: dict[str, np.ndarray],
    image: Image.Image,
    sl: slice,
    grid_size: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, float]:
    fg_mask = image_foreground_mask(image, grid_size, args.image_res)

    def fr_from_key(key: str) -> float:
        values = fused[key][sl]
        full = image_score(values, grid_size, args.image_res, args.top_frac)
        roi = image_score_masked(values, fg_mask, grid_size, args.image_res, args.top_frac)
        return 0.5 * (full + roi)

    scores = {
        "dsb_fr": fr_from_key("dsb"),
        "mlsb_hard_fr": fr_from_key("hard"),
        "mlsb_anchor_softmin_fr": fr_from_key("anchor"),
        "mlsb_proto_center_fr": fr_from_key("proto_center"),
    }
    if "prouter_hard" in fused:
        scores["mlsb_prouter_hard_fr"] = fr_from_key("prouter_hard")
        scores["mlsb_prouter_anchor_fr"] = fr_from_key("prouter_anchor")
        scores["mlsb_prouter_boost_fr"] = fr_from_key("prouter_boost")
    return scores


def add_support_calibrated_fr_scores(
    scores: dict[str, float],
    support_gate: SupportValGate,
    support_cal_scale: float,
    args: argparse.Namespace,
) -> None:
    scores["mlsb_support_cal_hard_fr"] = support_calibrated_score(
        scores["mlsb_hard_fr"],
        support_gate,
        support_cal_scale,
    )
    blend_weight = args.support_cal_blend_weight
    scores["mlsb_support_cal_blend_fr"] = (
        (1.0 - blend_weight) * scores["dsb_fr"]
        + blend_weight * scores["mlsb_support_cal_hard_fr"]
    )


def pseudo_validation_spatial(
    train_spatial: dict[str, np.ndarray],
    fit_idx: list[int],
    val_idx: list[int],
    banks: dict[str, MixtureBank],
    block_names: list[str],
    grid_size: tuple[int, int],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 15485863)
    n_tokens = grid_size[0] * grid_size[1]
    frac = float(np.clip(args.selfval_pseudo_frac, 0.0, 1.0))
    n_change = max(1, int(round(n_tokens * frac)))
    pseudo_spatial: dict[str, np.ndarray] = {}

    for name in block_names:
        val = train_spatial[name][val_idx].astype(np.float32, copy=True)
        flat = val.reshape(-1, val.shape[-1])
        fit = train_spatial[name][fit_idx].reshape(-1, val.shape[-1]).astype(np.float32)
        bank = banks[name]

        fit_coords = grid_coordinates(grid_size, len(fit_idx))
        fit_distances = assignment_distances(fit, fit_coords, bank.centers, args.coord_weight)
        fit_labels = np.argmin(fit_distances, axis=1)
        val_coords = grid_coordinates(grid_size, len(val_idx))
        val_distances = assignment_distances(flat, val_coords, bank.centers, args.coord_weight)
        val_labels = np.argmin(val_distances, axis=1)
        all_fit = np.arange(fit.shape[0])

        for image_idx in range(len(val_idx)):
            local_tokens = rng.choice(n_tokens, size=min(n_change, n_tokens), replace=False)
            global_tokens = image_idx * n_tokens + local_tokens
            donor_tokens = np.empty(global_tokens.shape[0], dtype=np.int64)
            for j, token_idx in enumerate(global_tokens):
                different = all_fit[fit_labels != val_labels[token_idx]]
                pool = different if different.size > 0 else all_fit
                donor_tokens[j] = int(rng.choice(pool))
            mix_min = min(float(args.pseudo_router_mix_min), float(args.pseudo_router_mix_max))
            mix_max = max(float(args.pseudo_router_mix_min), float(args.pseudo_router_mix_max))
            lam = rng.uniform(mix_min, mix_max, size=(global_tokens.shape[0], 1)).astype(np.float32)
            flat[global_tokens] = flat[global_tokens] + lam * (fit[donor_tokens] - flat[global_tokens])

        pseudo_spatial[name] = flat.reshape(val.shape).astype(np.float32)
    return pseudo_spatial


def support_selfval_selector(
    train_spatial: dict[str, np.ndarray],
    refs: list[Image.Image],
    block_names: list[str],
    grid_size: tuple[int, int],
    seed: int,
    args: argparse.Namespace,
) -> SupportSelfValSelector:
    fallback = args.selfval_fallback_method
    n_images = len(refs)
    if not args.enable_selfval_selector:
        return SupportSelfValSelector(fallback, "disabled", n_images, 0, {}, {})
    if n_images < args.support_val_min_images:
        return SupportSelfValSelector(fallback, "too_few_support_views", n_images, 0, {}, {})

    stride = max(int(args.support_val_stride), 2)
    val_idx = [idx for idx in range(n_images) if idx % stride == 0]
    fit_idx = [idx for idx in range(n_images) if idx % stride != 0]
    if len(val_idx) == 0 or len(fit_idx) < args.min_ref_images_for_mixture:
        return SupportSelfValSelector(fallback, "invalid_support_split", len(fit_idx), len(val_idx), {}, {})

    banks = {
        name: fit_mixture_bank(arr[fit_idx], grid_size, seed + 32452843, args)
        for name, arr in train_spatial.items()
    }
    normal_spatial = {name: arr[val_idx] for name, arr in train_spatial.items()}
    pseudo_spatial = pseudo_validation_spatial(
        train_spatial,
        fit_idx,
        val_idx,
        banks,
        block_names,
        grid_size,
        seed,
        args,
    )

    normal_fused = fused_validation_maps(normal_spatial, banks, block_names, grid_size, args)
    pseudo_fused = fused_validation_maps(pseudo_spatial, banks, block_names, grid_size, args)
    n_tokens = grid_size[0] * grid_size[1]

    normal_scores: list[dict[str, float]] = []
    pseudo_scores: list[dict[str, float]] = []
    for i, ref_idx in enumerate(val_idx):
        sl = slice(i * n_tokens, (i + 1) * n_tokens)
        normal_scores.append(raw_fr_scores_from_maps(normal_fused, refs[ref_idx], sl, grid_size, args))
        pseudo_scores.append(raw_fr_scores_from_maps(pseudo_fused, refs[ref_idx], sl, grid_size, args))

    split_gate = support_gate_from_scores(
        [scores["dsb_fr"] for scores in normal_scores],
        [scores["mlsb_hard_fr"] for scores in normal_scores],
        len(fit_idx),
        args,
    )
    split_scale = support_calibration_scale(
        split_gate,
        args.support_cal_scale_min,
        args.support_cal_scale_max,
    )
    for scores in [*normal_scores, *pseudo_scores]:
        add_support_calibrated_fr_scores(scores, split_gate, split_scale, args)

    available = set(normal_scores[0]) & set(pseudo_scores[0])
    candidates = [method for method in args.selfval_candidates if method in available]
    if fallback not in candidates and fallback in available:
        candidates.insert(0, fallback)
    if not candidates:
        return SupportSelfValSelector(fallback, "no_available_candidates", len(fit_idx), len(val_idx), {}, {})

    labels = np.asarray([0] * len(normal_scores) + [1] * len(pseudo_scores), dtype=np.uint8)
    method_aurocs: dict[str, float] = {}
    method_margins: dict[str, float] = {}
    selector_scores: dict[str, float] = {}
    for method in candidates:
        normal = np.asarray([scores[method] for scores in normal_scores], dtype=np.float64)
        pseudo = np.asarray([scores[method] for scores in pseudo_scores], dtype=np.float64)
        values = np.concatenate([normal, pseudo], axis=0)
        if np.unique(values).size < 2:
            auroc = 0.5
        else:
            auroc = float(roc_auc_score(labels, values))
        margin = float(np.median(pseudo) - np.median(normal))
        method_aurocs[method] = auroc
        method_margins[method] = margin
        selector_scores[method] = auroc + float(args.selfval_margin_weight) * np.tanh(margin)

    selected = candidates[0]
    best_score = selector_scores[selected]
    tie_eps = max(float(args.selfval_tie_eps), 0.0)
    for method in candidates[1:]:
        score = selector_scores[method]
        if score > best_score + tie_eps:
            selected = method
            best_score = score

    reason = "selected_by_support_pseudo_auroc"
    fallback_auc = method_aurocs.get(fallback, float("-inf"))
    selected_auc = method_aurocs.get(selected, float("-inf"))
    if selected != fallback and selected_auc < float(args.selfval_min_candidate_auc):
        selected = fallback
        reason = "fallback_low_selfval_confidence"
    elif selected != fallback and selected_auc < fallback_auc + float(args.selfval_min_auc_gain):
        selected = fallback
        reason = "fallback_no_selfval_gain"

    return SupportSelfValSelector(
        selected_method=selected,
        reason=reason,
        n_fit=len(fit_idx),
        n_val=len(val_idx),
        method_aurocs=method_aurocs,
        method_margins=method_margins,
    )


def support_validation_gate(
    train_spatial: dict[str, np.ndarray],
    refs: list[Image.Image],
    block_names: list[str],
    grid_size: tuple[int, int],
    seed: int,
    args: argparse.Namespace,
) -> SupportValGate:
    n_images = len(refs)
    if n_images < args.support_val_min_images:
        return SupportValGate(
            trust=0.0,
            n_fit=n_images,
            n_val=0,
            dsb_median=0.0,
            hard_median=0.0,
            dsb_q90=0.0,
            hard_q90=0.0,
            median_ratio=float("inf"),
            q90_ratio=float("inf"),
            reason="too_few_support_views",
        )

    stride = max(int(args.support_val_stride), 2)
    val_idx = [idx for idx in range(n_images) if idx % stride == 0]
    fit_idx = [idx for idx in range(n_images) if idx % stride != 0]
    if len(val_idx) == 0 or len(fit_idx) < args.min_ref_images_for_mixture:
        return SupportValGate(
            trust=0.0,
            n_fit=len(fit_idx),
            n_val=len(val_idx),
            dsb_median=0.0,
            hard_median=0.0,
            dsb_q90=0.0,
            hard_q90=0.0,
            median_ratio=float("inf"),
            q90_ratio=float("inf"),
            reason="invalid_support_split",
        )

    banks = {
        name: fit_mixture_bank(arr[fit_idx], grid_size, seed + 7919, args)
        for name, arr in train_spatial.items()
    }

    coords = grid_coordinates(grid_size, len(val_idx))
    block_z: dict[str, list[np.ndarray]] = {"dsb": [], "hard": [], "anchor": []}
    for name in block_names:
        bank = banks[name]
        features = train_spatial[name][val_idx].reshape(
            -1, train_spatial[name].shape[-1]
        ).astype(np.float32)
        global_energy = pca_energy(features, bank.global_pca)
        hard_energy = mixture_energy(features, coords, bank, args.coord_weight, "hard")
        anchor_energy = softmin_pair(global_energy, hard_energy, bank.router_temperature)
        block_z["dsb"].append(robust_log_z(global_energy, bank.train_global_energy))
        block_z["hard"].append(robust_log_z(hard_energy, bank.train_hard_energy))
        block_z["anchor"].append(robust_log_z(anchor_energy, bank.train_anchor_energy))

    fused = {
        key: np.mean(np.stack(values, axis=0), axis=0)
        for key, values in block_z.items()
    }
    n_tokens = grid_size[0] * grid_size[1]
    dsb_scores: list[float] = []
    hard_scores: list[float] = []
    for i, ref_idx in enumerate(val_idx):
        sl = slice(i * n_tokens, (i + 1) * n_tokens)
        fg_mask = image_foreground_mask(refs[ref_idx], grid_size, args.image_res)
        dsb_full = image_score(fused["dsb"][sl], grid_size, args.image_res, args.top_frac)
        dsb_roi = image_score_masked(
            fused["dsb"][sl], fg_mask, grid_size, args.image_res, args.top_frac
        )
        hard_full = image_score(
            fused["hard"][sl], grid_size, args.image_res, args.top_frac
        )
        hard_roi = image_score_masked(
            fused["hard"][sl], fg_mask, grid_size, args.image_res, args.top_frac
        )
        dsb_scores.append(0.5 * (dsb_full + dsb_roi))
        hard_scores.append(0.5 * (hard_full + hard_roi))

    dsb_arr = np.asarray(dsb_scores, dtype=np.float64)
    hard_arr = np.asarray(hard_scores, dtype=np.float64)
    dsb_median = float(np.median(dsb_arr))
    hard_median = float(np.median(hard_arr))
    dsb_q90 = float(np.percentile(dsb_arr, 90))
    hard_q90 = float(np.percentile(hard_arr, 90))
    median_ratio = percentile_ratio(hard_median, dsb_median)
    q90_ratio = percentile_ratio(hard_q90, dsb_q90)

    trust = 1.0
    reason = "trusted"
    if median_ratio > args.support_val_median_ratio_max:
        trust = 0.0
        reason = "median_ratio_high"
    if q90_ratio > args.support_val_q90_ratio_max:
        trust = 0.0
        reason = "q90_ratio_high" if reason == "trusted" else f"{reason}+q90_ratio_high"

    return SupportValGate(
        trust=trust,
        n_fit=len(fit_idx),
        n_val=len(val_idx),
        dsb_median=dsb_median,
        hard_median=hard_median,
        dsb_q90=dsb_q90,
        hard_q90=hard_q90,
        median_ratio=median_ratio,
        q90_ratio=q90_ratio,
        reason=reason,
    )


def support_refit_bank_gate(
    train_spatial: dict[str, np.ndarray],
    refs: list[Image.Image],
    block_names: list[str],
    grid_size: tuple[int, int],
    banks: dict[str, MixtureBank],
    args: argparse.Namespace,
) -> SupportValGate:
    """Score the split validation views with banks refit on all support views.

    This is intentionally an in-sample sensitivity diagnostic.  It uses the
    same validation indices as ``support_validation_gate`` but the banks that
    are actually used at test time, so it must not be presented as an
    independent validation estimate.
    """
    n_images = len(refs)
    stride = max(int(args.support_val_stride), 2)
    val_idx = [idx for idx in range(n_images) if idx % stride == 0]
    if n_images < args.support_val_min_images or len(val_idx) == 0:
        return SupportValGate(
            trust=0.0,
            n_fit=n_images,
            n_val=len(val_idx),
            dsb_median=0.0,
            hard_median=0.0,
            dsb_q90=0.0,
            hard_q90=0.0,
            median_ratio=float("inf"),
            q90_ratio=float("inf"),
            reason="refit_diagnostic_unavailable",
        )

    val_spatial = {name: arr[val_idx] for name, arr in train_spatial.items()}
    fused = fused_validation_maps(val_spatial, banks, block_names, grid_size, args)
    n_tokens = grid_size[0] * grid_size[1]
    scores = [
        raw_fr_scores_from_maps(
            fused,
            refs[ref_idx],
            slice(i * n_tokens, (i + 1) * n_tokens),
            grid_size,
            args,
        )
        for i, ref_idx in enumerate(val_idx)
    ]
    gate = support_gate_from_scores(
        [score["dsb_fr"] for score in scores],
        [score["mlsb_hard_fr"] for score in scores],
        n_images,
        args,
    )
    gate.reason = f"refit_bank_in_sample:{gate.reason}"
    return gate


def support_calibration_scale(
    support_gate: SupportValGate,
    scale_min: float,
    scale_max: float,
) -> float:
    dsb_iqr = support_gate.dsb_q90 - support_gate.dsb_median
    hard_iqr = support_gate.hard_q90 - support_gate.hard_median
    if dsb_iqr <= 1e-12 or hard_iqr <= 1e-12:
        return 1.0
    return float(np.clip(dsb_iqr / hard_iqr, scale_min, scale_max))


def support_calibrated_score(
    hard_score: float,
    support_gate: SupportValGate,
    scale: float,
) -> float:
    return float(
        support_gate.dsb_median
        + (hard_score - support_gate.hard_median) * scale
    )


def evaluate_category(
    extractor: MultiLayerExtractor,
    dataset_root: Path,
    category: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[MetricRow], list[dict[str, object]]]:
    set_seed(seed)
    refs = reference_images(
        train_paths(dataset_root, category),
        category,
        seed,
        args.k_shot,
        args.aug_count,
        args.image_res,
    )
    if not refs:
        raise RuntimeError(f"No reference images for {category}")

    train_spatial = collect_spatial_features(extractor, refs, args.batch_size)
    grid_size = extractor.grid_size
    if grid_size is None:
        raise RuntimeError("Extractor did not infer grid size.")

    banks = {
        name: fit_mixture_bank(arr, grid_size, seed, args)
        for name, arr in train_spatial.items()
    }
    block_names = [block.name for block in extractor.blocks]
    feature_adapters: dict[str, FeatureAdapterState] = {}
    adapted_train_spatial: dict[str, np.ndarray] = {}
    adapted_banks: dict[str, MixtureBank] = {}
    adapted_support_gate: SupportValGate | None = None
    adapted_support_cal_scale = 1.0
    if args.enable_feature_adapter:
        feature_adapters = {
            name: fit_feature_adapter(arr, grid_size, seed + 1009 * (idx + 1), args)
            for idx, (name, arr) in enumerate(train_spatial.items())
        }
        adapted_train_spatial = apply_feature_adapters_spatial(train_spatial, feature_adapters)
        adapted_banks = {
            name: fit_mixture_bank(arr, grid_size, seed + 4241, args)
            for name, arr in adapted_train_spatial.items()
        }
        adapted_support_gate = support_validation_gate(
            adapted_train_spatial,
            refs,
            block_names,
            grid_size,
            seed + 5939,
            args,
        )
        adapted_support_cal_scale = support_calibration_scale(
            adapted_support_gate,
            args.support_cal_scale_min,
            args.support_cal_scale_max,
        )
    support_gate = support_validation_gate(
        train_spatial,
        refs,
        block_names,
        grid_size,
        seed,
        args,
    )
    support_cal_scale = support_calibration_scale(
        support_gate,
        args.support_cal_scale_min,
        args.support_cal_scale_max,
    )
    refit_support_gate: SupportValGate | None = None
    refit_support_cal_scale = 1.0
    if args.diagnose_refit_calibration:
        refit_support_gate = support_refit_bank_gate(
            train_spatial,
            refs,
            block_names,
            grid_size,
            banks,
            args,
        )
        refit_support_cal_scale = support_calibration_scale(
            refit_support_gate,
            args.support_cal_scale_min,
            args.support_cal_scale_max,
        )
    selfval_selector = support_selfval_selector(
        train_spatial,
        refs,
        block_names,
        grid_size,
        seed,
        args,
    )

    methods = [
        "dsb_fusion_full",
        "dsb_fusion_roi",
        "dsb_fr",
        "mlsb_hard_full",
        "mlsb_hard_roi",
        "mlsb_hard_fr",
        "mlsb_soft_full",
        "mlsb_soft_roi",
        "mlsb_soft_fr",
        "mlsb_relgate_hard_full",
        "mlsb_relgate_hard_roi",
        "mlsb_relgate_hard_fr",
        "mlsb_relanchor_softmin_full",
        "mlsb_relanchor_softmin_roi",
        "mlsb_relanchor_softmin_fr",
        "mlsb_anchor_softmin_full",
        "mlsb_anchor_softmin_roi",
        "mlsb_anchor_softmin_fr",
        "mlsb_proto_center_full",
        "mlsb_proto_center_roi",
        "mlsb_proto_center_fr",
        "mlsb_valgate_hard_full",
        "mlsb_valgate_hard_roi",
        "mlsb_valgate_hard_fr",
        "mlsb_support_cal_hard_fr",
        "mlsb_support_cal_blend_fr",
    ]
    if args.diagnose_refit_calibration:
        methods.extend(
            [
                "mlsb_refit_support_cal_hard_fr",
                "mlsb_refit_support_cal_blend_fr",
            ]
        )
    if args.enable_feature_adapter:
        methods.extend(
            [
                "sfa_dsb_full",
                "sfa_dsb_roi",
                "sfa_dsb_fr",
                "sfa_hard_full",
                "sfa_hard_roi",
                "sfa_hard_fr",
                "sfa_anchor_softmin_full",
                "sfa_anchor_softmin_roi",
                "sfa_anchor_softmin_fr",
                "sfa_support_cal_hard_fr",
                "sfa_support_cal_blend_fr",
            ]
        )
    if args.enable_selfval_selector:
        methods.append("mlsb_selfval_select_fr")
    if args.enable_pseudo_router:
        methods.extend(
            [
                "mlsb_prouter_hard_full",
                "mlsb_prouter_hard_roi",
                "mlsb_prouter_hard_fr",
                "mlsb_prouter_anchor_full",
                "mlsb_prouter_anchor_roi",
                "mlsb_prouter_anchor_fr",
                "mlsb_prouter_boost_full",
                "mlsb_prouter_boost_roi",
                "mlsb_prouter_boost_fr",
            ]
        )
    if args.enable_component_calibration:
        methods.extend(
            [
                "mlsb_calhard_full",
                "mlsb_calhard_roi",
                "mlsb_calhard_fr",
                "mlsb_calanchor_softmin_full",
                "mlsb_calanchor_softmin_roi",
                "mlsb_calanchor_softmin_fr",
            ]
        )
    method_scores: dict[str, list[float]] = {name: [] for name in methods}
    detail_rows: list[dict[str, object]] = []
    labels: list[int] = []
    pixel_gt_maps: list[np.ndarray] = []
    pixel_maps: dict[str, list[np.ndarray]] = {
        "dsb_map": [],
        "mlsb_hard_map": [],
        "sc_mlsb_map": [],
    }
    pixel_paths: list[str] = []

    paths = test_paths(dataset_root, category, args.debug_limit)
    iterator = tqdm(paths, desc=f"{category} seed={seed}", leave=False)
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        extracted = extractor.extract(images)
        batch_size = len(batch_paths)
        coords = grid_coordinates(grid_size, batch_size)

        block_z: dict[str, list[np.ndarray]] = {
            "dsb": [],
            "hard": [],
            "soft": [],
            "relgate": [],
            "relanchor": [],
            "anchor": [],
            "proto_center": [],
        }
        if args.enable_pseudo_router:
            block_z["prouter_hard"] = []
            block_z["prouter_anchor"] = []
            block_z["prouter_boost"] = []
        if args.enable_component_calibration:
            block_z["calhard"] = []
            block_z["calanchor"] = []
        sfa_block_z: dict[str, list[np.ndarray]] = {}
        if args.enable_feature_adapter:
            sfa_block_z = {"dsb": [], "hard": [], "anchor": []}
        for name in block_names:
            features = extracted[name].reshape(-1, extracted[name].shape[-1]).astype(np.float32)
            bank = banks[name]
            global_energy = pca_energy(features, bank.global_pca)
            hard_energy = mixture_energy(features, coords, bank, args.coord_weight, "hard")
            soft_energy = mixture_energy(features, coords, bank, args.coord_weight, "soft")
            anchor_energy = softmin_pair(global_energy, hard_energy, bank.router_temperature)
            proto_energy = proto_center_energy(features, coords, bank, args.coord_weight)
            dsb_z = robust_log_z(global_energy, bank.train_global_energy)
            hard_z = robust_log_z(hard_energy, bank.train_hard_energy)
            soft_z = robust_log_z(soft_energy, bank.train_soft_energy)
            anchor_z = robust_log_z(anchor_energy, bank.train_anchor_energy)
            proto_z = robust_log_z(proto_energy, bank.train_proto_center_energy)
            rel_weights = reliability_router_weights(
                features,
                coords,
                bank,
                args.coord_weight,
                args.support_rel_assignment_power,
            )
            relgate_z = reliability_routed_z(dsb_z, hard_z, rel_weights)
            relanchor_z = reliability_routed_z(dsb_z, anchor_z, rel_weights)
            block_z["dsb"].append(dsb_z)
            block_z["hard"].append(hard_z)
            block_z["soft"].append(soft_z)
            block_z["relgate"].append(relgate_z)
            block_z["relanchor"].append(relanchor_z)
            block_z["anchor"].append(anchor_z)
            block_z["proto_center"].append(proto_z)
            if args.enable_pseudo_router:
                distances = assignment_distances(features, coords, bank.centers, args.coord_weight)
                router_signal = router_signal_matrix(
                    dsb_z,
                    hard_z,
                    soft_z,
                    anchor_z,
                    distances,
                    coords,
                    bank,
                )
                alpha = pseudo_router_alpha(
                    bank.pseudo_router,
                    router_signal,
                    args.pseudo_router_alpha_power,
                )
                block_z["prouter_hard"].append(reliability_routed_z(dsb_z, hard_z, alpha))
                block_z["prouter_anchor"].append(reliability_routed_z(dsb_z, anchor_z, alpha))
                block_z["prouter_boost"].append(
                    (dsb_z + alpha * np.maximum(hard_z - dsb_z, 0.0)).astype(np.float32)
                )
            if args.enable_component_calibration:
                calhard_z = calibrated_mixture_z(features, coords, bank, args.coord_weight, "hard")
                calanchor_z = softmin_pair(dsb_z, calhard_z, args.cal_router_z_temp)
                block_z["calhard"].append(calhard_z)
                block_z["calanchor"].append(calanchor_z)
            if args.enable_feature_adapter:
                adapted_features = apply_feature_adapter_flat(features, feature_adapters[name])
                adapted_bank = adapted_banks[name]
                adapted_global_energy = pca_energy(adapted_features, adapted_bank.global_pca)
                adapted_hard_energy = mixture_energy(
                    adapted_features,
                    coords,
                    adapted_bank,
                    args.coord_weight,
                    "hard",
                )
                adapted_anchor_energy = softmin_pair(
                    adapted_global_energy,
                    adapted_hard_energy,
                    adapted_bank.router_temperature,
                )
                sfa_block_z["dsb"].append(
                    robust_log_z(adapted_global_energy, adapted_bank.train_global_energy)
                )
                sfa_block_z["hard"].append(
                    robust_log_z(adapted_hard_energy, adapted_bank.train_hard_energy)
                )
                sfa_block_z["anchor"].append(
                    robust_log_z(adapted_anchor_energy, adapted_bank.train_anchor_energy)
                )

        fused = {
            key: np.mean(np.stack(values, axis=0), axis=0)
            for key, values in block_z.items()
        }
        sfa_fused = {
            key: np.mean(np.stack(values, axis=0), axis=0)
            for key, values in sfa_block_z.items()
        } if args.enable_feature_adapter else {}
        n_tokens = grid_size[0] * grid_size[1]
        for i, path in enumerate(batch_paths):
            label = 0 if path.parent.name == "good" else 1
            labels.append(label)
            sl = slice(i * n_tokens, (i + 1) * n_tokens)
            fg_mask = image_foreground_mask(images[i], grid_size, args.image_res)

            dsb_map = fused["dsb"][sl]
            hard_map = fused["hard"][sl]
            soft_map = fused["soft"][sl]
            relgate_map = fused["relgate"][sl]
            relanchor_map = fused["relanchor"][sl]
            scores = {
                "dsb_fusion_full": image_score(
                    dsb_map, grid_size, args.image_res, args.top_frac
                ),
                "dsb_fusion_roi": image_score_masked(
                    dsb_map, fg_mask, grid_size, args.image_res, args.top_frac
                ),
                "mlsb_hard_full": image_score(
                    hard_map, grid_size, args.image_res, args.top_frac
                ),
                "mlsb_hard_roi": image_score_masked(
                    hard_map, fg_mask, grid_size, args.image_res, args.top_frac
                ),
                "mlsb_soft_full": image_score(
                    soft_map, grid_size, args.image_res, args.top_frac
                ),
                "mlsb_soft_roi": image_score_masked(
                    soft_map, fg_mask, grid_size, args.image_res, args.top_frac
                ),
                "mlsb_relgate_hard_full": image_score(
                    relgate_map, grid_size, args.image_res, args.top_frac
                ),
                "mlsb_relgate_hard_roi": image_score_masked(
                    relgate_map, fg_mask, grid_size, args.image_res, args.top_frac
                ),
                "mlsb_relanchor_softmin_full": image_score(
                    relanchor_map, grid_size, args.image_res, args.top_frac
                ),
                "mlsb_relanchor_softmin_roi": image_score_masked(
                    relanchor_map, fg_mask, grid_size, args.image_res, args.top_frac
                ),
            }
            scores["dsb_fr"] = 0.5 * (
                scores["dsb_fusion_full"] + scores["dsb_fusion_roi"]
            )
            scores["mlsb_hard_fr"] = 0.5 * (
                scores["mlsb_hard_full"] + scores["mlsb_hard_roi"]
            )
            scores["mlsb_soft_fr"] = 0.5 * (
                scores["mlsb_soft_full"] + scores["mlsb_soft_roi"]
            )
            scores["mlsb_relgate_hard_fr"] = 0.5 * (
                scores["mlsb_relgate_hard_full"] + scores["mlsb_relgate_hard_roi"]
            )
            scores["mlsb_relanchor_softmin_fr"] = 0.5 * (
                scores["mlsb_relanchor_softmin_full"]
                + scores["mlsb_relanchor_softmin_roi"]
            )
            scores["mlsb_anchor_softmin_full"] = image_score(
                fused["anchor"][sl], grid_size, args.image_res, args.top_frac
            )
            scores["mlsb_anchor_softmin_roi"] = image_score_masked(
                fused["anchor"][sl], fg_mask, grid_size, args.image_res, args.top_frac
            )
            scores["mlsb_anchor_softmin_fr"] = 0.5 * (
                scores["mlsb_anchor_softmin_full"] + scores["mlsb_anchor_softmin_roi"]
            )
            scores["mlsb_proto_center_full"] = image_score(
                fused["proto_center"][sl], grid_size, args.image_res, args.top_frac
            )
            scores["mlsb_proto_center_roi"] = image_score_masked(
                fused["proto_center"][sl], fg_mask, grid_size, args.image_res, args.top_frac
            )
            scores["mlsb_proto_center_fr"] = 0.5 * (
                scores["mlsb_proto_center_full"] + scores["mlsb_proto_center_roi"]
            )
            if args.enable_pseudo_router:
                prouter_hard_map = fused["prouter_hard"][sl]
                prouter_anchor_map = fused["prouter_anchor"][sl]
                prouter_boost_map = fused["prouter_boost"][sl]
                scores["mlsb_prouter_hard_full"] = image_score(
                    prouter_hard_map, grid_size, args.image_res, args.top_frac
                )
                scores["mlsb_prouter_hard_roi"] = image_score_masked(
                    prouter_hard_map, fg_mask, grid_size, args.image_res, args.top_frac
                )
                scores["mlsb_prouter_hard_fr"] = 0.5 * (
                    scores["mlsb_prouter_hard_full"] + scores["mlsb_prouter_hard_roi"]
                )
                scores["mlsb_prouter_anchor_full"] = image_score(
                    prouter_anchor_map, grid_size, args.image_res, args.top_frac
                )
                scores["mlsb_prouter_anchor_roi"] = image_score_masked(
                    prouter_anchor_map, fg_mask, grid_size, args.image_res, args.top_frac
                )
                scores["mlsb_prouter_anchor_fr"] = 0.5 * (
                    scores["mlsb_prouter_anchor_full"] + scores["mlsb_prouter_anchor_roi"]
                )
                scores["mlsb_prouter_boost_full"] = image_score(
                    prouter_boost_map, grid_size, args.image_res, args.top_frac
                )
                scores["mlsb_prouter_boost_roi"] = image_score_masked(
                    prouter_boost_map, fg_mask, grid_size, args.image_res, args.top_frac
                )
                scores["mlsb_prouter_boost_fr"] = 0.5 * (
                    scores["mlsb_prouter_boost_full"] + scores["mlsb_prouter_boost_roi"]
                )
            if args.enable_component_calibration:
                calhard_map = fused["calhard"][sl]
                scores["mlsb_calhard_full"] = image_score(
                    calhard_map, grid_size, args.image_res, args.top_frac
                )
                scores["mlsb_calhard_roi"] = image_score_masked(
                    calhard_map, fg_mask, grid_size, args.image_res, args.top_frac
                )
                scores["mlsb_calhard_fr"] = 0.5 * (
                    scores["mlsb_calhard_full"] + scores["mlsb_calhard_roi"]
                )
                scores["mlsb_calanchor_softmin_full"] = image_score(
                    fused["calanchor"][sl], grid_size, args.image_res, args.top_frac
                )
                scores["mlsb_calanchor_softmin_roi"] = image_score_masked(
                    fused["calanchor"][sl], fg_mask, grid_size, args.image_res, args.top_frac
                )
                scores["mlsb_calanchor_softmin_fr"] = 0.5 * (
                    scores["mlsb_calanchor_softmin_full"]
                    + scores["mlsb_calanchor_softmin_roi"]
                )
            gate = support_gate.trust
            scores["mlsb_valgate_hard_full"] = (
                (1.0 - gate) * scores["dsb_fusion_full"]
                + gate * scores["mlsb_hard_full"]
            )
            scores["mlsb_valgate_hard_roi"] = (
                (1.0 - gate) * scores["dsb_fusion_roi"]
                + gate * scores["mlsb_hard_roi"]
            )
            scores["mlsb_valgate_hard_fr"] = 0.5 * (
                scores["mlsb_valgate_hard_full"] + scores["mlsb_valgate_hard_roi"]
            )
            scores["mlsb_support_cal_hard_fr"] = support_calibrated_score(
                scores["mlsb_hard_fr"],
                support_gate,
                support_cal_scale,
            )
            blend_weight = args.support_cal_blend_weight
            scores["mlsb_support_cal_blend_fr"] = (
                (1.0 - blend_weight) * scores["dsb_fr"]
                + blend_weight * scores["mlsb_support_cal_hard_fr"]
            )
            if args.enable_pixel_eval:
                dsb_tokens = fused["dsb"][sl]
                hard_tokens = fused["hard"][sl]
                calibrated_hard_tokens = (
                    support_gate.dsb_median
                    + (hard_tokens - support_gate.hard_median) * support_cal_scale
                )
                sc_tokens = (
                    (1.0 - blend_weight) * dsb_tokens
                    + blend_weight * calibrated_hard_tokens
                )
                pixel_gt_maps.append(
                    ground_truth_mask(dataset_root, category, path, args.image_res)
                )
                pixel_maps["dsb_map"].append(
                    post_process_map(dsb_tokens.reshape(grid_size), args.image_res)
                )
                pixel_maps["mlsb_hard_map"].append(
                    post_process_map(hard_tokens.reshape(grid_size), args.image_res)
                )
                pixel_maps["sc_mlsb_map"].append(
                    post_process_map(sc_tokens.reshape(grid_size), args.image_res)
                )
                pixel_paths.append(str(path))
            if args.diagnose_refit_calibration:
                if refit_support_gate is None:
                    raise RuntimeError("Refit-bank calibration gate was not initialized.")
                scores["mlsb_refit_support_cal_hard_fr"] = support_calibrated_score(
                    scores["mlsb_hard_fr"],
                    refit_support_gate,
                    refit_support_cal_scale,
                )
                scores["mlsb_refit_support_cal_blend_fr"] = (
                    (1.0 - blend_weight) * scores["dsb_fr"]
                    + blend_weight * scores["mlsb_refit_support_cal_hard_fr"]
                )
            if args.enable_feature_adapter:
                if adapted_support_gate is None:
                    raise RuntimeError("Feature adapter gate was not initialized.")
                sfa_dsb_map = sfa_fused["dsb"][sl]
                sfa_hard_map = sfa_fused["hard"][sl]
                sfa_anchor_map = sfa_fused["anchor"][sl]
                scores["sfa_dsb_full"] = image_score(
                    sfa_dsb_map, grid_size, args.image_res, args.top_frac
                )
                scores["sfa_dsb_roi"] = image_score_masked(
                    sfa_dsb_map, fg_mask, grid_size, args.image_res, args.top_frac
                )
                scores["sfa_dsb_fr"] = 0.5 * (
                    scores["sfa_dsb_full"] + scores["sfa_dsb_roi"]
                )
                scores["sfa_hard_full"] = image_score(
                    sfa_hard_map, grid_size, args.image_res, args.top_frac
                )
                scores["sfa_hard_roi"] = image_score_masked(
                    sfa_hard_map, fg_mask, grid_size, args.image_res, args.top_frac
                )
                scores["sfa_hard_fr"] = 0.5 * (
                    scores["sfa_hard_full"] + scores["sfa_hard_roi"]
                )
                scores["sfa_anchor_softmin_full"] = image_score(
                    sfa_anchor_map, grid_size, args.image_res, args.top_frac
                )
                scores["sfa_anchor_softmin_roi"] = image_score_masked(
                    sfa_anchor_map, fg_mask, grid_size, args.image_res, args.top_frac
                )
                scores["sfa_anchor_softmin_fr"] = 0.5 * (
                    scores["sfa_anchor_softmin_full"] + scores["sfa_anchor_softmin_roi"]
                )
                scores["sfa_support_cal_hard_fr"] = support_calibrated_score(
                    scores["sfa_hard_fr"],
                    adapted_support_gate,
                    adapted_support_cal_scale,
                )
                scores["sfa_support_cal_blend_fr"] = (
                    (1.0 - blend_weight) * scores["sfa_dsb_fr"]
                    + blend_weight * scores["sfa_support_cal_hard_fr"]
                )
            if args.enable_selfval_selector:
                selected = selfval_selector.selected_method
                if selected not in scores:
                    selected = args.selfval_fallback_method
                scores["mlsb_selfval_select_fr"] = scores.get(
                    selected,
                    scores["mlsb_support_cal_blend_fr"],
                )

            row: dict[str, object] = {
                "seed": seed,
                "category": category,
                "sample": f"{path.parent.name}/{path.name}",
                "label": label,
                "fg_ratio": float(np.mean(fg_mask)),
                "n_components": args.n_components,
                "coord_weight": args.coord_weight,
                "support_valgate_trust": support_gate.trust,
                "support_valgate_reason": support_gate.reason,
                "support_valgate_n_fit": support_gate.n_fit,
                "support_valgate_n_val": support_gate.n_val,
                "support_valgate_dsb_median": support_gate.dsb_median,
                "support_valgate_hard_median": support_gate.hard_median,
                "support_valgate_dsb_q90": support_gate.dsb_q90,
                "support_valgate_hard_q90": support_gate.hard_q90,
                "support_valgate_median_ratio": support_gate.median_ratio,
                "support_valgate_q90_ratio": support_gate.q90_ratio,
                "support_cal_scale": support_cal_scale,
                "support_cal_blend_weight": args.support_cal_blend_weight,
                "support_rel_assignment_power": args.support_rel_assignment_power,
                "enable_feature_adapter": args.enable_feature_adapter,
            }
            if args.diagnose_refit_calibration:
                if refit_support_gate is None:
                    raise RuntimeError("Refit-bank calibration gate was not initialized.")
                row["refit_support_valgate_reason"] = refit_support_gate.reason
                row["refit_support_valgate_n_fit"] = refit_support_gate.n_fit
                row["refit_support_valgate_n_val"] = refit_support_gate.n_val
                row["refit_support_valgate_dsb_median"] = refit_support_gate.dsb_median
                row["refit_support_valgate_hard_median"] = refit_support_gate.hard_median
                row["refit_support_valgate_dsb_q90"] = refit_support_gate.dsb_q90
                row["refit_support_valgate_hard_q90"] = refit_support_gate.hard_q90
                row["refit_support_valgate_median_ratio"] = refit_support_gate.median_ratio
                row["refit_support_valgate_q90_ratio"] = refit_support_gate.q90_ratio
                row["refit_support_cal_scale"] = refit_support_cal_scale
            if args.enable_feature_adapter:
                if adapted_support_gate is None:
                    raise RuntimeError("Feature adapter gate was not initialized.")
                row["sfa_support_valgate_trust"] = adapted_support_gate.trust
                row["sfa_support_valgate_reason"] = adapted_support_gate.reason
                row["sfa_support_valgate_n_fit"] = adapted_support_gate.n_fit
                row["sfa_support_valgate_n_val"] = adapted_support_gate.n_val
                row["sfa_support_valgate_dsb_median"] = adapted_support_gate.dsb_median
                row["sfa_support_valgate_hard_median"] = adapted_support_gate.hard_median
                row["sfa_support_valgate_dsb_q90"] = adapted_support_gate.dsb_q90
                row["sfa_support_valgate_hard_q90"] = adapted_support_gate.hard_q90
                row["sfa_support_valgate_median_ratio"] = adapted_support_gate.median_ratio
                row["sfa_support_valgate_q90_ratio"] = adapted_support_gate.q90_ratio
                row["sfa_support_cal_scale"] = adapted_support_cal_scale
            if args.enable_selfval_selector:
                row["selfval_selected_method"] = selfval_selector.selected_method
                row["selfval_reason"] = selfval_selector.reason
                row["selfval_n_fit"] = selfval_selector.n_fit
                row["selfval_n_val"] = selfval_selector.n_val
                for method, auroc in selfval_selector.method_aurocs.items():
                    safe_method = method.replace("mlsb_", "").replace("_fr", "")
                    row[f"selfval_{safe_method}_auroc"] = auroc
                for method, margin in selfval_selector.method_margins.items():
                    safe_method = method.replace("mlsb_", "").replace("_fr", "")
                    row[f"selfval_{safe_method}_margin"] = margin
            if args.enable_pseudo_router:
                row["pseudo_router_alpha_power"] = args.pseudo_router_alpha_power
                row["pseudo_router_mix_min"] = args.pseudo_router_mix_min
                row["pseudo_router_mix_max"] = args.pseudo_router_mix_max
            if args.enable_component_calibration:
                row["cal_router_z_temp"] = args.cal_router_z_temp
            for block_name in block_names:
                bank = banks[block_name]
                row[f"{block_name}_actual_components"] = len(bank.component_sizes)
                row[f"{block_name}_min_component_size"] = min(bank.component_sizes)
                row[f"{block_name}_max_component_size"] = max(bank.component_sizes)
                row[f"{block_name}_assignment_temperature"] = bank.temperature
                row[f"{block_name}_router_temperature"] = bank.router_temperature
                row[f"{block_name}_support_rel_min"] = float(np.min(bank.component_reliability))
                row[f"{block_name}_support_rel_mean"] = float(np.mean(bank.component_reliability))
                row[f"{block_name}_support_rel_max"] = float(np.max(bank.component_reliability))
                row[f"{block_name}_proto_center_train_median"] = float(
                    np.median(bank.train_proto_center_energy)
                )
                row[f"{block_name}_proto_center_train_q90"] = float(
                    np.percentile(bank.train_proto_center_energy, 90)
                )
                if args.enable_feature_adapter:
                    adapter = feature_adapters[block_name]
                    adapted_bank = adapted_banks[block_name]
                    row[f"{block_name}_adapter_rank"] = adapter.rank
                    row[f"{block_name}_adapter_normal_energy_before"] = adapter.normal_energy_before
                    row[f"{block_name}_adapter_normal_energy_after"] = adapter.normal_energy_after
                    row[f"{block_name}_adapter_pseudo_energy_before"] = adapter.pseudo_energy_before
                    row[f"{block_name}_adapter_pseudo_energy_after"] = adapter.pseudo_energy_after
                    row[f"{block_name}_adapter_delta_rms"] = adapter.delta_rms
                    row[f"{block_name}_sfa_actual_components"] = len(adapted_bank.component_sizes)
                    row[f"{block_name}_sfa_min_component_size"] = min(adapted_bank.component_sizes)
                    row[f"{block_name}_sfa_max_component_size"] = max(adapted_bank.component_sizes)
                if args.enable_pseudo_router:
                    if bank.pseudo_router is None:
                        row[f"{block_name}_pseudo_router_normal_alpha_mean"] = float("nan")
                        row[f"{block_name}_pseudo_router_pseudo_alpha_mean"] = float("nan")
                    else:
                        row[f"{block_name}_pseudo_router_normal_alpha_mean"] = float(
                            bank.pseudo_router["normal_alpha_mean"][0]
                        )
                        row[f"{block_name}_pseudo_router_pseudo_alpha_mean"] = float(
                            bank.pseudo_router["pseudo_alpha_mean"][0]
                        )
            for method, score in scores.items():
                method_scores[method].append(score)
                row[method] = score
            detail_rows.append(row)
        iterator.update(len(batch_paths))

    labels_np = np.asarray(labels, dtype=np.uint8)
    rows: list[MetricRow] = []
    for method, scores in method_scores.items():
        auroc = (
            float(roc_auc_score(labels_np, np.asarray(scores, dtype=np.float64)))
            if np.unique(labels_np).size > 1
            else float("nan")
        )
        rows.append(MetricRow(seed, category, method, auroc, len(labels)))
    if args.enable_pixel_eval:
        pixel_rows = getattr(args, "_pixel_metric_rows", [])
        for method, pred_maps in pixel_maps.items():
            pixel_rows.append(
                pixel_metric_row(
                    seed,
                    category,
                    method,
                    pixel_gt_maps,
                    pred_maps,
                    args.device,
                )
            )
        args._pixel_metric_rows = pixel_rows
        if args.save_pixel_maps:
            map_dir = args.outdir / "pixel_maps"
            map_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                map_dir / f"{category}_seed{seed}.npz",
                paths=np.asarray(pixel_paths),
                gt=np.stack(pixel_gt_maps).astype(np.uint8),
                dsb=np.stack(pixel_maps["dsb_map"]).astype(np.float32),
                hard=np.stack(pixel_maps["mlsb_hard_map"]).astype(np.float32),
                sc_mlsb=np.stack(pixel_maps["sc_mlsb_map"]).astype(np.float32),
            )
    return rows, detail_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"E:\FSAD\datasets\mvtec_anomaly_detection"))
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(r"E:\FSAD\SubspaceAD_original\experiments\support_mixture_subspace_pilot"),
    )
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--model_ckpt", default="facebook/dinov2-with-registers-large")
    parser.add_argument("--blocks", nargs="+", default=DEFAULT_BLOCKS[:2])
    parser.add_argument("--image_res", type=int, default=256)
    parser.add_argument("--global_pca_ev", type=float, default=0.95)
    parser.add_argument("--local_pca_ev", type=float, default=0.90)
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--min_cluster_tokens", type=int, default=64)
    parser.add_argument("--min_ref_images_for_mixture", type=int, default=4)
    parser.add_argument("--coord_weight", type=float, default=0.25)
    parser.add_argument("--assignment_temp_scale", type=float, default=2.0)
    parser.add_argument("--router_temp_scale", type=float, default=1.0)
    parser.add_argument("--cal_router_z_temp", type=float, default=1.0)
    parser.add_argument("--enable_component_calibration", action="store_true")
    parser.add_argument("--support_rel_percentile", type=float, default=90.0)
    parser.add_argument("--support_rel_gain_scale", type=float, default=0.5)
    parser.add_argument("--support_rel_assignment_power", type=float, default=0.5)
    parser.add_argument("--enable_feature_adapter", action="store_true")
    parser.add_argument("--adapter_rank", type=int, default=16)
    parser.add_argument("--adapter_steps", type=int, default=120)
    parser.add_argument("--adapter_lr", type=float, default=3e-3)
    parser.add_argument("--adapter_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adapter_strength", type=float, default=0.25)
    parser.add_argument("--adapter_max_tokens", type=int, default=8192)
    parser.add_argument("--adapter_batch_tokens", type=int, default=2048)
    parser.add_argument("--adapter_pca_ev", type=float, default=0.95)
    parser.add_argument("--adapter_margin_percentile", type=float, default=90.0)
    parser.add_argument("--adapter_margin_scale", type=float, default=1.25)
    parser.add_argument("--adapter_loss_temp", type=float, default=0.25)
    parser.add_argument("--adapter_normal_weight", type=float, default=1.0)
    parser.add_argument("--adapter_pseudo_weight", type=float, default=1.0)
    parser.add_argument("--adapter_reg_weight", type=float, default=0.05)
    parser.add_argument("--adapter_grad_clip", type=float, default=1.0)
    parser.add_argument("--adapter_pseudo_mix_min", type=float, default=0.35)
    parser.add_argument("--adapter_pseudo_mix_max", type=float, default=1.0)
    parser.add_argument("--adapter_pseudo_noise_std", type=float, default=0.0)
    parser.add_argument("--adapter_donor_candidates", type=int, default=8)
    parser.add_argument("--adapter_pseudo_mode", choices=["extrapolate", "interpolate"], default="extrapolate")
    parser.add_argument("--adapter_enforce_pseudo_margin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_pseudo_router", action="store_true")
    parser.add_argument("--pseudo_router_max_tokens", type=int, default=4096)
    parser.add_argument("--pseudo_router_mix_min", type=float, default=0.35)
    parser.add_argument("--pseudo_router_mix_max", type=float, default=1.0)
    parser.add_argument("--pseudo_router_alpha_power", type=float, default=1.0)
    parser.add_argument("--pseudo_router_c", type=float, default=1.0)
    parser.add_argument("--pseudo_router_max_iter", type=int, default=200)
    parser.add_argument("--enable_selfval_selector", action="store_true")
    parser.add_argument(
        "--selfval_candidates",
        nargs="+",
        default=[
            "mlsb_support_cal_blend_fr",
            "mlsb_anchor_softmin_fr",
            "mlsb_prouter_boost_fr",
            "dsb_fr",
        ],
    )
    parser.add_argument("--selfval_fallback_method", default="mlsb_support_cal_blend_fr")
    parser.add_argument("--selfval_pseudo_frac", type=float, default=0.08)
    parser.add_argument("--selfval_min_candidate_auc", type=float, default=0.75)
    parser.add_argument("--selfval_min_auc_gain", type=float, default=0.0)
    parser.add_argument("--selfval_tie_eps", type=float, default=0.0)
    parser.add_argument("--selfval_margin_weight", type=float, default=0.0)
    parser.add_argument("--support_val_min_images", type=int, default=6)
    parser.add_argument("--support_val_stride", type=int, default=3)
    parser.add_argument("--support_val_median_ratio_max", type=float, default=1.10)
    parser.add_argument("--support_val_q90_ratio_max", type=float, default=1.08)
    parser.add_argument("--support_cal_scale_min", type=float, default=0.25)
    parser.add_argument("--support_cal_scale_max", type=float, default=4.0)
    parser.add_argument("--support_cal_blend_weight", type=float, default=0.5)
    parser.add_argument(
        "--diagnose_refit_calibration",
        action="store_true",
        help="Compare split-bank calibration with in-sample final-bank calibration.",
    )
    parser.add_argument("--kmeans_batch_size", type=int, default=2048)
    parser.add_argument("--kmeans_max_iter", type=int, default=100)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--aug_count", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--top_frac", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug_limit", type=int, default=None)
    parser.add_argument("--enable_pixel_eval", action="store_true")
    parser.add_argument("--save_pixel_maps", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    blocks = parse_blocks(args.blocks)
    set_seed(args.seeds[0])
    extractor = MultiLayerExtractor(args.model_ckpt, args.image_res, blocks, device)

    all_metrics: list[MetricRow] = []
    all_details: list[dict[str, object]] = []
    for seed in args.seeds:
        for category in args.categories:
            print(f"=== seed={seed} category={category} ===")
            metrics, details = evaluate_category(extractor, args.dataset, category, seed, args)
            all_metrics.extend(metrics)
            all_details.extend(details)
            write_outputs(args.outdir, all_metrics, all_details)
            if args.enable_pixel_eval:
                pixel_path = args.outdir / "pixel_metrics_seed_category.csv"
                pixel_path.parent.mkdir(parents=True, exist_ok=True)
                pixel_rows = getattr(args, "_pixel_metric_rows", [])
                with pixel_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(pixel_rows[0]))
                    writer.writeheader()
                    writer.writerows(pixel_rows)
                print(f"Wrote {pixel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
