"""Pilot for multi-layer subspace banks.

This standalone script evaluates whether separate PCA banks built from
different DINOv2 layer groups are complementary enough to beat strong
single-configuration baselines.

It does not modify SubspaceAD's main.py.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from subspacead.core.pca import PCAModel  # noqa: E402
from subspacead.data.transforms import get_augmentation_transform  # noqa: E402
from subspacead.post_process.scoring import post_process_map  # noqa: E402


NO_AUG_CATEGORIES = {"transistor"}
DEFAULT_CATEGORIES = ["cable", "capsule", "screw", "grid", "wood", "transistor"]
DEFAULT_BLOCKS = [
    "A_output=-1,-2,-3,-4,-5,-6",
    "B_upper_mid=-7,-8,-9,-10,-11,-12",
    "C_lower_mid=-13,-14,-15,-16,-17,-18",
    "D_input=-19,-20,-21,-22,-23,-24",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class BlockSpec:
    name: str
    layers: tuple[int, ...]


@dataclass
class MetricRow:
    seed: int
    category: str
    method: str
    image_auroc: float
    n_test: int


def parse_blocks(items: list[str]) -> list[BlockSpec]:
    blocks: list[BlockSpec] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Block must use name=layer,layer format: {item}")
        name, layers_s = item.split("=", 1)
        layers = tuple(int(x.strip()) for x in layers_s.split(",") if x.strip())
        if not name or not layers:
            raise ValueError(f"Invalid block spec: {item}")
        blocks.append(BlockSpec(name=name, layers=layers))
    return blocks


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_top_frac(values: np.ndarray, frac: float = 0.01) -> float:
    flat = values.reshape(-1)
    k = max(1, int(flat.size * frac))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def image_score(values: np.ndarray, grid_size: tuple[int, int], image_res: int, frac: float) -> float:
    score_map = values.reshape(grid_size)
    processed = post_process_map(score_map, image_res)
    return mean_top_frac(processed, frac)


def gray_foreground_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = cv2.bitwise_not(bright)

    def border_fraction(mask: np.ndarray) -> float:
        border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
        return float(np.mean(border > 0))

    chosen = min([bright, dark], key=border_fraction)
    ratio = float(np.mean(chosen > 0))
    if ratio < 0.02 or ratio > 0.98:
        return np.ones_like(gray, dtype=bool)
    return chosen > 0


def image_foreground_mask(
    image: Image.Image,
    grid_size: tuple[int, int],
    image_res: int,
) -> np.ndarray:
    gray = np.asarray(image.resize((image_res, image_res)).convert("L"))
    mask = gray_foreground_mask(gray)
    mask_grid = cv2.resize(
        mask.astype(np.uint8),
        (grid_size[1], grid_size[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    ratio = float(np.mean(mask_grid))
    if ratio < 0.02 or ratio > 0.98:
        mask_grid[:] = True
    return mask_grid.reshape(-1)


def image_score_masked(
    values: np.ndarray,
    mask: np.ndarray,
    grid_size: tuple[int, int],
    image_res: int,
    frac: float,
) -> float:
    score_map = values.reshape(grid_size)
    processed = post_process_map(score_map, image_res)
    mask_img = cv2.resize(
        mask.reshape(grid_size).astype(np.uint8),
        (image_res, image_res),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    selected = processed[mask_img]
    if selected.size == 0:
        return mean_top_frac(processed, frac)
    return mean_top_frac(selected, frac)


def local_texture_values(
    image: Image.Image,
    grid_size: tuple[int, int],
    image_res: int,
) -> np.ndarray:
    gray = np.asarray(image.resize((image_res, image_res)).convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    lap = np.abs(cv2.Laplacian(gray_eq, cv2.CV_32F, ksize=3))
    sx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sx * sx + sy * sy)
    texture = cv2.GaussianBlur(0.5 * lap + 0.5 * grad, (3, 3), 0)
    texture_grid = cv2.resize(
        texture,
        (grid_size[1], grid_size[0]),
        interpolation=cv2.INTER_AREA,
    )
    return texture_grid.reshape(-1).astype(np.float32)


def collect_texture_reference(
    images: list[Image.Image],
    grid_size: tuple[int, int],
    image_res: int,
) -> np.ndarray:
    return np.concatenate(
        [local_texture_values(image, grid_size, image_res) for image in images],
        axis=0,
    )


def texture_boost_map(
    fusion_values: np.ndarray,
    texture_z: np.ndarray,
    strength: float,
    clip: float,
) -> np.ndarray:
    clipped = np.minimum(np.maximum(texture_z, 0.0), max(float(clip), 0.0))
    return (fusion_values + float(strength) * clipped).astype(np.float32)


def robust_log_z(values: np.ndarray, normal_values: np.ndarray) -> np.ndarray:
    x = np.log1p(np.maximum(values, 0.0).astype(np.float64))
    ref = np.log1p(np.maximum(normal_values, 0.0).astype(np.float64))
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        q25, q75 = np.percentile(ref, [25, 75])
        scale = max(float((q75 - q25) / 1.349), 1e-12)
    return np.maximum((x - med) / scale, 0.0).astype(np.float32)


def support_reliability_weights(
    block_names: list[str],
    train_energy: dict[str, np.ndarray],
    grid_size: tuple[int, int],
    n_ref_images: int,
    image_res: int,
    top_frac: float,
    blend: float,
    power: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n_tokens = grid_size[0] * grid_size[1]
    if n_ref_images <= 0:
        raise ValueError("n_ref_images must be positive")

    penalties: list[float] = []
    debug: dict[str, float] = {}
    for name in block_names:
        values = train_energy[name]
        ref_scores = []
        for idx in range(n_ref_images):
            start = idx * n_tokens
            stop = start + n_tokens
            if stop > values.shape[0]:
                break
            z = robust_log_z(values[start:stop], values)
            ref_scores.append(image_score(z, grid_size, image_res, top_frac))
        if not ref_scores:
            ref_scores = [1.0]

        scores = np.asarray(ref_scores, dtype=np.float64)
        q25, q75 = np.percentile(scores, [25, 75])
        penalty = float(np.median(scores) + 0.5 * (q75 - q25))
        penalties.append(max(penalty, 1e-6))
        debug[f"support_penalty_{name}"] = penalty

    raw = np.power(1.0 / np.asarray(penalties, dtype=np.float64), power)
    raw = raw / max(float(raw.sum()), 1e-12)
    uniform = np.full_like(raw, 1.0 / len(raw))
    blend = min(max(float(blend), 0.0), 1.0)
    weights = (1.0 - blend) * uniform + blend * raw
    weights = weights / max(float(weights.sum()), 1e-12)

    for name, weight in zip(block_names, weights):
        debug[f"support_weight_{name}"] = float(weight)
    debug["support_selected_block"] = block_names[int(np.argmax(weights))]
    return weights.astype(np.float32), debug


def pca_energy(features: np.ndarray, pca_params: dict) -> np.ndarray:
    mu = np.asarray(pca_params["mu"], dtype=np.float32)
    comps = np.asarray(pca_params["components"], dtype=np.float32)
    k = int(pca_params["k"])
    c = comps[:, :k]
    x = features.astype(np.float32, copy=False)
    centered = x - mu
    projected = centered @ c
    recon = projected @ c.T + mu
    residual = x - recon
    return np.sum(residual * residual, axis=1).astype(np.float32)


def fit_pca(features: np.ndarray, pca_ev: float) -> dict:
    feature_dim = features.shape[1]

    def gen():
        yield features

    model = PCAModel(k=None, ev=pca_ev, whiten=False)
    return model.fit(gen, feature_dim, features.shape[0], 1)


class MultiLayerExtractor:
    def __init__(self, model_ckpt: str, image_res: int, blocks: list[BlockSpec], device: str):
        self.processor = AutoImageProcessor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).eval().to(device)
        self.image_res = image_res
        self.blocks = blocks
        self.device = device
        self.grid_size: tuple[int, int] | None = None

    def _spatial_from_seq(
        self,
        seq_tokens: torch.Tensor,
        drop_front: int,
        n_expected: int,
        h_p: int,
        w_p: int,
    ) -> torch.Tensor:
        tokens = seq_tokens[:, drop_front : drop_front + n_expected, :]
        return tokens.reshape(tokens.shape[0], h_p, w_p, tokens.shape[-1])

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        size = {"height": self.image_res, "width": self.image_res}
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            do_resize=True,
            size=size,
            do_center_crop=False,
            crop_size=size,
        ).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        cfg = self.model.config
        ps = cfg.patch_size
        num_reg = getattr(cfg, "num_register_tokens", 0)
        drop_front = 1 + num_reg
        h_p, w_p = self.image_res // ps, self.image_res // ps
        n_expected = h_p * w_p
        self.grid_size = (h_p, w_p)

        by_block: dict[str, np.ndarray] = {}
        for block in self.blocks:
            feats = []
            for layer in block.layers:
                try:
                    feats.append(
                        self._spatial_from_seq(
                            hidden_states[layer], drop_front, n_expected, h_p, w_p
                        )
                    )
                except IndexError as exc:
                    raise IndexError(
                        f"Layer {layer} is unavailable for this model "
                        f"(hidden_states={len(hidden_states)})."
                    ) from exc
            fused = torch.stack(feats, dim=0).mean(dim=0)
            by_block[block.name] = fused.detach().cpu().numpy().astype(np.float32)
        return by_block


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def train_paths(dataset_root: Path, category: str) -> list[Path]:
    return image_files(dataset_root / category / "train" / "good")


def test_paths(dataset_root: Path, category: str, debug_limit: int | None) -> list[Path]:
    test_root = dataset_root / category / "test"
    paths = sorted(
        path
        for path in test_root.glob("*/*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if debug_limit is None:
        return paths
    good = [p for p in paths if p.parent.name == "good"][:debug_limit]
    bad = [p for p in paths if p.parent.name != "good"][:debug_limit]
    return sorted(good + bad)


def reference_images(
    paths: list[Path],
    category: str,
    seed: int,
    k_shot: int,
    aug_count: int,
    image_res: int,
) -> list[Image.Image]:
    rng = random.Random(seed)
    shuffled = paths.copy()
    rng.shuffle(shuffled)
    selected = shuffled[: min(k_shot, len(shuffled))]

    aug_transform = None
    if aug_count > 0 and category not in NO_AUG_CATEGORIES:
        aug_transform = get_augmentation_transform(["rotate"], image_res)

    images: list[Image.Image] = []
    for path in selected:
        img = Image.open(path).convert("RGB")
        images.append(img)
        if aug_transform is not None:
            for _ in range(aug_count):
                images.append(aug_transform(img))
    return images


def collect_train_features(
    extractor: MultiLayerExtractor,
    images: list[Image.Image],
    batch_size: int,
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {b.name: [] for b in extractor.blocks}
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        extracted = extractor.extract(batch)
        for name, arr in extracted.items():
            parts[name].append(arr.reshape(-1, arr.shape[-1]))
    return {name: np.concatenate(chunks, axis=0) for name, chunks in parts.items()}


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

    train_features = collect_train_features(extractor, refs, args.batch_size)
    grid_size = extractor.grid_size
    if grid_size is None:
        raise RuntimeError("Extractor did not infer grid size.")

    pca_params: dict[str, dict] = {}
    train_energy: dict[str, np.ndarray] = {}
    for block in extractor.blocks:
        feats = train_features[block.name]
        pca_params[block.name] = fit_pca(feats, args.pca_ev)
        train_energy[block.name] = pca_energy(feats, pca_params[block.name])
    texture_reference = collect_texture_reference(refs, grid_size, args.image_res)

    block_names = [b.name for b in extractor.blocks]
    support_weights, support_debug = support_reliability_weights(
        block_names,
        train_energy,
        grid_size,
        len(refs),
        args.image_res,
        args.top_frac,
        args.support_weight_blend,
        args.support_weight_power,
    )
    selected_block_idx = int(np.argmax(support_weights))

    method_scores: dict[str, list[float]] = {b.name: [] for b in extractor.blocks}
    method_scores.update(
        {
            "fusion_z_mean": [],
            "fusion_z_max": [],
            "fusion_z_top2": [],
            "fusion_z_support_weighted": [],
            "fusion_z_support_select": [],
            "fusion_z_mean_fgscore": [],
            "fusion_z_max_fgscore": [],
            "fusion_z_top2_fgscore": [],
            "texture_only": [],
            "texture_only_fgscore": [],
            "fusion_z_mean_texscore": [],
            "fusion_z_mean_tex_fgscore": [],
        }
    )
    for block in extractor.blocks:
        method_scores[f"{block.name}_fgscore"] = []
    detail_rows: list[dict[str, object]] = []
    labels: list[int] = []

    paths = test_paths(dataset_root, category, args.debug_limit)
    iterator = tqdm(paths, desc=f"{category} seed={seed}", leave=False)
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        extracted = extractor.extract(images)
        for i, path in enumerate(batch_paths):
            fg_mask = image_foreground_mask(images[i], grid_size, args.image_res)
            texture_z = robust_log_z(
                local_texture_values(images[i], grid_size, args.image_res),
                texture_reference,
            )
            label = 0 if path.parent.name == "good" else 1
            labels.append(label)
            row: dict[str, object] = {
                "seed": seed,
                "category": category,
                "sample": f"{path.parent.name}/{path.name}",
                "label": label,
                "fg_ratio": float(np.mean(fg_mask)),
                **support_debug,
            }
            texture_score = image_score(texture_z, grid_size, args.image_res, args.top_frac)
            texture_fg_score = image_score_masked(
                texture_z, fg_mask, grid_size, args.image_res, args.top_frac
            )
            method_scores["texture_only"].append(texture_score)
            method_scores["texture_only_fgscore"].append(texture_fg_score)
            row["texture_only"] = texture_score
            row["texture_only_fgscore"] = texture_fg_score

            z_maps = []
            for block in extractor.blocks:
                tokens = extracted[block.name][i].reshape(-1, extracted[block.name].shape[-1])
                energy = pca_energy(tokens, pca_params[block.name])
                score = image_score(energy, grid_size, args.image_res, args.top_frac)
                method_scores[block.name].append(score)
                row[block.name] = score
                fg_score = image_score_masked(
                    energy, fg_mask, grid_size, args.image_res, args.top_frac
                )
                fg_method = f"{block.name}_fgscore"
                method_scores[fg_method].append(fg_score)
                row[fg_method] = fg_score
                z_maps.append(robust_log_z(energy, train_energy[block.name]))

            z_stack = np.stack(z_maps, axis=0)
            fusion_maps = {
                "fusion_z_mean": np.mean(z_stack, axis=0),
                "fusion_z_max": np.max(z_stack, axis=0),
                "fusion_z_top2": np.mean(
                    np.sort(z_stack, axis=0)[-min(2, z_stack.shape[0]) :], axis=0
                ),
                "fusion_z_support_weighted": np.average(
                    z_stack, axis=0, weights=support_weights
                ),
                "fusion_z_support_select": z_stack[selected_block_idx],
            }
            for method, fmap in fusion_maps.items():
                score = image_score(fmap, grid_size, args.image_res, args.top_frac)
                method_scores[method].append(score)
                row[method] = score
                fg_method = f"{method}_fgscore"
                if fg_method in method_scores:
                    fg_score = image_score_masked(
                        fmap, fg_mask, grid_size, args.image_res, args.top_frac
                    )
                    method_scores[fg_method].append(fg_score)
                    row[fg_method] = fg_score

            texture_boosted = texture_boost_map(
                fusion_maps["fusion_z_mean"],
                texture_z,
                args.texture_strength,
                args.texture_clip,
            )
            texture_boosted_score = image_score(
                texture_boosted, grid_size, args.image_res, args.top_frac
            )
            texture_boosted_fg_score = image_score_masked(
                texture_boosted, fg_mask, grid_size, args.image_res, args.top_frac
            )
            method_scores["fusion_z_mean_texscore"].append(texture_boosted_score)
            method_scores["fusion_z_mean_tex_fgscore"].append(texture_boosted_fg_score)
            row["fusion_z_mean_texscore"] = texture_boosted_score
            row["fusion_z_mean_tex_fgscore"] = texture_boosted_fg_score

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
    return rows, detail_rows


def write_outputs(
    outdir: Path,
    metrics: list[MetricRow],
    details: list[dict[str, object]],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = outdir / "metrics_by_seed_category.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "category", "method", "image_auroc", "n_test"]
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    "seed": row.seed,
                    "category": row.category,
                    "method": row.method,
                    "image_auroc": f"{row.image_auroc:.6f}",
                    "n_test": row.n_test,
                }
            )

    details_path = outdir / "image_scores.csv"
    if details:
        with details_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
            writer.writeheader()
            writer.writerows(details)

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in metrics:
        grouped.setdefault((row.method, row.category), []).append(row.image_auroc)

    summary = []
    for (method, category), vals in sorted(grouped.items()):
        arr = np.asarray(vals, dtype=np.float64)
        summary.append(
            {
                "method": method,
                "category": category,
                "n": len(vals),
                "mean_image_auroc": float(np.nanmean(arr)),
                "std_image_auroc": float(np.nanstd(arr, ddof=1)) if len(vals) > 1 else 0.0,
            }
        )

    by_method: dict[str, list[float]] = {}
    for row in summary:
        by_method.setdefault(row["method"], []).append(row["mean_image_auroc"])
    for method, vals in sorted(by_method.items()):
        arr = np.asarray(vals, dtype=np.float64)
        summary.append(
            {
                "method": method,
                "category": "Average",
                "n": len(vals),
                "mean_image_auroc": float(np.nanmean(arr)),
                "std_image_auroc": float(np.nanstd(arr, ddof=1)) if len(vals) > 1 else 0.0,
            }
        )

    summary_path = outdir / "summary_by_method_category.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "category",
                "n",
                "mean_image_auroc",
                "std_image_auroc",
            ],
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    **row,
                    "mean_image_auroc": f"{row['mean_image_auroc']:.6f}",
                    "std_image_auroc": f"{row['std_image_auroc']:.6f}",
                }
            )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {details_path}")
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-layer subspace bank pilot")
    parser.add_argument("--dataset", type=Path, default=Path(r"E:\FSAD\datasets\mvtec_anomaly_detection"))
    parser.add_argument("--outdir", type=Path, default=Path(r"E:\FSAD\multilayer_subspace_pilot"))
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--model_ckpt", default="facebook/dinov2-with-registers-large")
    parser.add_argument("--blocks", nargs="+", default=DEFAULT_BLOCKS)
    parser.add_argument("--image_res", type=int, default=256)
    parser.add_argument("--pca_ev", type=float, default=0.95)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--aug_count", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--top_frac", type=float, default=0.01)
    parser.add_argument("--support_weight_blend", type=float, default=0.5)
    parser.add_argument("--support_weight_power", type=float, default=1.0)
    parser.add_argument("--texture_strength", type=float, default=0.25)
    parser.add_argument("--texture_clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug_limit", type=int, default=None)
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
            metrics, details = evaluate_category(
                extractor, args.dataset, category, seed, args
            )
            all_metrics.extend(metrics)
            all_details.extend(details)
            write_outputs(args.outdir, all_metrics, all_details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
