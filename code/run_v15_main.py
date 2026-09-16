# -*- coding: utf-8 -*-
"""
V15 official-baseline benchmark built on the frozen V13 MC-PrivGLR protocol.

Research question
-----------------
When only a small fraction of federated clients drift, can the server use a
history of locally private monitoring losses to spend a fixed training budget
on the clients that still need adaptation?

Formal comparison set
---------------------
1. Random and Oracle are lower/upper controls.
2. Power-of-Choice-OGPM is the published loss-based rule with the current loss
   replaced by the common private OGPM report.
3. Oort-OGPM calls UCBsampler.py shipped by the official NeurIPS-2024 HiCS-FL
   repository, with the private report as its statistical utility.
4. HiCS-FL uses the official repository's clustering primitives and only stale
   final-layer deltas from already selected clients.  It is explicitly marked
   source-adapted because the V13 data/model loop is retained.
5. MC-PrivGLR-top2 is the frozen V13 proposal.  No V13 candidate search occurs
   in this formal benchmark.
6. FedGCS-OGPM imports the official IJCAI-2024 AUTOS encoder/predictor/decoder
   and adapts its record interface to the same private OGPM reports.

Privacy statement
-----------------
Each monitoring-window report separately satisfies event-level epsilon-LDP.
Repeated reports compose over time. This program does NOT claim that the whole
trajectory has total privacy cost epsilon.

Dependencies: Python >= 3.10, numpy, pandas, torch, torchvision, and the
official Optimal-GPM closed_form_mechanism.py in the same directory.
The file does not import V4/V5/V6 training modules.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18

from v15_official_baseline_adapters import FedGCSAdapter, HiCSAdapter, OortAdapter

try:
    # Official PETS'25 source file from:
    # https://github.com/ZhengYeah/Optimal-GPM
    from closed_form_mechanism import classical_mechanism_01 as official_classical_ogpm_01
except ImportError as exc:
    official_classical_ogpm_01 = None
    OFFICIAL_OGPM_IMPORT_ERROR = exc
else:
    OFFICIAL_OGPM_IMPORT_ERROR = None


VERSION = "V15_FedGCS_official_architecture_frozen_MCPrivGLR_v1"
ACTIVE_DATASET = "cifar10"
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def set_dataset_statistics(dataset: str) -> None:
    """Set the public image shape/normalization used by every policy."""
    global ACTIVE_DATASET, CIFAR_MEAN, CIFAR_STD
    ACTIVE_DATASET = str(dataset).lower()
    if ACTIVE_DATASET == "femnist":
        # A fixed public normalization; it is never fitted on a private client.
        mean = [0.5]
        std = [0.5]
    elif ACTIVE_DATASET == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    else:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    channels = len(mean)
    CIFAR_MEAN = torch.tensor(mean).view(1, channels, 1, 1)
    CIFAR_STD = torch.tensor(std).view(1, channels, 1, 1)


# =============================================================================
# Reproducibility and small utilities
# =============================================================================


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def amp_autocast(device: torch.device, enabled: bool):
    active = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.autocast(device_type=device.type, enabled=active)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=active)


def make_grad_scaler(device: torch.device, enabled: bool):
    active = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=active)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=active)


def parse_epsilon(token: str | float) -> float:
    text = str(token).strip().lower()
    if text in {"inf", "+inf", "infinity", "none", "nonprivate"}:
        return float("inf")
    value = float(text)
    if value <= 0:
        raise ValueError("Every finite epsilon must be positive")
    return value


def epsilon_label(epsilon: float) -> str:
    return "no_privacy" if np.isinf(epsilon) else f"eps_{epsilon:g}"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def stable_hash(payload: dict) -> str:
    blob = json.dumps(json_safe(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def safe_filename_token(value: str) -> str:
    """Return a deterministic Windows-safe token for cache filenames."""
    text = str(value)
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in text)


def file_sha256(path: str | Path) -> Optional[str]:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically so interrupted runs never leave a valid-looking partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def bootstrap_mean_ci(values: Sequence[float], runs: int, seed: int) -> Tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1 or runs <= 0:
        return float(x.mean()), float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    means = np.empty(runs, dtype=np.float64)
    for start in range(0, runs, 2048):
        count = min(2048, runs - start)
        draw = rng.integers(0, x.size, size=(count, x.size))
        means[start:start + count] = x[draw].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def top_b(scores: np.ndarray, budget: int, tie: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    tie = np.asarray(tie, dtype=np.float64)
    order = np.lexsort((tie, -np.nan_to_num(scores, nan=-np.inf)))
    return order[: min(int(budget), scores.size)].astype(np.int64)


def average_precision_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    tp = np.cumsum(ranked)
    precision = tp / (np.arange(y.size) + 1.0)
    return float((precision * ranked).sum() / positives)


def roc_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(y.size, dtype=np.float64)
    i = 0
    while i < y.size:
        j = i + 1
        while j < y.size and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# =============================================================================
# Exact continuous OGPM and its post-quantization channel
# =============================================================================


def require_official_ogpm() -> None:
    if official_classical_ogpm_01 is None:
        raise ImportError(
            "V13 requires the official Optimal-GPM file closed_form_mechanism.py "
            "in the same directory as this script."
        ) from OFFICIAL_OGPM_IMPORT_ERROR


def classical_ogpm_parameters(epsilon: float, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Call the supplied official classical_mechanism_01 for every input."""
    require_official_ogpm()
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    if np.isinf(epsilon):
        return x.copy(), x.copy(), np.full_like(x, np.inf), np.zeros_like(x)

    flat = x.reshape(-1)
    left = np.empty_like(flat)
    right = np.empty_like(flat)
    high = np.empty_like(flat)
    low = np.empty_like(flat)
    for j, value in enumerate(flat):
        densities, endpoints = official_classical_ogpm_01(float(epsilon), float(value))
        densities = np.asarray(densities, dtype=np.float64)
        endpoints = np.asarray(endpoints, dtype=np.float64)
        if densities.shape != (3,) or endpoints.shape != (4,):
            raise RuntimeError("Unexpected output from official classical_mechanism_01")
        low[j], high[j] = densities[0], densities[1]
        left[j], right[j] = endpoints[1], endpoints[2]
    shape = x.shape
    return left.reshape(shape), right.reshape(shape), high.reshape(shape), low.reshape(shape)


def ogpm_report_quantized(x: np.ndarray, epsilon: float, levels: int, uniforms: np.ndarray) -> np.ndarray:
    """Sample classical OGPM exactly, then post-process to an output-bin id."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    u = np.clip(np.asarray(uniforms, dtype=np.float64), 0.0, np.nextafter(1.0, 0.0))
    if x.shape != u.shape:
        raise ValueError("x and uniforms must have identical shapes")
    if np.isinf(epsilon):
        y = x
    else:
        left, right, high, low = classical_ogpm_parameters(epsilon, x)
        w1 = low * left
        w2 = high * (right - left)
        cut2 = w1 + w2
        y = np.empty_like(x)
        m1 = u < w1
        m2 = (~m1) & (u < cut2)
        m3 = ~(m1 | m2)
        if np.any(m1):
            y[m1] = u[m1] / low[m1]
        if np.any(m2):
            y[m2] = left[m2] + (u[m2] - w1[m2]) / high[m2]
        if np.any(m3):
            y[m3] = right[m3] + (u[m3] - cut2[m3]) / low[m3]
        y = np.clip(y, 0.0, 1.0)
    return np.minimum((y * int(levels)).astype(np.int64), int(levels) - 1)


def _overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def ogpm_discrete_channel(epsilon: float, levels: int, grid: np.ndarray) -> np.ndarray:
    """Q[g,y] = P(quantized OGPM output y | continuous input grid[g])."""
    grid = np.clip(np.asarray(grid, dtype=np.float64), 0.0, 1.0)
    q = np.zeros((grid.size, int(levels)), dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(levels) + 1)
    if np.isinf(epsilon):
        ids = np.minimum((grid * levels).astype(np.int64), levels - 1)
        q[np.arange(grid.size), ids] = 1.0
        return q
    left, right, high, low = classical_ogpm_parameters(epsilon, grid)
    for g in range(grid.size):
        for y in range(levels):
            a, b = float(edges[y]), float(edges[y + 1])
            q[g, y] = (
                low[g] * _overlap(a, b, 0.0, float(left[g]))
                + high[g] * _overlap(a, b, float(left[g]), float(right[g]))
                + low[g] * _overlap(a, b, float(right[g]), 1.0)
            )
    q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-15)
    return q


def validate_channel(epsilon: float, channel: np.ndarray, tolerance: float = 1e-9) -> dict:
    q = np.asarray(channel, dtype=np.float64)
    row_error = float(np.max(np.abs(q.sum(axis=1) - 1.0)))
    min_probability = float(q.min())
    if np.isinf(epsilon):
        ratio = float("inf")
        valid = row_error <= tolerance and min_probability >= -tolerance
    else:
        per_output_min = np.maximum(q.min(axis=0), 1e-300)
        ratio = float(np.max(q.max(axis=0) / per_output_min))
        valid = (
            row_error <= tolerance
            and min_probability >= -tolerance
            and ratio <= math.exp(epsilon) * (1.0 + 1e-8)
        )
    return {
        "epsilon": epsilon,
        "row_sum_max_error": row_error,
        "min_probability": min_probability,
        "max_likelihood_ratio": ratio,
        "exp_epsilon": float("inf") if np.isinf(epsilon) else math.exp(epsilon),
        "valid": bool(valid),
    }


# =============================================================================
# CIFAR data, sparse drift stream, and model
# =============================================================================


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float().div_(255.0)
        image = (image - CIFAR_MEAN[0]) / CIFAR_STD[0]
        return image, int(self.labels[idx])


class FEMNISTCNN(nn.Module):
    """Compact 28x28 grayscale network with a named final ``fc`` layer.

    GroupNorm avoids client-specific BatchNorm buffers and makes weighted
    FedAvg well defined even for small, non-IID writer batches.  The named
    classifier is also the exact interface consumed by the HiCS adapter.
    """

    def __init__(self, num_classes: int = 62) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.fc = nn.Linear(256, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embedding(self.features(x)))


def make_model(num_classes: int = 10) -> nn.Module:
    if ACTIVE_DATASET == "femnist":
        return FEMNISTCNN(num_classes=num_classes)
    try:
        model = resnet18(weights=None, num_classes=num_classes)
    except TypeError:  # torchvision < 0.13 compatibility
        model = resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_cifar(
    root: str,
    dataset: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    dataset_key = str(dataset).lower()
    if dataset_key == "cifar10":
        dataset_class, num_classes = CIFAR10, 10
    elif dataset_key == "cifar100":
        dataset_class, num_classes = CIFAR100, 100
    else:
        raise ValueError("--dataset must be cifar10 or cifar100")
    train = dataset_class(root=root, train=True, download=True)
    test = dataset_class(root=root, train=False, download=True)
    return (
        np.asarray(train.data, dtype=np.uint8),
        np.asarray(train.targets, dtype=np.int64),
        np.asarray(test.data, dtype=np.uint8),
        np.asarray(test.targets, dtype=np.int64),
        num_classes,
    )


def _find_femnist_parquet(root: Path, explicit_path: str = "") -> List[Path]:
    """Resolve a local FEMNIST parquet without requiring Hub connectivity."""
    roots: List[Path] = []
    if explicit_path:
        roots.append(Path(explicit_path))
    roots.extend([
        root / "femnist_hf",
        root / "femnist_processed",
        root / "hf_cache",
    ])
    files: List[Path] = []
    for candidate in roots:
        if candidate.is_file() and candidate.suffix.lower() == ".parquet":
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(candidate.rglob("*.parquet"))
    # The HF cache may contain duplicate snapshots of the same shard.
    unique = sorted({path.resolve() for path in files})
    return unique


def _open_femnist_dataset(root: str, explicit_path: str = ""):
    """Open FEMNIST from save_to_disk output or a cached local parquet shard."""
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "FEMNIST requires `pip install datasets pyarrow pillow`."
        ) from exc

    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend([Path(root) / "femnist_hf", Path(root) / "femnist_processed"])
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "dataset_dict.json").exists() or (candidate / "state.json").exists():
            loaded = load_from_disk(str(candidate))
            if isinstance(loaded, DatasetDict):
                return loaded["train"]
            return loaded

    parquet_files = _find_femnist_parquet(Path(root), explicit_path)
    if not parquet_files:
        raise FileNotFoundError(
            "No local FEMNIST dataset found. Expected a datasets.save_to_disk "
            "directory or a parquet shard below <root>/femnist_hf, "
            "<root>/femnist_processed, or <root>/hf_cache."
        )
    # One 814277-row shard is the expected flwrlabs/femnist artifact.  If the
    # cache contains duplicates, use the largest shard rather than concatenating.
    parquet_path = max(parquet_files, key=lambda path: path.stat().st_size)
    print(f"Loading local FEMNIST parquet: {parquet_path}")
    return load_dataset(
        "parquet", data_files={"train": str(parquet_path)}, split="train",
        cache_dir=str(Path(root) / "femnist_processed"),
    )


def _decode_femnist_rows(dataset, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode only the selected writers, keeping RAM far below the full corpus."""
    subset = dataset.select(np.asarray(rows, dtype=np.int64).tolist())
    images: List[np.ndarray] = []
    for image in subset["image"]:
        if hasattr(image, "convert"):
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        else:
            array = np.asarray(image, dtype=np.uint8)
            if array.ndim == 3:
                array = array[..., :3].mean(axis=2).astype(np.uint8)
        if array.shape != (28, 28):
            from PIL import Image
            array = np.asarray(
                Image.fromarray(array).resize((28, 28), resample=Image.BILINEAR),
                dtype=np.uint8,
            )
        images.append(array[..., None])
    labels = np.asarray(subset["character"], dtype=np.int64)
    return np.stack(images, axis=0), labels


def load_femnist_federated(args) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, int,
    List[np.ndarray], List[np.ndarray], np.ndarray,
]:
    """Build natural writer-clients and deterministic local train/test splits.

    Unlike CIFAR, FEMNIST already has a federated identity: ``writer_id``.
    Repartitioning it with a Dirichlet distribution would erase that structure,
    so exactly ``num_clients`` eligible writers are sampled once and every
    writer is split locally.  The resulting compact arrays are cached.
    """
    root = Path(args.root)
    cache_dir = root / "femnist_processed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"v15_writers_n{args.num_clients}_seed{args.data_seed}_"
        f"min{args.femnist_min_writer_samples}_test{args.femnist_test_fraction:.3f}.npz"
    )
    cache_path = cache_dir / cache_name
    if cache_path.exists() and not args.rebuild_femnist_cache:
        payload = np.load(cache_path, allow_pickle=False)
        train_x = payload["train_x"]
        train_y = payload["train_y"]
        test_x = payload["test_x"]
        test_y = payload["test_y"]
        train_offsets = payload["train_offsets"]
        test_offsets = payload["test_offsets"]
        train_clients = [
            np.arange(train_offsets[i], train_offsets[i + 1], dtype=np.int64)
            for i in range(args.num_clients)
        ]
        test_clients = [
            np.arange(test_offsets[i], test_offsets[i + 1], dtype=np.int64)
            for i in range(args.num_clients)
        ]
        print(f"Loaded FEMNIST writer cache: {cache_path}")
        return (
            train_x, train_y, test_x, test_y, 62,
            train_clients, test_clients,
            np.arange(train_y.size, dtype=np.int64),
        )

    dataset = _open_femnist_dataset(args.root, args.femnist_path)
    required = {"image", "writer_id", "character"}
    if not required.issubset(set(dataset.column_names)):
        raise ValueError(
            f"FEMNIST columns {dataset.column_names} do not include {sorted(required)}"
        )
    writers = np.asarray(dataset["writer_id"], dtype=str)
    unique_writers, inverse, counts = np.unique(
        writers, return_inverse=True, return_counts=True,
    )
    eligible = np.flatnonzero(counts >= int(args.femnist_min_writer_samples))
    if eligible.size < args.num_clients:
        raise ValueError(
            f"Only {eligible.size} FEMNIST writers have at least "
            f"{args.femnist_min_writer_samples} samples; need {args.num_clients}."
        )
    rng = np.random.default_rng(int(args.data_seed))
    chosen_codes = rng.choice(eligible, size=args.num_clients, replace=False)

    train_original: List[np.ndarray] = []
    test_original: List[np.ndarray] = []
    for code in chosen_codes.tolist():
        rows = np.flatnonzero(inverse == int(code))
        rng.shuffle(rows)
        n_test = max(10, int(round(rows.size * args.femnist_test_fraction)))
        n_test = min(n_test, rows.size - int(args.min_client_samples))
        if n_test < 1:
            raise ValueError("FEMNIST writer split leaves no held-out examples")
        test_original.append(rows[:n_test])
        train_original.append(rows[n_test:])

    train_lengths = np.asarray([len(x) for x in train_original], dtype=np.int64)
    test_lengths = np.asarray([len(x) for x in test_original], dtype=np.int64)
    train_offsets = np.concatenate([[0], np.cumsum(train_lengths)])
    test_offsets = np.concatenate([[0], np.cumsum(test_lengths)])
    train_rows = np.concatenate(train_original)
    test_rows = np.concatenate(test_original)
    train_x, train_y = _decode_femnist_rows(dataset, train_rows)
    test_x, test_y = _decode_femnist_rows(dataset, test_rows)
    train_clients = [
        np.arange(train_offsets[i], train_offsets[i + 1], dtype=np.int64)
        for i in range(args.num_clients)
    ]
    test_clients = [
        np.arange(test_offsets[i], test_offsets[i + 1], dtype=np.int64)
        for i in range(args.num_clients)
    ]
    np.savez_compressed(
        cache_path,
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
        train_offsets=train_offsets, test_offsets=test_offsets,
        writer_ids=unique_writers[chosen_codes],
    )
    print(
        f"Built FEMNIST natural-client cache: {cache_path} | "
        f"writers={args.num_clients}, train={train_y.size}, test={test_y.size}, "
        f"min/max train-client={train_lengths.min()}/{train_lengths.max()}"
    )
    return (
        train_x, train_y, test_x, test_y, 62,
        train_clients, test_clients,
        np.arange(train_y.size, dtype=np.int64),
    )


def stratified_public_split(labels: np.ndarray, public_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = int(public_size) // classes.size
    public: List[np.ndarray] = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls)
        public.append(rng.choice(idx, size=per_class, replace=False))
    public_idx = np.concatenate(public)
    mask = np.ones(labels.size, dtype=bool)
    mask[public_idx] = False
    return public_idx, np.flatnonzero(mask)


def dirichlet_partition(
    labels: np.ndarray,
    candidate_indices: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int,
    max_attempts: int = 200,
) -> List[np.ndarray]:
    labels = np.asarray(labels)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + 7919 * attempt)
        clients: List[List[int]] = [[] for _ in range(num_clients)]
        for cls in np.unique(labels[candidate_indices]):
            idx = candidate_indices[labels[candidate_indices] == cls].copy()
            rng.shuffle(idx)
            proportions = rng.dirichlet(np.full(num_clients, float(alpha)))
            cuts = (np.cumsum(proportions)[:-1] * idx.size).astype(int)
            for client, chunk in enumerate(np.split(idx, cuts)):
                clients[client].extend(chunk.tolist())
        arrays = [np.asarray(x, dtype=np.int64) for x in clients]
        if min(len(x) for x in arrays) >= int(min_size):
            return arrays
    sizes = [len(x) for x in arrays]
    raise RuntimeError(
        f"Could not build a Dirichlet partition with min_size={min_size}; "
        f"last min/max={min(sizes)}/{max(sizes)}. Lower --min_client_samples."
    )


def partition_test_by_train_priors(
    train_labels: np.ndarray,
    train_clients: List[np.ndarray],
    test_labels: np.ndarray,
    num_clients: int,
    seed: int,
    num_classes: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    priors = np.zeros((num_clients, int(num_classes)), dtype=np.float64)
    for i, idx in enumerate(train_clients):
        priors[i] = np.bincount(
            train_labels[idx], minlength=int(num_classes),
        ) + 0.5
    clients: List[List[int]] = [[] for _ in range(num_clients)]
    for cls in range(int(num_classes)):
        idx = np.flatnonzero(test_labels == cls).copy()
        rng.shuffle(idx)
        p = priors[:, cls] / priors[:, cls].sum()
        owners = rng.choice(num_clients, size=idx.size, p=p)
        for sample, owner in zip(idx.tolist(), owners.tolist()):
            clients[owner].append(sample)
    # A tiny test pool is harmless because evaluation samples with replacement.
    global_idx = np.arange(test_labels.size, dtype=np.int64)
    out = []
    for i, values in enumerate(clients):
        if values:
            out.append(np.asarray(values, dtype=np.int64))
        else:
            out.append(rng.choice(global_idx, size=10, replace=False))
    return out


def images_to_device(images_uint8: np.ndarray, device: torch.device, rotate_mask: Optional[np.ndarray] = None) -> torch.Tensor:
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    x.div_(255.0)
    if rotate_mask is not None and np.any(rotate_mask):
        mask = torch.as_tensor(rotate_mask, dtype=torch.bool, device=device)
        x[mask] = torch.rot90(x[mask], k=1, dims=(2, 3))
    mean = CIFAR_MEAN.to(device)
    std = CIFAR_STD.to(device)
    return (x - mean) / std


@dataclass
class StreamPlan:
    changed: np.ndarray
    onset: np.ndarray
    drift_probability: np.ndarray
    monitor_indices: np.ndarray
    train_indices: np.ndarray
    eval_indices: Dict[int, np.ndarray]
    monitor_rotate_u: np.ndarray
    train_rotate_u: np.ndarray
    eval_rotate_u: Dict[int, np.ndarray]
    ogpm_uniforms: np.ndarray
    ties: np.ndarray
    common_random: np.ndarray
    client_sizes: np.ndarray


def sample_client_indices(pools: List[np.ndarray], shape_tail: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    n = len(pools)
    result = np.empty((shape_tail[0], n, *shape_tail[1:]), dtype=np.int64)
    for t in range(shape_tail[0]):
        for i, pool in enumerate(pools):
            result[t, i] = rng.choice(pool, size=shape_tail[1:], replace=True)
    return result


def make_stream_plan(args, fraction: float, seed: int, train_clients: List[np.ndarray], test_clients: List[np.ndarray]) -> StreamPlan:
    rng = np.random.default_rng(seed * 1000003 + int(round(fraction * 10000)) * 97)
    n_changed = max(1, int(round(args.num_clients * float(fraction))))
    changed_ids = rng.choice(args.num_clients, size=n_changed, replace=False)
    changed = np.zeros(args.num_clients, dtype=bool)
    changed[changed_ids] = True
    onset = np.full(args.num_clients, args.horizon + 1, dtype=np.int64)
    onset[changed_ids] = args.change_time + rng.integers(0, max(1, args.async_window), size=n_changed)
    p = np.zeros((args.horizon, args.num_clients), dtype=np.float64)
    for t in range(args.horizon):
        age = t - onset
        if args.ramp_windows <= 1:
            p[t] = (age >= 0).astype(np.float64)
        else:
            p[t] = np.clip((age + 1) / float(args.ramp_windows), 0.0, 1.0)
        p[t, ~changed] = 0.0
    monitor_indices = sample_client_indices(train_clients, (args.horizon, args.window_batch), rng)
    train_indices = sample_client_indices(
        train_clients, (args.horizon, args.local_steps, args.local_batch), rng,
    )
    eval_times = list(range(0, args.horizon, args.eval_every))
    if (args.horizon - 1) not in eval_times:
        eval_times.append(args.horizon - 1)
    eval_indices = {}
    eval_rotate_u = {}
    for t in eval_times:
        eval_indices[t] = sample_client_indices(test_clients, (1, args.eval_client_batch), rng)[0]
        eval_rotate_u[t] = rng.random((args.num_clients, args.eval_client_batch))
    common_random = np.zeros((args.horizon, args.budget_b), dtype=np.int64)
    for t in range(args.horizon):
        common_random[t] = rng.choice(args.num_clients, size=args.budget_b, replace=False)
    return StreamPlan(
        changed=changed,
        onset=onset,
        drift_probability=p,
        monitor_indices=monitor_indices,
        train_indices=train_indices,
        eval_indices=eval_indices,
        monitor_rotate_u=rng.random((args.horizon, args.num_clients, args.window_batch)),
        train_rotate_u=rng.random(
            (args.horizon, args.num_clients, args.local_steps, args.local_batch)
        ),
        eval_rotate_u=eval_rotate_u,
        ogpm_uniforms=rng.random((args.horizon, args.num_clients)),
        ties=rng.random((args.horizon, args.num_clients)),
        common_random=common_random,
        client_sizes=np.asarray([len(pool) for pool in train_clients], dtype=np.float64),
    )


def client_batch(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    drift_probability: float,
    rotate_u: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotate = np.asarray(rotate_u) < float(drift_probability)
    x = images_to_device(images[indices], device, rotate)
    y = torch.as_tensor(labels[indices], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def monitor_all_clients(
    model: nn.Module,
    t: int,
    plan: StreamPlan,
    images: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    chunk_clients: int,
    amp: bool,
) -> np.ndarray:
    model.eval()
    n = plan.monitor_indices.shape[1]
    losses = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_clients):
        stop = min(n, start + chunk_clients)
        idx = plan.monitor_indices[t, start:stop].reshape(-1)
        probs = np.repeat(plan.drift_probability[t, start:stop], plan.monitor_indices.shape[2])
        rotate = plan.monitor_rotate_u[t, start:stop].reshape(-1) < probs
        x = images_to_device(images[idx], device, rotate)
        y = torch.as_tensor(labels[idx], dtype=torch.long, device=device)
        with amp_autocast(device, amp):
            per_sample = F.cross_entropy(model(x), y, reduction="none")
        losses[start:stop] = per_sample.view(stop - start, -1).mean(dim=1).cpu().numpy()
    return losses


def _freeze_batchnorm_running_stats(model: nn.Module) -> None:
    """Keep BN buffers common across clients while still training affine terms."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def aggregate_local_states(
    global_model: nn.Module,
    local_states: List[dict],
    weights: np.ndarray,
) -> None:
    """Standard weighted FedAvg over floating parameters and buffers."""
    if not local_states:
        return
    weights = np.asarray(weights, dtype=np.float64)
    weights /= np.maximum(weights.sum(), 1e-15)
    base = global_model.state_dict()
    averaged = {}
    for key, reference in base.items():
        if torch.is_floating_point(reference):
            value = torch.zeros_like(reference)
            for weight, state in zip(weights.tolist(), local_states):
                value.add_(state[key].to(reference.device), alpha=float(weight))
            averaged[key] = value
        else:
            # BN counters are not scientifically relevant here because running
            # statistics are frozen during local adaptation.
            averaged[key] = reference
    global_model.load_state_dict(averaged, strict=True)


def adapt_one_fedavg_round(
    model: nn.Module,
    selected: np.ndarray,
    t: int,
    plan: StreamPlan,
    images: np.ndarray,
    labels: np.ndarray,
    args,
    device: torch.device,
) -> dict:
    """
    Genuine synchronous FedAvg.

    Each selected client receives an independent copy of the current global
    model, performs ``local_steps`` optimizer steps on its own stream, and
    uploads one model delta.  No client optimizer state is shared.
    """
    selected = np.asarray(selected, dtype=np.int64)
    local_states: List[dict] = []
    final_losses: List[float] = []
    fc_weight_delta: List[np.ndarray] = []
    fc_bias_delta: List[np.ndarray] = []
    if args.fedavg_weighting == "samples":
        weights = plan.client_sizes[selected]
    else:
        weights = np.ones(selected.size, dtype=np.float64)

    global_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    for client in selected:
        local_model = make_model(args.num_classes).to(device)
        local_model.load_state_dict(global_state, strict=True)
        local_model.train()
        optimizer = torch.optim.SGD(
            local_model.parameters(), lr=args.adapt_lr,
            momentum=args.adapt_momentum,
            weight_decay=args.adapt_weight_decay,
        )
        scaler = make_grad_scaler(device, args.amp)
        for local_step in range(args.local_steps):
            idx = plan.train_indices[t, client, local_step]
            x, y = client_batch(
                images, labels, idx, plan.drift_probability[t, client],
                plan.train_rotate_u[t, client, local_step], device,
            )
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device, args.amp):
                loss = F.cross_entropy(local_model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        local_state = {
            key: value.detach().clone()
            for key, value in local_model.state_dict().items()
        }
        local_states.append(local_state)
        final_losses.append(float(loss.detach().cpu()))
        fc_weight_delta.append(
            (local_state["fc.weight"] - global_state["fc.weight"])
            .detach().cpu().numpy().astype(np.float64, copy=False)
        )
        fc_bias_delta.append(
            (local_state["fc.bias"] - global_state["fc.bias"])
            .detach().cpu().numpy().astype(np.float64, copy=False)
        )
        del local_model, optimizer, scaler

    aggregate_local_states(model, local_states, weights)
    del local_states, global_state
    return {
        "selected": selected.copy(),
        "final_losses": np.asarray(final_losses, dtype=np.float64),
        "fc_weight_delta": np.stack(fc_weight_delta, axis=0),
        "fc_bias_delta": np.stack(fc_bias_delta, axis=0),
    }


@torch.no_grad()
def evaluate_clients(
    model: nn.Module,
    t: int,
    plan: StreamPlan,
    images: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    chunk_clients: int,
    amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    indices = plan.eval_indices[t]
    n, batch = indices.shape
    losses = np.empty(n, dtype=np.float64)
    acc = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_clients):
        stop = min(n, start + chunk_clients)
        idx = indices[start:stop].reshape(-1)
        probs = np.repeat(plan.drift_probability[t, start:stop], batch)
        rotate = plan.eval_rotate_u[t][start:stop].reshape(-1) < probs
        x = images_to_device(images[idx], device, rotate)
        y = torch.as_tensor(labels[idx], dtype=torch.long, device=device)
        with amp_autocast(device, amp):
            logits = model(x)
            per_sample = F.cross_entropy(logits, y, reduction="none").view(stop - start, batch)
        correct = (logits.argmax(dim=1) == y).view(stop - start, batch)
        losses[start:stop] = per_sample.mean(dim=1).cpu().numpy()
        acc[start:stop] = correct.float().mean(dim=1).cpu().numpy()
    return losses, acc


# =============================================================================
# Channel-aware observation model and online controllers
# =============================================================================


def gaussian_grid_weights(grid: np.ndarray, center: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-4)
    z = -0.5 * ((grid - float(center)) / sigma) ** 2
    z -= z.max()
    w = np.exp(z)
    return w / np.maximum(w.sum(), 1e-15)


@dataclass
class EmissionModel:
    baseline_x: np.ndarray
    q0: np.ndarray
    q1: np.ndarray
    stable_x_numerator: np.ndarray
    drift_x_numerator: np.ndarray


@dataclass
class PersistentShiftEmission:
    """Client-specific OGPM emissions for one stable and M persistent states."""

    baseline_x: np.ndarray
    q0: np.ndarray
    q_shift: np.ndarray
    shifts: np.ndarray


def fit_emission_model(
    calibration_reports: np.ndarray,
    channel: np.ndarray,
    grid: np.ndarray,
    shrinkage: float,
    baseline_sigma: float,
    drift_shifts: Sequence[float],
    emission_floor: float,
) -> EmissionModel:
    """
    Fit a regularized point baseline from private reports, then build stable
    and positive-shift mixture emissions through the exact OGPM channel.

    This deliberately avoids V6's ill-conditioned full channel inversion.
    """
    reports = np.asarray(calibration_reports, dtype=np.int64)
    _, n = reports.shape
    levels = channel.shape[1]
    counts = np.zeros((n, levels), dtype=np.float64)
    for client in range(n):
        counts[client] = np.bincount(reports[:, client], minlength=levels)
    global_frequency = counts.sum(axis=0)
    global_frequency /= np.maximum(global_frequency.sum(), 1.0)
    regularized_counts = counts + float(shrinkage) * global_frequency[None, :]
    safe_channel = np.maximum(channel, float(emission_floor))
    safe_channel /= safe_channel.sum(axis=1, keepdims=True)
    log_likelihood = regularized_counts @ np.log(safe_channel).T
    baseline_index = np.argmax(log_likelihood, axis=1)
    baseline_x = grid[baseline_index]

    q0 = np.empty((n, levels), dtype=np.float64)
    q1 = np.empty_like(q0)
    stable_num = np.empty_like(q0)
    drift_num = np.empty_like(q0)
    shifts = np.asarray(list(drift_shifts), dtype=np.float64)
    if shifts.size == 0 or np.any(shifts <= 0):
        raise ValueError("--drift_shifts must contain positive values")

    for client, center in enumerate(baseline_x):
        w0 = gaussian_grid_weights(grid, center, baseline_sigma)
        w1 = np.zeros_like(grid)
        for delta in shifts:
            w1 += gaussian_grid_weights(grid, min(1.0, center + delta), baseline_sigma)
        w1 /= shifts.size
        q0[client] = w0 @ safe_channel
        q1[client] = w1 @ safe_channel
        stable_num[client] = (w0 * grid) @ safe_channel
        drift_num[client] = (w1 * grid) @ safe_channel

    q0 = np.maximum(q0, emission_floor)
    q1 = np.maximum(q1, emission_floor)
    q0 /= q0.sum(axis=1, keepdims=True)
    q1 /= q1.sum(axis=1, keepdims=True)
    return EmissionModel(baseline_x, q0, q1, stable_num, drift_num)


def fit_persistent_shift_emission(
    calibration_reports: np.ndarray,
    channel: np.ndarray,
    grid: np.ndarray,
    shrinkage: float,
    baseline_sigma: float,
    shifts: Sequence[float],
    emission_floor: float,
) -> PersistentShiftEmission:
    """Build trajectory-consistent shift hypotheses through the OGPM channel.

    A single shift index is persistent along a client's trajectory.  This is
    different from first averaging the shift emissions and then multiplying
    that per-window mixture over time.
    """
    reports = np.asarray(calibration_reports, dtype=np.int64)
    if reports.ndim != 2:
        raise ValueError("calibration_reports must have shape [time, clients]")
    _, n = reports.shape
    levels = channel.shape[1]
    shift_grid = np.asarray(list(shifts), dtype=np.float64)
    if shift_grid.size == 0 or np.any(shift_grid <= 0) or np.any(shift_grid >= 1):
        raise ValueError("--pspa_shifts must contain values strictly in (0,1)")
    if np.any(np.diff(shift_grid) <= 0):
        raise ValueError("--pspa_shifts must be strictly increasing")

    counts = np.zeros((n, levels), dtype=np.float64)
    for client in range(n):
        counts[client] = np.bincount(reports[:, client], minlength=levels)
    global_frequency = counts.sum(axis=0)
    global_frequency /= np.maximum(global_frequency.sum(), 1.0)
    regularized_counts = counts + float(shrinkage) * global_frequency[None, :]
    safe_channel = np.maximum(channel, float(emission_floor))
    safe_channel /= safe_channel.sum(axis=1, keepdims=True)
    log_likelihood = regularized_counts @ np.log(safe_channel).T
    baseline_x = grid[np.argmax(log_likelihood, axis=1)]

    q0 = np.empty((n, levels), dtype=np.float64)
    q_shift = np.empty((n, shift_grid.size, levels), dtype=np.float64)
    for client, center in enumerate(baseline_x):
        stable_weights = gaussian_grid_weights(grid, center, baseline_sigma)
        q0[client] = stable_weights @ safe_channel
        for shift_index, delta in enumerate(shift_grid):
            shifted_weights = gaussian_grid_weights(
                grid, min(1.0, center + float(delta)), baseline_sigma,
            )
            q_shift[client, shift_index] = shifted_weights @ safe_channel

    q0 = np.maximum(q0, emission_floor)
    q_shift = np.maximum(q_shift, emission_floor)
    q0 /= q0.sum(axis=1, keepdims=True)
    q_shift /= q_shift.sum(axis=2, keepdims=True)
    return PersistentShiftEmission(baseline_x, q0, q_shift, shift_grid)


@dataclass
class OnlineController:
    kind: str
    epsilon: Optional[float]
    args: argparse.Namespace
    grid: np.ndarray
    channel: Optional[np.ndarray]
    calibration_reports: List[np.ndarray] = field(default_factory=list)
    emission: Optional[EmissionModel] = None
    belief: Optional[np.ndarray] = None
    severity: Optional[np.ndarray] = None
    cusum: Optional[np.ndarray] = None
    age: Optional[np.ndarray] = None
    previous_selected: Optional[np.ndarray] = None
    latent_posterior: Optional[np.ndarray] = None
    latent_transition: Optional[np.ndarray] = None
    upward_prior: Optional[np.ndarray] = None
    baseline_x: Optional[np.ndarray] = None
    need: Optional[np.ndarray] = None
    previous_need: Optional[np.ndarray] = None
    response_alpha: Optional[np.ndarray] = None
    response_beta: Optional[np.ndarray] = None
    response_mean: Optional[np.ndarray] = None
    marginal_value: Optional[np.ndarray] = None
    tail_probability: Optional[np.ndarray] = None
    counterfactual_no_action_need: Optional[np.ndarray] = None
    counterfactual_action_need: Optional[np.ndarray] = None
    pooled_response_alpha: float = 0.0
    pooled_response_beta: float = 0.0
    passive_improvement: float = 0.0
    pspa_emission: Optional[PersistentShiftEmission] = None
    pspa_posterior: Optional[np.ndarray] = None
    pspa_expected_shift: Optional[np.ndarray] = None
    pspa_change_probability: Optional[np.ndarray] = None
    mure_evidence: Optional[np.ndarray] = None
    mure_standardized: Optional[np.ndarray] = None
    mure_score: Optional[np.ndarray] = None
    mure_null_variance: Optional[np.ndarray] = None
    mcglr_emission: Optional[PersistentShiftEmission] = None
    mcglr_charts: Optional[np.ndarray] = None
    mcglr_null_variance: Optional[np.ndarray] = None
    mcglr_score: Optional[np.ndarray] = None
    published_adapter: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        n = self.args.num_clients
        self.belief = np.full(n, self.args.belief_prior, dtype=np.float64)
        self.severity = np.zeros(n, dtype=np.float64)
        self.cusum = np.zeros(n, dtype=np.float64)
        self.age = np.zeros(n, dtype=np.float64)
        self.previous_selected = np.zeros(n, dtype=bool)
        self.need = np.zeros(n, dtype=np.float64)
        self.previous_need = np.zeros(n, dtype=np.float64)
        prior_strength = float(self.args.response_prior_strength)
        prior_mean = float(self.args.response_prior_mean)
        self.response_alpha = np.full(n, prior_strength * prior_mean, dtype=np.float64)
        self.response_beta = np.full(n, prior_strength * (1.0 - prior_mean), dtype=np.float64)
        self.response_mean = np.full(n, prior_mean, dtype=np.float64)
        self.pooled_response_alpha = prior_strength * prior_mean
        self.pooled_response_beta = prior_strength * (1.0 - prior_mean)
        self.marginal_value = np.zeros(n, dtype=np.float64)
        self.tail_probability = np.zeros(n, dtype=np.float64)
        self.counterfactual_no_action_need = np.zeros(n, dtype=np.float64)
        self.counterfactual_action_need = np.zeros(n, dtype=np.float64)
        self.pspa_expected_shift = np.zeros(n, dtype=np.float64)
        self.pspa_change_probability = np.zeros(n, dtype=np.float64)
        scales = len(self.args.mure_decays)
        self.mure_evidence = np.zeros((n, scales), dtype=np.float64)
        self.mure_standardized = np.zeros((n, scales), dtype=np.float64)
        self.mure_score = np.zeros(n, dtype=np.float64)
        self.mure_null_variance = np.ones(n, dtype=np.float64)
        self.mcglr_score = np.zeros(n, dtype=np.float64)

    def initialize_published_baseline(
        self,
        client_sizes: np.ndarray,
        stream_seed: int,
        device: torch.device,
    ) -> None:
        """Create a source-backed selector only after stream metadata exists."""
        if self.kind == "oort_private":
            self.published_adapter = OortAdapter(
                self.args.official_hics_root,
                client_sizes,
                int(stream_seed) + 17011,
            )
        elif self.kind == "hics2024":
            self.published_adapter = HiCSAdapter(
                self.args.official_hics_root,
                client_sizes,
                self.args.num_classes,
                self.args.fc_features,
                self.args.budget_b,
                int(stream_seed) + 19001,
                temperature=self.args.hics_temperature,
                lambda_entropy=self.args.hics_lambda,
                gamma=self.args.hics_gamma,
                num_groups=self.args.hics_groups,
                horizon=self.args.horizon - self.args.calib_end,
            )
        elif self.kind == "fedgcs_private":
            self.published_adapter = FedGCSAdapter(
                self.args.official_fedgcs_root,
                self.args.official_fedgcs_commit,
                self.args.num_clients,
                self.args.budget_b,
                self.args.report_levels,
                int(stream_seed) + 23003,
                device,
                candidates_per_round=self.args.fedgcs_candidates_per_round,
                replay_size=self.args.fedgcs_replay_size,
                train_every=self.args.fedgcs_train_every,
                train_epochs=self.args.fedgcs_train_epochs,
                batch_size=self.args.fedgcs_batch_size,
                learning_rate=self.args.fedgcs_lr,
                trade_off=self.args.fedgcs_trade_off,
                hidden_size=self.args.fedgcs_hidden_size,
                mlp_hidden_size=self.args.fedgcs_mlp_hidden_size,
                top_k=self.args.fedgcs_top_k,
                gradient_steps=self.args.fedgcs_gradient_steps,
            )

    def observe_training_feedback(
        self,
        t: int,
        selected: np.ndarray,
        reports: Optional[np.ndarray],
        feedback: dict,
    ) -> None:
        """Update published selectors using only allowed post-round feedback."""
        if self.kind == "oort_private":
            if self.published_adapter is None or reports is None:
                raise RuntimeError("Oort-OGPM requires an initialized adapter and reports")
            self.published_adapter.observe(
                t, selected, reports, self.args.report_levels,
            )
        elif self.kind == "hics2024":
            if self.published_adapter is None:
                raise RuntimeError("HiCS-FL adapter was not initialized")
            self.published_adapter.observe(
                selected,
                feedback["fc_weight_delta"],
                feedback["fc_bias_delta"],
            )

    def record_calibration(self, reports: np.ndarray) -> None:
        if (
            self.kind in {"topb", "cusum", "belief", "poc_private", "oort_private", "fedgcs_private"}
            or self.kind.startswith("privmav")
            or self.kind.startswith("pspa")
            or self.kind.startswith("mure")
            or self.kind.startswith("caps_")
            or self.kind.startswith("mcglr_")
        ):
            self.calibration_reports.append(np.asarray(reports, dtype=np.int64).copy())

    def ensure_calibrated(self) -> None:
        needs_binary = (
            self.kind in {"cusum", "belief"}
            or self.kind.startswith("mure")
            or self.kind.startswith("caps_")
        )
        needs_continuous = self.kind.startswith("privmav")
        needs_pspa = self.kind.startswith("pspa")
        needs_mcglr = self.kind.startswith("mcglr_")
        if not needs_binary and not needs_continuous and not needs_pspa and not needs_mcglr:
            return
        if needs_binary and self.emission is not None:
            return
        if needs_continuous and self.latent_posterior is not None:
            return
        if needs_pspa and self.pspa_posterior is not None:
            return
        if needs_mcglr and self.mcglr_emission is not None:
            return
        if len(self.calibration_reports) < self.args.calib_end:
            raise RuntimeError(
                f"Only {len(self.calibration_reports)} calibration reports; "
                f"need calib_end={self.args.calib_end}"
            )
        reports = np.stack(self.calibration_reports[:self.args.calib_end])
        if needs_mcglr:
            self.mcglr_emission = fit_persistent_shift_emission(
                reports,
                self.channel,
                self.grid,
                self.args.baseline_shrinkage,
                self.args.baseline_sigma,
                self.args.mcglr_shifts,
                self.args.emission_floor,
            )
            n = reports.shape[1]
            m = self.mcglr_emission.shifts.size
            self.mcglr_charts = np.zeros((n, m), dtype=np.float64)
            llr_table = (
                np.log(self.mcglr_emission.q_shift)
                - np.log(self.mcglr_emission.q0[:, None, :])
            )
            null_mean = np.sum(
                self.mcglr_emission.q0[:, None, :] * llr_table,
                axis=2,
            )
            null_var = np.sum(
                self.mcglr_emission.q0[:, None, :]
                * (llr_table - null_mean[:, :, None]) ** 2,
                axis=2,
            )
            self.mcglr_null_variance = np.maximum(null_var, 1e-8)
            return
        if needs_pspa:
            self.pspa_emission = fit_persistent_shift_emission(
                reports,
                self.channel,
                self.grid,
                self.args.baseline_shrinkage,
                self.args.baseline_sigma,
                self.args.pspa_shifts,
                self.args.emission_floor,
            )
            n = reports.shape[1]
            m = len(self.args.pspa_shifts)
            prior = float(self.args.pspa_prior)
            posterior = np.empty((n, m + 1), dtype=np.float64)
            posterior[:, 0] = 1.0 - prior
            posterior[:, 1:] = prior / m
            self.pspa_posterior = posterior
            self.pspa_expected_shift[:] = (
                posterior[:, 1:] @ self.pspa_emission.shifts
            )
            self.pspa_change_probability[:] = posterior[:, 1:].sum(axis=1)
            return
        if needs_binary:
            # This fixed positive-shift model is retained only for historical
            # likelihood baselines; PrivMAV does not use q1.
            self.emission = fit_emission_model(
                reports, self.channel, self.grid,
                self.args.baseline_shrinkage,
                self.args.baseline_sigma,
                self.args.drift_shifts,
                self.args.emission_floor,
            )
            if self.kind.startswith("mure"):
                llr = np.log(self.emission.q1) - np.log(self.emission.q0)
                null_mean = np.sum(self.emission.q0 * llr, axis=1)
                null_var = np.sum(
                    self.emission.q0 * (llr - null_mean[:, None]) ** 2,
                    axis=1,
                )
                self.mure_null_variance = np.maximum(null_var, 1e-8)
            return

        n = reports.shape[1]
        counts = np.zeros((n, self.channel.shape[1]), dtype=np.float64)
        for client in range(n):
            counts[client] = np.bincount(
                reports[:, client], minlength=self.channel.shape[1],
            )
        global_frequency = counts.sum(axis=0)
        global_frequency /= np.maximum(global_frequency.sum(), 1.0)
        regularized = counts + self.args.baseline_shrinkage * global_frequency[None, :]
        safe_channel = np.maximum(self.channel, self.args.emission_floor)
        logp = regularized @ np.log(safe_channel).T
        logp -= logp.max(axis=1, keepdims=True)
        posterior = np.exp(logp)
        posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-15)
        self.latent_posterior = posterior
        self.baseline_x = posterior @ self.grid
        self.need = np.sum(
            posterior * np.maximum(
                self.grid[None, :] - self.baseline_x[:, None]
                - self.args.need_margin,
                0.0,
            ),
            axis=1,
        )
        threshold = self.baseline_x[:, None] + self.args.need_margin
        self.tail_probability = np.sum(
            posterior * (self.grid[None, :] > threshold),
            axis=1,
        )
        self.previous_need = self.need.copy()

        sigma = max(float(self.args.latent_process_sigma), 1e-4)
        transition = np.exp(
            -0.5 * ((self.grid[:, None] - self.grid[None, :]) / sigma) ** 2
        )
        transition /= np.maximum(transition.sum(axis=1, keepdims=True), 1e-15)
        self.latent_transition = transition
        upward = np.zeros_like(posterior)
        for client, baseline in enumerate(self.baseline_x):
            mask = self.grid >= baseline + self.args.need_margin
            if not np.any(mask):
                mask[-1] = True
            upward[client, mask] = 1.0
            upward[client] /= upward[client].sum()
        self.upward_prior = upward

    def update_pspa_posterior(self, reports: np.ndarray) -> None:
        """One exact Bayes update for stable and persistent-shift states."""
        self.ensure_calibrated()
        if self.pspa_emission is None or self.pspa_posterior is None:
            raise RuntimeError("PSPA emission model was not initialized")
        m = self.pspa_emission.shifts.size
        persistence = float(self.args.pspa_persistence)
        hazard = float(self.args.pspa_hazard)

        transition = np.zeros((m + 1, m + 1), dtype=np.float64)
        transition[0, 0] = 1.0 - hazard
        transition[0, 1:] = hazard / m
        for state in range(1, m + 1):
            transition[state, state] = persistence
            transition[state, 0] = 1.0 - persistence

        predicted = self.pspa_posterior @ transition
        y = np.asarray(reports, dtype=np.int64)
        clients = np.arange(self.args.num_clients)
        likelihood = np.empty_like(predicted)
        likelihood[:, 0] = self.pspa_emission.q0[clients, y]
        for state in range(m):
            likelihood[:, state + 1] = self.pspa_emission.q_shift[
                clients, state, y,
            ]
        posterior = predicted * np.maximum(likelihood, self.args.emission_floor)
        posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-15)
        self.pspa_posterior = posterior
        self.pspa_change_probability = posterior[:, 1:].sum(axis=1)
        self.pspa_expected_shift = posterior[:, 1:] @ self.pspa_emission.shifts

    def update_mure_evidence(self, reports: np.ndarray) -> np.ndarray:
        """Update our multi-timescale residual evidence from one OGPM report.

        All memories receive the same exact channel log-likelihood innovation.
        They differ only in their public forgetting factors.  The innovation is
        centered under each client's fitted stable emission and standardized
        by its exact null variance, avoiding a systematic preference for noisy
        or intrinsically high-loss clients.
        """
        self.ensure_calibrated()
        if self.emission is None:
            raise RuntimeError("MURE emission model was not initialized")
        y = np.asarray(reports, dtype=np.int64)
        idx = np.arange(self.args.num_clients)
        llr_table = np.log(self.emission.q1) - np.log(self.emission.q0)
        null_mean = np.sum(self.emission.q0 * llr_table, axis=1)
        innovation = (
            llr_table[idx, y]
            - null_mean
            - float(self.args.mure_drift_margin)
        )

        decays = np.asarray(self.args.mure_decays, dtype=np.float64)
        self.mure_evidence = (
            self.mure_evidence * decays[None, :]
            + innovation[:, None]
        )
        steady_variance = (
            self.mure_null_variance[:, None]
            / np.maximum(1.0 - decays[None, :] ** 2, 1e-8)
        )
        self.mure_standardized = (
            self.mure_evidence / np.sqrt(np.maximum(steady_variance, 1e-8))
        )
        return self.mure_standardized

    def mure_consensus_score(self, mode: str) -> np.ndarray:
        z = np.asarray(self.mure_standardized, dtype=np.float64)
        if mode == "mean":
            return z.mean(axis=1)
        if mode == "median":
            return np.median(z, axis=1)
        if mode == "softmin":
            beta = float(self.args.mure_softmin_beta)
            shifted = -beta * z
            maximum = shifted.max(axis=1, keepdims=True)
            log_mean_exp = (
                maximum[:, 0]
                + np.log(np.exp(shifted - maximum).mean(axis=1))
            )
            return -log_mean_exp / beta
        if mode == "consensus":
            return (
                z.mean(axis=1)
                - float(self.args.mure_consistency_penalty) * z.std(axis=1)
            )
        raise ValueError(f"Unknown MURE score mode: {mode}")

    def latent_risk(self, posterior: np.ndarray) -> np.ndarray:
        """Posterior expected unresolved loss above a private baseline."""
        excess = np.maximum(
            self.grid[None, :] - self.baseline_x[:, None]
            - self.args.need_margin,
            0.0,
        )
        return np.sum(np.asarray(posterior, dtype=np.float64) * excess, axis=1)

    def predict_no_action(self, posterior: np.ndarray) -> np.ndarray:
        """One-window latent prediction when the client is not trained."""
        predicted = np.asarray(posterior, dtype=np.float64) @ self.latent_transition
        hazard = float(self.args.latent_hazard)
        predicted = (1.0 - hazard) * predicted + hazard * self.upward_prior
        predicted /= np.maximum(predicted.sum(axis=1, keepdims=True), 1e-15)
        return predicted

    def contract_toward_baseline(
        self,
        posterior: np.ndarray,
        response: float,
    ) -> np.ndarray:
        """
        Action transition: training contracts only the excess state toward the
        client's private calibration baseline.  Linear interpolation on the
        public grid preserves mass and introduces no extra private access.
        """
        posterior = np.asarray(posterior, dtype=np.float64)
        response = float(np.clip(response, 0.0, 1.0))
        n, grid_size = posterior.shape
        contracted = np.zeros_like(posterior)
        scale = float(grid_size - 1)
        for client in range(n):
            baseline = float(self.baseline_x[client])
            target = self.grid.copy()
            above = target > baseline
            target[above] = baseline + (1.0 - response) * (target[above] - baseline)
            position = np.clip(target * scale, 0.0, scale)
            lower = np.floor(position).astype(np.int64)
            upper = np.minimum(lower + 1, grid_size - 1)
            upper_weight = position - lower
            np.add.at(
                contracted[client], lower,
                posterior[client] * (1.0 - upper_weight),
            )
            np.add.at(
                contracted[client], upper,
                posterior[client] * upper_weight,
            )
        contracted /= np.maximum(contracted.sum(axis=1, keepdims=True), 1e-15)
        return contracted

    def pooled_response_mean(self) -> float:
        total = self.pooled_response_alpha + self.pooled_response_beta
        return float(self.pooled_response_alpha / max(total, 1e-15))

    def update_pooled_private_response(
        self,
        counterfactual_need: np.ndarray,
        no_action_observed_need: np.ndarray,
    ) -> None:
        """
        Learn one shrinkage-regularized action response from all previously
        selected clients.  The no-action posterior is used for this update, so
        the assumed action transition cannot manufacture its own evidence.
        """
        raw_improvement = np.asarray(counterfactual_need) - np.asarray(
            no_action_observed_need,
        )
        unselected = ~self.previous_selected
        self.passive_improvement = (
            float(np.median(raw_improvement[unselected]))
            if np.any(unselected) else 0.0
        )
        selected = self.previous_selected
        if np.any(selected):
            incremental = raw_improvement[selected] - self.passive_improvement
            denom = np.maximum(
                counterfactual_need[selected], self.args.response_need_floor,
            )
            soft_success = np.clip(incremental / denom, 0.0, 1.0)
            confidence = np.clip(
                counterfactual_need[selected]
                / self.args.response_confidence_scale,
                0.0,
                1.0,
            )
            prior_strength = float(self.args.response_prior_strength)
            prior_mean = float(self.args.response_prior_mean)
            prior_alpha = prior_strength * prior_mean
            prior_beta = prior_strength * (1.0 - prior_mean)
            decay = float(self.args.response_forgetting)
            self.pooled_response_alpha = (
                prior_alpha
                + decay * (self.pooled_response_alpha - prior_alpha)
                + float(np.sum(confidence * soft_success))
            )
            self.pooled_response_beta = (
                prior_beta
                + decay * (self.pooled_response_beta - prior_beta)
                + float(np.sum(confidence * (1.0 - soft_success)))
            )
        pooled = self.pooled_response_mean()
        self.response_mean[:] = pooled
        self.response_alpha[:] = self.pooled_response_alpha
        self.response_beta[:] = self.pooled_response_beta

    def update_continuous_posterior(self, reports: np.ndarray) -> None:
        self.ensure_calibrated()
        no_action_prediction = self.predict_no_action(self.latent_posterior)
        counterfactual_need = self.latent_risk(no_action_prediction)

        y = np.asarray(reports, dtype=np.int64)
        likelihood = np.maximum(
            self.channel[:, y].T,
            self.args.emission_floor,
        )

        # This posterior deliberately excludes the previous action and is used
        # only to estimate its response without self-confirmation.
        no_action_observed = no_action_prediction * likelihood
        no_action_observed /= np.maximum(
            no_action_observed.sum(axis=1, keepdims=True), 1e-15,
        )
        no_action_observed_need = self.latent_risk(no_action_observed)

        action_conditioned = self.kind == "privmav_act"
        if action_conditioned and np.any(self.previous_selected):
            response_before_observation = self.pooled_response_mean()
            action_prediction = self.contract_toward_baseline(
                no_action_prediction,
                response_before_observation,
            )
            predicted = np.where(
                self.previous_selected[:, None],
                action_prediction,
                no_action_prediction,
            )
        else:
            predicted = no_action_prediction

        posterior = predicted * likelihood
        posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-15)
        self.latent_posterior = posterior
        self.need = self.latent_risk(posterior)
        threshold = self.baseline_x[:, None] + self.args.need_margin
        self.tail_probability = np.sum(
            posterior * (self.grid[None, :] > threshold),
            axis=1,
        )

        if action_conditioned:
            self.update_pooled_private_response(
                counterfactual_need,
                no_action_observed_need,
            )
        else:
            self.passive_improvement = 0.0

    def counterfactual_action_value(self) -> np.ndarray:
        """Expected next-window unresolved-risk reduction from one update."""
        no_action = self.predict_no_action(self.latent_posterior)
        action = self.contract_toward_baseline(
            no_action,
            self.pooled_response_mean(),
        )
        self.counterfactual_no_action_need = self.latent_risk(no_action)
        self.counterfactual_action_need = self.latent_risk(action)
        value = (
            self.counterfactual_no_action_need
            - self.counterfactual_action_need
        )
        return np.maximum(value, 0.0)

    def select(
        self,
        t: int,
        reports: Optional[np.ndarray],
        tie: np.ndarray,
        common_random: np.ndarray,
        active: Optional[np.ndarray] = None,
        oracle_normalized_loss: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = self.args.num_clients
        self.age += 1.0

        if t < self.args.calib_end:
            ids = np.asarray(common_random, dtype=np.int64)
            scores = 1.0 - np.asarray(tie)
        elif self.kind == "random":
            ids = np.asarray(common_random, dtype=np.int64)
            scores = 1.0 - np.asarray(tie)
        elif self.kind == "oracle":
            if active is None or oracle_normalized_loss is None:
                raise ValueError("Oracle requires active identities and unprivatized losses")
            scores = np.asarray(oracle_normalized_loss, dtype=np.float64).copy()
            scores += np.asarray(active, dtype=np.float64) * 2.0
            ids = top_b(scores, self.args.budget_b, tie)
        elif self.kind == "avs2025_raw":
            # Fixed-budget port of the official selection rule in
            # Thomas et al., "Adaption via Selection" (ESOCC 2025), official
            # repository commit 2f4b7ac: sample candidates, request their raw
            # current losses, then retain the highest-loss clients.  The raw
            # loss makes this a deliberately non-private contemporary baseline.
            if oracle_normalized_loss is None:
                raise ValueError("AvS-2025 raw baseline requires current raw local loss")
            pool_size = min(
                n, max(self.args.budget_b,
                       self.args.avs_candidate_multiplier * self.args.budget_b),
            )
            candidate_order = np.argsort(np.asarray(tie, dtype=np.float64))
            candidates = candidate_order[:pool_size]
            candidate_scores = np.asarray(oracle_normalized_loss, dtype=np.float64)[candidates]
            local_tie = np.asarray(tie, dtype=np.float64)[candidates]
            local_ids = top_b(candidate_scores, self.args.budget_b, local_tie)
            ids = candidates[local_ids]
            scores = np.full(n, -np.inf, dtype=np.float64)
            scores[candidates] = candidate_scores
        elif self.kind == "poc_private":
            # Power-of-Choice from the official HiCS-FL benchmark: draw dB
            # candidates and choose the B largest losses.  Here the only
            # admissible loss proxy is the common OGPM report bin.
            pool_size = min(
                n, max(self.args.budget_b,
                       self.args.poc_candidate_multiplier * self.args.budget_b),
            )
            candidates = np.argsort(np.asarray(tie, dtype=np.float64))[:pool_size]
            private_scores = (
                np.asarray(reports, dtype=np.float64) + 0.5
            ) / self.args.report_levels
            local_ids = top_b(
                private_scores[candidates], self.args.budget_b,
                np.asarray(tie, dtype=np.float64)[candidates],
            )
            ids = candidates[local_ids]
            scores = np.full(n, -np.inf, dtype=np.float64)
            scores[candidates] = private_scores[candidates]
        elif self.kind == "oort_private":
            if self.published_adapter is None:
                raise RuntimeError("Oort-OGPM adapter was not initialized")
            if t < self.args.calib_end + self.args.oort_warmup_rounds:
                ids = np.asarray(common_random, dtype=np.int64)
                scores = self.published_adapter.score_vector()
                scores[ids] = np.nanmax(
                    np.nan_to_num(scores, nan=0.0, neginf=0.0)
                ) + 1.0
            else:
                ids = self.published_adapter.select(self.args.budget_b)
                scores = self.published_adapter.score_vector()
        elif self.kind == "hics2024":
            if self.published_adapter is None:
                raise RuntimeError("HiCS-FL adapter was not initialized")
            ids, scores = self.published_adapter.select(common_random, tie)
        elif self.kind == "fedgcs_private":
            if self.published_adapter is None:
                raise RuntimeError("FedGCS-OGPM adapter was not initialized")
            ids, scores = self.published_adapter.select(
                np.asarray(reports, dtype=np.int64), tie, common_random,
            )
        elif self.kind == "topb":
            # One-window evidence ablation: no temporal history.
            scores = (np.asarray(reports, dtype=np.float64) + 0.5) / self.args.report_levels
            ids = top_b(scores, self.args.budget_b, tie)
        elif self.kind == "cusum":
            self.ensure_calibrated()
            idx = np.arange(n)
            y = np.asarray(reports, dtype=np.int64)
            llr = np.log(self.emission.q1[idx, y]) - np.log(self.emission.q0[idx, y])
            self.cusum = np.maximum(0.0, self.cusum + llr - self.args.cusum_drift)
            scores = self.cusum.copy()
            ids = top_b(scores, self.args.budget_b, tie)
        elif self.kind == "belief":
            self.ensure_calibrated()
            idx = np.arange(n)
            y = np.asarray(reports, dtype=np.int64)
            predicted = (
                self.belief * self.args.persist_if_untrained
                + (1.0 - self.belief) * self.args.belief_hazard
            )
            predicted = np.clip(predicted, 1e-8, 1.0 - 1e-8)
            l0 = self.emission.q0[idx, y]
            l1 = self.emission.q1[idx, y]
            evidence = (1.0 - predicted) * l0 + predicted * l1
            posterior = predicted * l1 / np.maximum(evidence, 1e-15)
            posterior = np.clip(posterior, 1e-8, 1.0 - 1e-8)

            x_num = (
                (1.0 - predicted) * self.emission.stable_x_numerator[idx, y]
                + predicted * self.emission.drift_x_numerator[idx, y]
            )
            expected_x = x_num / np.maximum(evidence, 1e-15)
            instant_severity = np.clip(
                (expected_x - self.emission.baseline_x)
                / np.maximum(1.0 - self.emission.baseline_x, 0.10),
                0.0,
                1.0,
            )
            self.severity = (
                self.args.severity_ema * instant_severity
                + (1.0 - self.args.severity_ema) * self.severity
            )

            age_bonus = self.args.age_weight * np.minimum(self.age / self.args.age_cap, 1.0)
            scores = posterior * (1.0 + self.args.severity_weight * self.severity) + age_bonus
            self.belief = posterior
            ids = top_b(scores, self.args.budget_b, tie)
        elif self.kind.startswith("mcglr_"):
            # Exact channel-aware GLR charts.  Each chart corresponds to one
            # predeclared positive loss shift; no true drift label or raw loss
            # is observed by this controller.
            self.ensure_calibrated()
            idx = np.arange(n)
            y = np.asarray(reports, dtype=np.int64)
            llr = (
                np.log(self.mcglr_emission.q_shift[idx, :, y])
                - np.log(self.mcglr_emission.q0[idx, y])[:, None]
            )
            self.mcglr_charts = np.maximum(
                0.0,
                self.mcglr_charts + llr - self.args.mcglr_drift,
            )
            mode = self.kind[len("mcglr_"):]
            if mode == "max":
                scores = np.max(self.mcglr_charts, axis=1)
            elif mode == "softmax":
                beta = float(self.args.mcglr_softmax_beta)
                z = beta * self.mcglr_charts
                zmax = np.max(z, axis=1, keepdims=True)
                scores = (
                    zmax[:, 0]
                    + np.log(np.mean(np.exp(z - zmax), axis=1))
                ) / beta
            elif mode == "top2":
                k = min(2, self.mcglr_charts.shape[1])
                scores = np.mean(
                    np.partition(self.mcglr_charts, -k, axis=1)[:, -k:],
                    axis=1,
                )
            elif mode == "zmax":
                scores = np.max(
                    self.mcglr_charts
                    / np.sqrt(self.mcglr_null_variance),
                    axis=1,
                )
            else:
                raise ValueError(f"Unknown MC-PrivGLR mode: {mode}")
            self.mcglr_score = np.asarray(scores, dtype=np.float64).copy()
            ids = top_b(scores, self.args.budget_b, tie)
        elif self.kind.startswith("caps_r"):
            self.ensure_calibrated()
            try:
                retention = int(self.kind[len("caps_r"):]) / 100.0
            except ValueError as exc:
                raise ValueError(
                    f"Malformed CAPS controller kind: {self.kind}"
                ) from exc
            if not (0.0 < retention < 1.0):
                raise ValueError(f"Invalid CAPS retention in {self.kind}")

            idx = np.arange(n)
            y = np.asarray(reports, dtype=np.int64)
            # Controlled two-state transition.  If a client was selected, its
            # unresolved probability survives the action only with the fixed
            # public retention.  A fresh drift can still arrive through the
            # same hazard used by the historical Belief baseline.
            effective_previous = np.where(
                self.previous_selected,
                retention * self.belief,
                self.belief,
            )
            predicted = (
                effective_previous * self.args.persist_if_untrained
                + (1.0 - effective_previous) * self.args.belief_hazard
            )
            predicted = np.clip(predicted, 1e-8, 1.0 - 1e-8)
            l0 = self.emission.q0[idx, y]
            l1 = self.emission.q1[idx, y]
            evidence = (1.0 - predicted) * l0 + predicted * l1
            posterior = predicted * l1 / np.maximum(evidence, 1e-15)
            posterior = np.clip(posterior, 1e-8, 1.0 - 1e-8)
            self.belief = posterior
            scores = posterior.copy()
            ids = top_b(scores, self.args.budget_b, tie)
        elif self.kind.startswith("mure_"):
            self.update_mure_evidence(np.asarray(reports, dtype=np.int64))
            try:
                mode, retention_token = self.kind[len("mure_"):].rsplit("_r", 1)
                retention = int(retention_token) / 100.0
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Malformed MURE controller kind: {self.kind}"
                ) from exc
            if not (0.0 <= retention <= 1.0):
                raise ValueError(f"Invalid MURE retention in {self.kind}")
            scores = self.mure_consensus_score(mode)
            self.mure_score = scores.copy()
            ids = top_b(scores, self.args.budget_b, tie)
            if t >= self.args.calib_end and retention < 1.0:
                # One unified action-conditioned state transition.  We retire
                # only old evidence; the next private report remains untouched
                # and can rebuild priority if the client still needs training.
                self.mure_evidence[ids] *= retention
        elif self.kind.startswith("pspa"):
            self.update_pspa_posterior(np.asarray(reports, dtype=np.int64))
            if self.kind == "pspa_mean":
                # Bayes risk for a one-step unresolved-shift objective.
                scores = self.pspa_expected_shift.copy()
            elif self.kind == "pspa_prob":
                # Ablation: ignores posterior drift magnitude.
                scores = self.pspa_change_probability.copy()
            else:
                raise ValueError(f"Unknown PSPA controller kind: {self.kind}")
            ids = top_b(scores, self.args.budget_b, tie)
        elif self.kind.startswith("privmav"):
            self.update_continuous_posterior(np.asarray(reports, dtype=np.int64))
            if self.kind == "privmav_need":
                scores = self.need.copy()
            elif self.kind == "privmav_tail":
                # Probability that the latent loss exceeds the private
                # client-specific calibration reference by need_margin.
                scores = self.tail_probability.copy()
            elif self.kind == "privmav_act":
                scores = self.counterfactual_action_value()
            else:
                raise ValueError(f"Unknown PrivMAV controller kind: {self.kind}")
            self.marginal_value = scores.copy()
            ids = top_b(scores, self.args.budget_b, tie)
        else:
            raise ValueError(f"Unknown controller kind: {self.kind}")

        self.previous_selected[:] = False
        self.previous_selected[ids] = True
        self.previous_need = self.need.copy()
        self.age[ids] = 0.0
        return ids, np.asarray(scores, dtype=np.float64)


def policy_specs(args) -> List[Tuple[str, str, Optional[float]]]:
    all_specs: List[Tuple[str, str, Optional[float]]] = [
        ("random", "random", None),
        ("avs2025_raw", "avs2025_raw", None),
        ("oracle", "oracle", None),
    ]
    for epsilon in args.epsilons:
        label = epsilon_label(epsilon)
        all_specs.extend([
            (f"poc_ogpm_{label}", "poc_private", epsilon),
            (f"oort_official_ogpm_{label}", "oort_private", epsilon),
            (f"hics2024_source_adapted_ogpm_{label}", "hics2024", epsilon),
            (f"fedgcs2024_source_adapted_ogpm_{label}", "fedgcs_private", epsilon),
            (f"topb_ogpm_{label}", "topb", epsilon),
            (f"cusum_ogpm_{label}", "cusum", epsilon),
            (f"belief_shift_ogpm_{label}", "belief", epsilon),
        ])
        for mode in ("max", "softmax", "top2", "zmax"):
            all_specs.append((
                f"mcglr_{mode}_ogpm_{label}",
                f"mcglr_{mode}",
                epsilon,
            ))

    requested = set(args.policy_subset)
    if "all" in requested:
        return all_specs
    if "paper" in requested:
        return [
            spec for spec in all_specs
            if spec[1] in {
                "random", "oracle", "poc_private", "oort_private",
                "hics2024", "fedgcs_private", "mcglr_top2",
            }
        ]
    if "screen" in requested:
        return [
            spec for spec in all_specs
            if spec[1] in {"random", "oracle", "cusum", "belief"}
            or spec[1].startswith("mcglr_")
        ]

    selected = [
        spec for spec in all_specs
        if spec[0] in requested or spec[1] in requested
    ]
    unresolved = requested - {name for name, _, _ in selected} - {
        kind for _, kind, _ in selected
    }
    if unresolved:
        raise ValueError(
            "Unknown --policy_subset token(s): " + ", ".join(sorted(unresolved))
        )
    if not selected:
        raise ValueError("--policy_subset selected no policies")
    return selected


def policy_provenance(kind: str) -> Tuple[str, str]:
    if kind == "oort_private":
        return (
            "official-benchmark-code+OGPM-interface",
            "HiCS-FL/UCBsampler.py (Oort baseline in NeurIPS-2024 artifact)",
        )
    if kind == "hics2024":
        return (
            "source-adapted+official-primitives",
            "HiCS-FL/clustering.py and server/server_hics.py, NeurIPS 2024",
        )
    if kind == "fedgcs_private":
        return (
            "official-architecture+source-adapted-OGPM-interface",
            "GenerativeFL/autos encoder-predictor-decoder, IJCAI 2024",
        )
    if kind == "poc_private":
        return (
            "source-adapted+OGPM-interface",
            "HiCS-FL/server/server_poc.py (Power-of-Choice baseline)",
        )
    if kind.startswith("mcglr_"):
        return "ours-frozen", "V13 MC-PrivGLR-top2"
    if kind == "random":
        return "control", "uniform random selection"
    if kind == "oracle":
        return "upper-bound", "true drift identities and raw monitoring loss"
    return "diagnostic", kind


# =============================================================================
# Warm-up, calibration anchors, one-policy closed-loop experiment
# =============================================================================


def extract_state_dict(payload) -> dict:
    if isinstance(payload, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise ValueError("Checkpoint does not contain a recognized state_dict")


def train_or_load_warmup(
    args,
    train_x: np.ndarray,
    train_y: np.ndarray,
    private_idx: np.ndarray,
    train_clients: List[np.ndarray],
    test_x: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
) -> Tuple[dict, float]:
    path = Path(args.warmup_ckpt)
    model = make_model(args.num_classes).to(device)
    loaded = False
    if path.exists() and not args.rebuild_warmup:
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            checkpoint_mode = (
                payload.get("warmup_mode") if isinstance(payload, dict) else None
            )
            if checkpoint_mode != args.warmup_mode:
                raise ValueError(
                    f"checkpoint warmup_mode={checkpoint_mode} but "
                    f"requested {args.warmup_mode}"
                )
            model.load_state_dict(extract_state_dict(payload), strict=True)
            loaded = True
            print(f"Loaded warm-up checkpoint: {path.resolve()}")
        except Exception as exc:
            print(f"Warm-up checkpoint incompatible ({exc}); retraining.")
    if not loaded:
        if args.warmup_mode == "central":
            print(
                f"Training CENTRAL smoke-test warm-up for {args.warmup_epochs} epochs..."
            )
            dataset = ArrayDataset(train_x, train_y, private_idx)
            generator = torch.Generator().manual_seed(args.seed)
            loader = DataLoader(
                dataset, batch_size=args.warmup_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                generator=generator,
            )
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.warmup_lr, momentum=0.9,
                weight_decay=args.warmup_weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.warmup_epochs),
            )
            scaler = make_grad_scaler(device, args.amp)
            for epoch in range(args.warmup_epochs):
                model.train()
                running = 0.0
                seen = 0
                for x, y in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with amp_autocast(device, args.amp):
                        loss = F.cross_entropy(model(x), y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running += float(loss.detach()) * y.numel()
                    seen += y.numel()
                scheduler.step()
                print(
                    f"  central epoch {epoch + 1:02d}/{args.warmup_epochs}: "
                    f"loss={running / max(seen, 1):.4f}"
                )
        else:
            print(
                f"Training TRUE-FEDAVG warm-up for {args.warmup_rounds} rounds | "
                f"clients/round={args.warmup_clients_per_round} | "
                f"local_steps={args.warmup_local_steps}"
            )
            rng = np.random.default_rng(args.seed + 4301)
            for round_index in range(args.warmup_rounds):
                selected = rng.choice(
                    args.num_clients,
                    size=min(args.warmup_clients_per_round, args.num_clients),
                    replace=False,
                )
                progress = round_index / max(1, args.warmup_rounds - 1)
                lr = 0.5 * args.warmup_lr * (1.0 + math.cos(math.pi * progress))
                global_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                local_states: List[dict] = []
                final_losses = []
                for client in selected:
                    local_model = make_model(args.num_classes).to(device)
                    local_model.load_state_dict(global_state, strict=True)
                    local_model.train()
                    optimizer = torch.optim.SGD(
                        local_model.parameters(), lr=lr,
                        momentum=args.warmup_momentum,
                        weight_decay=args.warmup_weight_decay,
                    )
                    scaler = make_grad_scaler(device, args.amp)
                    for _ in range(args.warmup_local_steps):
                        idx = rng.choice(
                            train_clients[int(client)],
                            size=args.warmup_batch_size,
                            replace=True,
                        )
                        x = images_to_device(train_x[idx], device)
                        y = torch.as_tensor(
                            train_y[idx], dtype=torch.long, device=device,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        with amp_autocast(device, args.amp):
                            loss = F.cross_entropy(local_model(x), y)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                local_model.parameters(), args.grad_clip,
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    final_losses.append(float(loss.detach()))
                    local_states.append({
                        key: value.detach().clone()
                        for key, value in local_model.state_dict().items()
                    })
                    del local_model, optimizer, scaler
                warm_weights = np.asarray(
                    [len(train_clients[int(client)]) for client in selected],
                    dtype=np.float64,
                )
                aggregate_local_states(model, local_states, warm_weights)
                del local_states, global_state
                if (
                    round_index == 0
                    or (round_index + 1) % args.warmup_log_every == 0
                    or round_index + 1 == args.warmup_rounds
                ):
                    print(
                        f"  FedAvg round {round_index + 1:03d}/{args.warmup_rounds}: "
                        f"lr={lr:.5f}, mean-local-loss={np.mean(final_losses):.4f}"
                    )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "version": VERSION,
            "warmup_mode": args.warmup_mode,
        }, path)
        print(f"Saved warm-up checkpoint: {path.resolve()}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, test_y.size, args.infer_batch_size):
            stop = min(test_y.size, start + args.infer_batch_size)
            x = images_to_device(test_x[start:stop], device)
            y = torch.as_tensor(test_y[start:stop], dtype=torch.long, device=device)
            with amp_autocast(device, args.amp):
                logits = model(x)
            correct += int((logits.argmax(1) == y).sum())
            total += y.numel()
    accuracy = correct / max(total, 1)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, float(accuracy)


def normalize_losses(losses: np.ndarray, scale: float) -> np.ndarray:
    """Monotone bounded map with no hard upper clipping or private anchors."""
    losses = np.maximum(np.asarray(losses, dtype=np.float64), 0.0)
    return losses / np.maximum(losses + float(scale), 1e-12)


def build_or_load_calibration_report_cache(
    args,
    fraction: float,
    seed: int,
    initial_state: dict,
    plan: StreamPlan,
    train_x: np.ndarray,
    train_y: np.ndarray,
    required_epsilons: Sequence[float],
    device: torch.device,
    cache_dir: Path,
) -> Optional[Dict[str, np.ndarray]]:
    """Cache the monitoring-only prefix shared by every private policy.

    The cache is valid only when calibration performs no adaptation.  Reports
    after calibration are deliberately never shared because every scheduling
    policy then owns a different model and a different closed loop.
    """
    if args.adapt_during_calibration or not required_epsilons:
        return None
    labels = [epsilon_label(float(e)) for e in required_epsilons]
    cache_dir.mkdir(parents=True, exist_ok=True)
    fraction_tag = int(round(10000 * fraction))
    warmup_path = Path(args.warmup_ckpt)
    warmup_stamp = {"path": str(warmup_path)}
    if warmup_path.exists():
        stat = warmup_path.stat()
        warmup_stamp.update({
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    signature = stable_hash({
        "version": VERSION,
        "seed": seed,
        "fraction": fraction,
        "calib_end": args.calib_end,
        "num_clients": args.num_clients,
        "window_batch": args.window_batch,
        "loss_scale": args.loss_scale,
        "report_levels": args.report_levels,
        "epsilons": labels,
        "warmup_checkpoint": warmup_stamp,
    })
    path = cache_dir / (
        f"v15_calibration_f{fraction_tag:04d}_s{seed}_{signature}.npz"
    )
    if path.exists() and not args.rebuild_calibration_cache:
        with np.load(path, allow_pickle=False) as payload:
            cached = {label: payload[label].astype(np.int64) for label in labels}
        expected = (args.calib_end, args.num_clients)
        if all(value.shape == expected for value in cached.values()):
            print(f"  calibration cache loaded: {path.name}", flush=True)
            return cached
        print(f"  invalid calibration cache ignored: {path.name}", flush=True)

    model = make_model(args.num_classes).to(device)
    model.load_state_dict(initial_state)
    reports_by_label = {
        label: np.empty((args.calib_end, args.num_clients), dtype=np.int64)
        for label in labels
    }
    for t in range(args.calib_end):
        raw_monitor = monitor_all_clients(
            model, t, plan, train_x, train_y, device,
            args.monitor_chunk_clients, args.amp,
        )
        normalized = normalize_losses(raw_monitor, args.loss_scale)
        for epsilon, label in zip(required_epsilons, labels):
            reports_by_label[label][t] = ogpm_report_quantized(
                normalized,
                float(epsilon),
                args.report_levels,
                plan.ogpm_uniforms[t],
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **reports_by_label)
    temporary.replace(path)
    print(f"  calibration cache saved: {path.name}", flush=True)
    return reports_by_label


def round_identity_metrics(active: np.ndarray, selected: np.ndarray, scores: np.ndarray) -> dict:
    active = np.asarray(active, dtype=bool)
    selected_mask = np.zeros(active.size, dtype=bool)
    selected_mask[np.asarray(selected, dtype=np.int64)] = True
    positives = int(active.sum())
    if positives == 0:
        return {
            "identity_auprc": float("nan"),
            "identity_auroc": float("nan"),
            "recall_at_b": float("nan"),
            "false_selection_rate": float("nan"),
        }
    return {
        "identity_auprc": average_precision_binary(active, scores),
        "identity_auroc": roc_auc_binary(active, scores),
        "recall_at_b": float((selected_mask & active).sum() / positives),
        "false_selection_rate": float((selected_mask & ~active).sum() / max(1, selected_mask.sum())),
    }


def run_one_policy(
    name: str,
    kind: str,
    epsilon: Optional[float],
    args,
    fraction: float,
    seed: int,
    initial_state: dict,
    plan: StreamPlan,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    loss_scale: float,
    channels: Dict[str, np.ndarray],
    grid: np.ndarray,
    device: torch.device,
    calibration_report_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    model = make_model(args.num_classes).to(device)
    model.load_state_dict(initial_state)
    channel = None if epsilon is None else channels[epsilon_label(epsilon)]
    controller = OnlineController(kind, epsilon, args, grid, channel)
    controller.initialize_published_baseline(plan.client_sizes, seed, device)
    curve_rows: List[dict] = []
    decision_rows: List[dict] = []
    identity_rows: List[dict] = []
    first_selected = np.full(args.num_clients, np.nan, dtype=np.float64)
    report_bits = int(math.ceil(math.log2(args.report_levels)))
    selection_seconds = 0.0
    monitoring_seconds = 0.0
    adaptation_seconds = 0.0

    for t in range(args.horizon):
        active = t >= plan.onset
        reports: Optional[np.ndarray] = None
        normalized: Optional[np.ndarray] = None

        # Calibration is a monitoring-only prefix by default. Start the online
        # allocation phase with equal ages and no fictitious previous action.
        if t == args.calib_end and not args.adapt_during_calibration:
            controller.age[:] = 0.0
            controller.previous_selected[:] = False

        private_controller = (
            kind in {
                "topb", "cusum", "belief", "poc_private",
                "oort_private", "hics2024", "fedgcs_private",
            }
            or kind.startswith("privmav")
            or kind.startswith("pspa")
            or kind.startswith("mure_")
            or kind.startswith("caps_")
            or kind.startswith("mcglr_")
        )

        if (
            t < args.calib_end
            and not args.adapt_during_calibration
            and private_controller
            and calibration_report_cache is not None
        ):
            cache_label = epsilon_label(float(epsilon))
            reports = np.asarray(
                calibration_report_cache[cache_label][t], dtype=np.int64,
            )
            controller.record_calibration(reports)
            clock = time.perf_counter()
            selected, scores = controller.select(
                t, reports, plan.ties[t], plan.common_random[t], active=None,
            )
            selection_seconds += time.perf_counter() - clock
        elif t < args.calib_end and not args.adapt_during_calibration:
            # No policy adapts during the calibration prefix by default, and
            # every controller selects the same common-random clients there.
            # Non-private policies therefore need no redundant monitoring.
            clock = time.perf_counter()
            selected, scores = controller.select(
                t, None, plan.ties[t], plan.common_random[t], active=None,
            )
            selection_seconds += time.perf_counter() - clock
        elif kind == "random":
            clock = time.perf_counter()
            selected, scores = controller.select(
                t, None, plan.ties[t], plan.common_random[t], active=None,
            )
            selection_seconds += time.perf_counter() - clock
        else:
            clock = time.perf_counter()
            raw_monitor = monitor_all_clients(
                model, t, plan, train_x, train_y, device,
                args.monitor_chunk_clients, args.amp,
            )
            monitoring_seconds += time.perf_counter() - clock
            normalized = normalize_losses(raw_monitor, loss_scale)
            if private_controller:
                reports = ogpm_report_quantized(
                    normalized, float(epsilon), args.report_levels, plan.ogpm_uniforms[t],
                )
                if t < args.calib_end:
                    controller.record_calibration(reports)
            clock = time.perf_counter()
            selected, scores = controller.select(
                t,
                reports,
                plan.ties[t],
                plan.common_random[t],
                active=active if kind == "oracle" else None,
                oracle_normalized_loss=(
                    normalized if kind in {"oracle", "avs2025_raw"} else None
                ),
            )
            selection_seconds += time.perf_counter() - clock

        newly = np.isnan(first_selected[selected]) & active[selected]
        first_selected[selected[newly]] = t
        id_metrics = round_identity_metrics(active, selected, scores)
        identity_rows.append({"t": t, **id_metrics})

        if args.save_decisions:
            decision_rows.append({
                "changed_fraction": fraction,
                "seed": seed,
                "t": t,
                "policy": name,
                "selected_ids": json.dumps(selected.tolist()),
                "active_ids": json.dumps(np.flatnonzero(active).tolist()),
                "score_vector": json.dumps(np.round(scores, 8).tolist()),
                "report_bins": "" if reports is None else json.dumps(reports.tolist()),
                "belief_vector": (
                    json.dumps(np.round(controller.belief, 8).tolist())
                    if kind == "belief" or kind.startswith("caps_") else ""
                ),
                "mcglr_score_vector": (
                    json.dumps(np.round(controller.mcglr_score, 8).tolist())
                    if kind.startswith("mcglr_") else ""
                ),
                "severity_vector": (
                    json.dumps(np.round(controller.severity, 8).tolist())
                    if kind == "belief" else ""
                ),
                "need_vector": (
                    json.dumps(np.round(controller.need, 8).tolist())
                    if kind.startswith("privmav") else ""
                ),
                "response_mean_vector": (
                    json.dumps(np.round(controller.response_mean, 8).tolist())
                    if kind.startswith("privmav") else ""
                ),
                "marginal_value_vector": (
                    json.dumps(np.round(controller.marginal_value, 8).tolist())
                    if kind.startswith("privmav") else ""
                ),
                "counterfactual_no_action_need_vector": (
                    json.dumps(
                        np.round(controller.counterfactual_no_action_need, 8).tolist()
                    )
                    if kind == "privmav_act" else ""
                ),
                "counterfactual_action_need_vector": (
                    json.dumps(
                        np.round(controller.counterfactual_action_need, 8).tolist()
                    )
                    if kind == "privmav_act" else ""
                ),
                "pooled_action_response": (
                    float(controller.pooled_response_mean())
                    if kind == "privmav_act" else ""
                ),
                "tail_probability_vector": (
                    json.dumps(np.round(controller.tail_probability, 8).tolist())
                    if kind.startswith("privmav") else ""
                ),
                "passive_improvement": (
                    float(controller.passive_improvement)
                    if kind.startswith("privmav") else ""
                ),
                "pspa_change_probability_vector": (
                    json.dumps(
                        np.round(controller.pspa_change_probability, 8).tolist()
                    ) if kind.startswith("pspa") else ""
                ),
                "pspa_expected_shift_vector": (
                    json.dumps(
                        np.round(controller.pspa_expected_shift, 8).tolist()
                    ) if kind.startswith("pspa") else ""
                ),
                "mure_standardized_evidence": (
                    json.dumps(
                        np.round(controller.mure_standardized, 8).tolist()
                    ) if kind.startswith("mure_") else ""
                ),
                "mure_score_vector": (
                    json.dumps(np.round(controller.mure_score, 8).tolist())
                    if kind.startswith("mure_") else ""
                ),
            })

        should_adapt = args.adapt_during_calibration or t >= args.calib_end
        if should_adapt:
            clock = time.perf_counter()
            feedback = adapt_one_fedavg_round(
                model, selected, t, plan, train_x, train_y, args, device,
            )
            controller.observe_training_feedback(
                t, selected, reports, feedback,
            )
            adaptation_seconds += time.perf_counter() - clock
        else:
            controller.previous_selected[:] = False

        if t in plan.eval_indices:
            losses, acc = evaluate_clients(
                model, t, plan, test_x, test_y, device,
                args.eval_chunk_clients, args.amp,
            )
            changed = plan.changed
            stable = ~changed
            adaptation_rounds = (
                t + 1 if args.adapt_during_calibration
                else max(0, t - args.calib_end + 1)
            )
            curve_rows.append({
                "changed_fraction": fraction,
                "seed": seed,
                "t": t,
                "policy": name,
                "epsilon": "none" if epsilon is None else epsilon_label(epsilon),
                "global_acc": float(acc.mean()),
                "changed_acc": float(acc[changed].mean()),
                "stable_acc": float(acc[stable].mean()),
                "balanced_group_acc": float(0.5 * (acc[changed].mean() + acc[stable].mean())),
                "worst_group_acc": float(min(acc[changed].mean(), acc[stable].mean())),
                "changed_loss": float(losses[changed].mean()),
                "stable_loss": float(losses[stable].mean()),
                "model_uploads": int(adaptation_rounds * args.budget_b),
                "model_upload_bytes": int(adaptation_rounds * args.budget_b * args.model_bytes),
                "monitor_report_bits": int(
                    0 if kind in {"random", "oracle"}
                    else (
                        (t + 1)
                        * min(
                            args.num_clients,
                            args.avs_candidate_multiplier * args.budget_b,
                        )
                        * 32
                        if kind == "avs2025_raw"
                        else (t + 1) * args.num_clients * report_bits
                    )
                ),
            })

    curves = pd.DataFrame(curve_rows)
    ids = pd.DataFrame(identity_rows)
    post_mask = ids.t >= args.change_time
    post_eval = curves[curves.t >= args.post_start]
    changed_delays = first_selected[plan.changed] - plan.onset[plan.changed]
    changed_delays = changed_delays[np.isfinite(changed_delays) & (changed_delays >= 0)]
    summary = {
        "changed_fraction": fraction,
        "seed": seed,
        "policy": name,
        "source_status": policy_provenance(kind)[0],
        "source_detail": policy_provenance(kind)[1],
        "epsilon": "none" if epsilon is None else epsilon_label(epsilon),
        "mean_global_acc_post": float(post_eval.global_acc.mean()),
        "mean_changed_acc_post": float(post_eval.changed_acc.mean()),
        "mean_stable_acc_post": float(post_eval.stable_acc.mean()),
        "balanced_group_acc_post": float(post_eval.balanced_group_acc.mean()),
        "worst_group_acc_post": float(post_eval.worst_group_acc.mean()),
        "mean_changed_loss_post": float(post_eval.changed_loss.mean()),
        "mean_stable_loss_post": float(post_eval.stable_loss.mean()),
        "identity_auprc_post": float(ids.loc[post_mask, "identity_auprc"].mean()),
        "identity_auroc_post": float(ids.loc[post_mask, "identity_auroc"].mean()),
        "recall_at_b_post": float(ids.loc[post_mask, "recall_at_b"].mean()),
        "false_selection_rate_post": float(ids.loc[post_mask, "false_selection_rate"].mean()),
        "mean_first_selection_delay": float(changed_delays.mean()) if changed_delays.size else float("nan"),
        "final_model_uploads": int(
            (args.horizon if args.adapt_during_calibration else args.horizon - args.calib_end)
            * args.budget_b
        ),
        "final_model_upload_bytes": int(
            (args.horizon if args.adapt_during_calibration else args.horizon - args.calib_end)
            * args.budget_b * args.model_bytes
        ),
        "final_model_download_bytes": int(
            (args.horizon if args.adapt_during_calibration else args.horizon - args.calib_end)
            * args.budget_b * args.model_bytes
        ),
        "final_monitor_report_bits": int(
            0 if kind in {"random", "oracle"}
            else (
                args.horizon
                * min(
                    args.num_clients,
                    args.avs_candidate_multiplier * args.budget_b,
                )
                * 32
                if kind == "avs2025_raw"
                else args.horizon * args.num_clients * report_bits
            )
        ),
        "total_selection_seconds": float(selection_seconds),
        "mean_selection_milliseconds": float(
            1000.0 * selection_seconds / max(1, args.horizon)
        ),
        "total_monitoring_seconds": float(monitoring_seconds),
        "total_adaptation_seconds": float(adaptation_seconds),
    }
    summary["final_uplink_bits"] = int(
        summary["final_model_upload_bytes"] * 8
        + summary["final_monitor_report_bits"]
    )
    summary["final_bidirectional_bits"] = int(
        (summary["final_model_upload_bytes"] + summary["final_model_download_bytes"]) * 8
        + summary["final_monitor_report_bits"]
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return curves, summary, pd.DataFrame(decision_rows)


# =============================================================================
# Recovery, aggregation, diagnostics, and main program
# =============================================================================


def add_recovery_metrics(curves: pd.DataFrame, summaries: pd.DataFrame, args) -> pd.DataFrame:
    result = summaries.copy()
    result["recovered"] = 0
    result["recovery_round"] = np.nan
    result["client_rounds_to_recovery"] = np.nan
    result["bytes_to_recovery"] = np.nan
    result["total_bits_to_recovery"] = np.nan
    result["uplink_bits_to_recovery"] = np.nan
    result["downlink_bits_to_recovery"] = np.nan
    result["bidirectional_bits_to_recovery"] = np.nan
    result["random_changed_acc_reference"] = np.nan
    result["oracle_changed_acc_reference"] = np.nan
    result["recovery_changed_acc_target"] = np.nan
    result["oracle_normalized_loss_recovery"] = np.nan
    result["oracle_normalized_balanced_recovery"] = np.nan
    result["changed_loss_regret_vs_oracle"] = np.nan
    result["balanced_acc_gap_vs_oracle"] = np.nan
    result["stable_acc_drop_vs_random"] = np.nan
    result["cumulative_changed_loss_regret_vs_oracle"] = np.nan
    report_bits = int(math.ceil(math.log2(args.report_levels)))

    for (fraction, seed), group in curves.groupby(["changed_fraction", "seed"]):
        oracle = group[group.policy == "oracle"].sort_values("t")
        random_curve = group[group.policy == "random"].sort_values("t")
        if oracle.empty or random_curve.empty:
            continue
        random_reference = float(
            random_curve.tail(args.oracle_final_points).changed_acc.mean()
        )
        oracle_reference = float(
            oracle.tail(args.oracle_final_points).changed_acc.mean()
        )
        post_oracle = oracle[oracle.t >= args.post_start]
        post_random = random_curve[random_curve.t >= args.post_start]
        oracle_loss = float(post_oracle.changed_loss.mean())
        random_loss = float(post_random.changed_loss.mean())
        oracle_balanced = float(post_oracle.balanced_group_acc.mean())
        random_balanced = float(post_random.balanced_group_acc.mean())
        random_stable = float(post_random.stable_acc.mean())
        gain = oracle_reference - random_reference
        if gain <= 1e-8:
            continue
        target = random_reference + args.recovery_ratio * gain

        for policy, policy_curve in group.groupby("policy"):
            policy_curve = policy_curve.sort_values("t")
            mask = (
                np.isclose(result.changed_fraction, fraction)
                & (result.seed == seed)
                & (result.policy == policy)
            )
            result.loc[mask, "random_changed_acc_reference"] = random_reference
            result.loc[mask, "oracle_changed_acc_reference"] = oracle_reference
            result.loc[mask, "recovery_changed_acc_target"] = target
            eligible = policy_curve[policy_curve.t >= args.post_start]
            policy_loss = float(eligible.changed_loss.mean())
            policy_balanced = float(eligible.balanced_group_acc.mean())
            policy_stable = float(eligible.stable_acc.mean())
            loss_denominator = random_loss - oracle_loss
            balanced_denominator = oracle_balanced - random_balanced
            if abs(loss_denominator) > 1e-12:
                result.loc[mask, "oracle_normalized_loss_recovery"] = (
                    random_loss - policy_loss
                ) / loss_denominator
            if abs(balanced_denominator) > 1e-12:
                result.loc[mask, "oracle_normalized_balanced_recovery"] = (
                    policy_balanced - random_balanced
                ) / balanced_denominator
            result.loc[mask, "changed_loss_regret_vs_oracle"] = (
                policy_loss - oracle_loss
            )
            result.loc[mask, "balanced_acc_gap_vs_oracle"] = (
                policy_balanced - oracle_balanced
            )
            result.loc[mask, "stable_acc_drop_vs_random"] = (
                random_stable - policy_stable
            )
            paired_curve = eligible[["t", "changed_loss"]].merge(
                post_oracle[["t", "changed_loss"]],
                on="t", suffixes=("_policy", "_oracle"),
            )
            if not paired_curve.empty:
                result.loc[mask, "cumulative_changed_loss_regret_vs_oracle"] = float(
                    (
                        paired_curve.changed_loss_policy
                        - paired_curve.changed_loss_oracle
                    ).sum()
                )
            values = eligible.changed_acc.to_numpy(dtype=float) >= target
            recovery_t = None
            patience = max(1, args.recovery_patience)
            for j in range(0, max(0, values.size - patience + 1)):
                if bool(values[j:j + patience].all()):
                    recovery_t = int(eligible.iloc[j].t)
                    break
            if recovery_t is None:
                continue
            adaptation_rounds = (
                recovery_t + 1 if args.adapt_during_calibration
                else max(0, recovery_t - args.calib_end + 1)
            )
            client_rounds = adaptation_rounds * args.budget_b
            model_bits = client_rounds * args.model_bytes * 8
            if policy in {"random", "oracle"}:
                monitoring_bits = 0
            elif policy == "avs2025_raw":
                monitoring_bits = (
                    (recovery_t + 1)
                    * min(
                        args.num_clients,
                        args.avs_candidate_multiplier * args.budget_b,
                    )
                    * 32
                )
            else:
                monitoring_bits = (
                    (recovery_t + 1) * args.num_clients * report_bits
                )
            result.loc[mask, "recovered"] = 1
            result.loc[mask, "recovery_round"] = recovery_t
            result.loc[mask, "client_rounds_to_recovery"] = client_rounds
            result.loc[mask, "bytes_to_recovery"] = client_rounds * args.model_bytes
            result.loc[mask, "total_bits_to_recovery"] = model_bits + monitoring_bits
            result.loc[mask, "uplink_bits_to_recovery"] = model_bits + monitoring_bits
            result.loc[mask, "downlink_bits_to_recovery"] = model_bits
            result.loc[mask, "bidirectional_bits_to_recovery"] = (
                2 * model_bits + monitoring_bits
            )
    return result


def summarize_primary(primary: pd.DataFrame, metrics: Sequence[str], bootstrap_runs: int) -> pd.DataFrame:
    rows = []
    for policy, group in primary.groupby("policy"):
        for m, metric in enumerate(metrics):
            mean, lo, hi = bootstrap_mean_ci(
                group[metric].to_numpy(dtype=float), bootstrap_runs, 70000 + m,
            )
            rows.append({
                "policy": policy,
                "metric": metric,
                "mean": mean,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n": int(group[metric].notna().sum()),
            })
    return pd.DataFrame(rows)


def strongest_private_reference(
    primary: pd.DataFrame,
    epsilon: float,
) -> Optional[str]:
    label = epsilon_label(epsilon)
    names = [
        f"cusum_ogpm_{label}",
        f"belief_shift_ogpm_{label}",
    ]
    refs = primary[primary.policy.isin(names)]
    if refs.empty:
        return None
    means = refs.groupby("policy").mean(numeric_only=True)
    return str(means.mean_changed_loss_post.idxmin())


def choose_mcglr_candidate(
    primary: pd.DataFrame,
    epsilon: float,
) -> Optional[str]:
    """Apply the fixed development-selection rule within the MC-PrivGLR family."""
    label = epsilon_label(epsilon)
    reference_name = strongest_private_reference(primary, epsilon)
    candidates = primary[
        primary.policy.str.startswith("mcglr_")
        & primary.policy.str.endswith(f"_ogpm_{label}")
    ]
    reference = primary[primary.policy == reference_name]
    if candidates.empty or reference.empty:
        return None
    ref_means = reference[
        ["mean_changed_loss_post", "balanced_group_acc_post", "mean_stable_acc_post"]
    ].mean()
    board = candidates.groupby("policy", as_index=False)[[
        "mean_changed_loss_post", "balanced_group_acc_post", "mean_stable_acc_post",
    ]].mean()
    admissible = board[
        (board.balanced_group_acc_post >= ref_means.balanced_group_acc_post - 0.005)
        & (board.mean_stable_acc_post >= ref_means.mean_stable_acc_post - 0.010)
    ]
    if admissible.empty:
        return None
    admissible = admissible.sort_values(
        ["mean_changed_loss_post", "balanced_group_acc_post", "policy"],
        ascending=[True, False, True],
    )
    return str(admissible.iloc[0].policy)


def paired_comparisons(primary: pd.DataFrame, args) -> pd.DataFrame:
    label = epsilon_label(args.primary_epsilon)
    proposed = f"mcglr_top2_ogpm_{label}"
    if proposed not in set(primary.policy):
        return pd.DataFrame(columns=[
            "proposed", "baseline", "metric", "preferred_direction",
            "mean_diff_proposed_minus_baseline", "ci95_lo", "ci95_hi",
            "n_pairs",
        ])
    baselines = [
        "random",
        f"poc_ogpm_{label}",
        f"oort_official_ogpm_{label}",
        f"hics2024_source_adapted_ogpm_{label}",
        f"fedgcs2024_source_adapted_ogpm_{label}",
    ]
    available = set(primary.policy.unique())
    baselines = [name for name in baselines if name in available]
    metrics = [
        "identity_auprc_post",
        "recall_at_b_post",
        "balanced_group_acc_post",
        "mean_changed_acc_post",
        "mean_changed_loss_post",
        "false_selection_rate_post",
        "client_rounds_to_recovery",
        "oracle_normalized_loss_recovery",
        "changed_loss_regret_vs_oracle",
        "cumulative_changed_loss_regret_vs_oracle",
    ]
    lower_is_better = {
        "mean_changed_loss_post", "false_selection_rate_post",
        "client_rounds_to_recovery", "changed_loss_regret_vs_oracle",
        "cumulative_changed_loss_regret_vs_oracle",
    }
    rows = []
    for b, baseline in enumerate(baselines):
        a = primary[primary.policy == proposed]
        z = primary[primary.policy == baseline]
        for m, metric in enumerate(metrics):
            merged = a[["seed", metric]].merge(
                z[["seed", metric]], on="seed", suffixes=("_proposed", "_baseline"),
            )
            diff = merged[f"{metric}_proposed"] - merged[f"{metric}_baseline"]
            mean, lo, hi = bootstrap_mean_ci(
                diff.to_numpy(dtype=float), args.bootstrap_runs, 81000 + 100 * b + m,
            )
            rows.append({
                "proposed": proposed,
                "baseline": baseline,
                "metric": metric,
                "preferred_direction": "lower" if metric in lower_is_better else "higher",
                "mean_diff_proposed_minus_baseline": mean,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_pairs": int(np.isfinite(diff).sum()),
            })
    return pd.DataFrame(rows)


def development_leaderboard(primary: pd.DataFrame, args) -> pd.DataFrame:
    """Transparent screening table; no arbitrary weighted composite score."""
    metrics = [
        "mean_changed_acc_post",
        "mean_stable_acc_post",
        "balanced_group_acc_post",
        "mean_changed_loss_post",
        "identity_auprc_post",
        "recall_at_b_post",
        "false_selection_rate_post",
        "recovered",
        "client_rounds_to_recovery",
        "oracle_normalized_loss_recovery",
        "oracle_normalized_balanced_recovery",
        "changed_loss_regret_vs_oracle",
        "cumulative_changed_loss_regret_vs_oracle",
    ]
    board = primary.groupby("policy", as_index=False)[metrics].mean()
    label = epsilon_label(args.primary_epsilon)
    refs = {
        "random": "random",
        "cusum": f"cusum_ogpm_{label}",
        "belief": f"belief_shift_ogpm_{label}",
    }
    indexed = board.set_index("policy")
    for short, policy in refs.items():
        if policy not in indexed.index:
            continue
        ref = indexed.loc[policy]
        board[f"delta_changed_acc_vs_{short}"] = (
            board.mean_changed_acc_post - ref.mean_changed_acc_post
        )
        board[f"delta_balanced_acc_vs_{short}"] = (
            board.balanced_group_acc_post - ref.balanced_group_acc_post
        )
        board[f"delta_changed_loss_vs_{short}"] = (
            board.mean_changed_loss_post - ref.mean_changed_loss_post
        )
    if "random" in indexed.index:
        board["stable_acc_drop_vs_random"] = (
            indexed.loc["random", "mean_stable_acc_post"]
            - board.mean_stable_acc_post
        )
    board["private_candidate"] = board.policy.str.contains(
        "mcglr_", regex=False,
    )
    board = board.sort_values(
        ["mean_changed_loss_post", "balanced_group_acc_post"],
        ascending=[True, False],
    ).reset_index(drop=True)
    board.insert(0, "screen_rank", np.arange(1, len(board) + 1))
    return board


def legacy_v9_predeclared_pretest_gate(primary: pd.DataFrame, args) -> pd.DataFrame:
    """Apply fixed screening rules without inspecting individual outcomes.

    This is a development gate, not a significance test.  Its purpose is to
    prevent post-hoc method selection after seeing three-seed results.  The
    final paper run must use fresh seeds and report paired confidence
    intervals irrespective of whether they are favourable.
    """
    rows = []
    primary_label = epsilon_label(args.primary_epsilon)

    for epsilon in args.epsilons:
        label = epsilon_label(epsilon)
        proposed_name = f"privmav_act_ogpm_{label}"
        cusum_name = f"cusum_ogpm_{label}"
        proposed = primary[primary.policy == proposed_name]
        cusum = primary[primary.policy == cusum_name]
        merged = proposed.merge(cusum, on="seed", suffixes=("_v9", "_cusum"))
        if merged.empty:
            rows.append({
                "gate": f"channel_{label}_versus_cusum",
                "required": True,
                "passed": False,
                "n_pairs": 0,
                "loss_diff_v9_minus_ref": np.nan,
                "balanced_acc_diff_v9_minus_ref": np.nan,
                "stable_acc_diff_v9_minus_ref": np.nan,
                "loss_win_count": 0,
                "minimum_loss_wins": 0,
                "criterion": "missing paired rows",
            })
            continue

        loss_diff = (
            merged.mean_changed_loss_post_v9
            - merged.mean_changed_loss_post_cusum
        )
        balanced_diff = (
            merged.balanced_group_acc_post_v9
            - merged.balanced_group_acc_post_cusum
        )
        stable_diff = (
            merged.mean_stable_acc_post_v9
            - merged.mean_stable_acc_post_cusum
        )
        n_pairs = int(len(merged))
        loss_wins = int((loss_diff < 0.0).sum())
        minimum_wins = int(math.ceil(2.0 * n_pairs / 3.0))

        if label == primary_label:
            passed = bool(
                loss_diff.mean() < 0.0
                and loss_wins >= minimum_wins
                and balanced_diff.mean() >= -0.005
                and stable_diff.mean() >= -0.010
            )
            criterion = (
                "mean changed-loss diff < 0; loss wins >= ceil(2n/3); "
                "balanced-acc diff >= -0.005; stable-acc diff >= -0.010"
            )
        else:
            # A fixed 1% relative margin avoids rejecting the method for a
            # numerically immaterial loss change at a harder privacy channel.
            loss_margin = 0.01 * abs(merged.mean_changed_loss_post_cusum.mean())
            passed = bool(
                loss_diff.mean() <= loss_margin
                and balanced_diff.mean() >= -0.005
                and stable_diff.mean() >= -0.010
            )
            criterion = (
                f"mean changed-loss diff <= fixed 1% reference margin "
                f"({loss_margin:.6g}); balanced-acc diff >= -0.005; "
                "stable-acc diff >= -0.010"
            )

        rows.append({
            "gate": f"channel_{label}_versus_cusum",
            "required": True,
            "passed": passed,
            "n_pairs": n_pairs,
            "loss_diff_v9_minus_ref": float(loss_diff.mean()),
            "balanced_acc_diff_v9_minus_ref": float(balanced_diff.mean()),
            "stable_acc_diff_v9_minus_ref": float(stable_diff.mean()),
            "loss_win_count": loss_wins,
            "minimum_loss_wins": minimum_wins if label == primary_label else 0,
            "criterion": criterion,
        })

    if args.include_ablations:
        proposed_name = f"privmav_act_ogpm_{primary_label}"
        tail_name = f"posterior_tail_ogpm_{primary_label}"
        proposed = primary[primary.policy == proposed_name]
        tail = primary[primary.policy == tail_name]
        merged = proposed.merge(tail, on="seed", suffixes=("_v9", "_tail"))
        if merged.empty:
            passed = False
            n_pairs = loss_wins = minimum_wins = 0
            loss_mean = balanced_mean = stable_mean = np.nan
        else:
            loss_diff = (
                merged.mean_changed_loss_post_v9
                - merged.mean_changed_loss_post_tail
            )
            balanced_diff = (
                merged.balanced_group_acc_post_v9
                - merged.balanced_group_acc_post_tail
            )
            stable_diff = (
                merged.mean_stable_acc_post_v9
                - merged.mean_stable_acc_post_tail
            )
            n_pairs = int(len(merged))
            loss_wins = int((loss_diff < 0.0).sum())
            minimum_wins = int(math.ceil(2.0 * n_pairs / 3.0))
            loss_mean = float(loss_diff.mean())
            balanced_mean = float(balanced_diff.mean())
            stable_mean = float(stable_diff.mean())
            passed = bool(
                loss_mean < 0.0
                and loss_wins >= minimum_wins
                and balanced_mean >= -0.005
                and stable_mean >= -0.010
            )
        rows.append({
            "gate": "action_transition_versus_posterior_tail",
            "required": True,
            "passed": passed,
            "n_pairs": n_pairs,
            "loss_diff_v9_minus_ref": loss_mean,
            "balanced_acc_diff_v9_minus_ref": balanced_mean,
            "stable_acc_diff_v9_minus_ref": stable_mean,
            "loss_win_count": loss_wins,
            "minimum_loss_wins": minimum_wins,
            "criterion": (
                "mean changed-loss diff < 0; loss wins >= ceil(2n/3); "
                "balanced-acc diff >= -0.005; stable-acc diff >= -0.010"
            ),
        })

    gate = pd.DataFrame(rows)
    required_pass = bool(gate.loc[gate.required, "passed"].all()) if not gate.empty else False
    gate["overall_pass"] = required_pass
    gate["decision"] = "FREEZE_FOR_CONFIRMATION" if required_pass else "DO_NOT_CONFIRM"
    return gate


def legacy_v12_predeclared_pretest_gate(primary: pd.DataFrame, args) -> pd.DataFrame:
    """Predeclared PSPA-vs-CUSUM development gate.

    This is an engineering screen, not a significance test.  Confirmatory
    experiments must use fresh seeds after all PSPA choices are frozen.
    """
    rows: List[dict] = []
    primary_label = epsilon_label(args.primary_epsilon)
    for epsilon in args.epsilons:
        label = epsilon_label(epsilon)
        proposed_name = f"pspa_mean_ogpm_{label}"
        reference_name = strongest_private_reference(primary, epsilon)
        proposed = primary[primary.policy == proposed_name]
        reference = primary[primary.policy == reference_name]
        merged = proposed.merge(
            reference, on="seed", suffixes=("_v10", "_cusum"),
        )
        if merged.empty:
            continue
        loss_diff = (
            merged.mean_changed_loss_post_v10
            - merged.mean_changed_loss_post_cusum
        )
        balanced_diff = (
            merged.balanced_group_acc_post_v10
            - merged.balanced_group_acc_post_cusum
        )
        stable_diff = (
            merged.mean_stable_acc_post_v10
            - merged.mean_stable_acc_post_cusum
        )
        n_pairs = int(len(merged))
        loss_wins = int((loss_diff < 0.0).sum())
        minimum_wins = int(math.ceil(2.0 * n_pairs / 3.0))
        if label == primary_label:
            passed = bool(
                loss_diff.mean() < 0.0
                and loss_wins >= minimum_wins
                and balanced_diff.mean() >= -0.003
                and stable_diff.mean() >= -0.010
            )
            criterion = (
                "mean changed-loss diff < 0; wins >= ceil(2n/3); "
                "balanced diff >= -0.003; stable diff >= -0.010"
            )
        else:
            margin = 0.01 * abs(
                float(merged.mean_changed_loss_post_cusum.mean())
            )
            passed = bool(
                loss_diff.mean() <= margin
                and balanced_diff.mean() >= -0.005
                and stable_diff.mean() >= -0.010
            )
            criterion = (
                f"changed-loss diff <= 1% reference margin ({margin:.6g}); "
                "balanced diff >= -0.005; stable diff >= -0.010"
            )
        rows.append({
            "gate": f"pspa_{label}_versus_cusum",
            "required": True,
            "passed": passed,
            "n_pairs": n_pairs,
            "loss_diff_v12_minus_ref": float(loss_diff.mean()),
            "balanced_acc_diff_v12_minus_ref": float(balanced_diff.mean()),
            "stable_acc_diff_v12_minus_ref": float(stable_diff.mean()),
            "loss_win_count": loss_wins,
            "minimum_loss_wins": minimum_wins if label == primary_label else 0,
            "criterion": criterion,
        })

    if args.include_ablations:
        proposed_name = f"pspa_mean_ogpm_{primary_label}"
        ablation_name = f"pspa_prob_ogpm_{primary_label}"
        proposed = primary[primary.policy == proposed_name]
        ablation = primary[primary.policy == ablation_name]
        merged = proposed.merge(
            ablation, on="seed", suffixes=("_mean", "_prob"),
        )
        if not merged.empty:
            loss_diff = (
                merged.mean_changed_loss_post_mean
                - merged.mean_changed_loss_post_prob
            )
            balanced_diff = (
                merged.balanced_group_acc_post_mean
                - merged.balanced_group_acc_post_prob
            )
            stable_diff = (
                merged.mean_stable_acc_post_mean
                - merged.mean_stable_acc_post_prob
            )
            n_pairs = int(len(merged))
            loss_wins = int((loss_diff < 0.0).sum())
            minimum_wins = int(math.ceil(2.0 * n_pairs / 3.0))
            passed = bool(
                loss_diff.mean() <= 0.0
                and balanced_diff.mean() >= -0.003
                and stable_diff.mean() >= -0.010
            )
            rows.append({
                "gate": "expected_shift_versus_probability_ablation",
                "required": False,
                "passed": passed,
                "n_pairs": n_pairs,
                "loss_diff_v12_minus_ref": float(loss_diff.mean()),
                "balanced_acc_diff_v12_minus_ref": float(balanced_diff.mean()),
                "stable_acc_diff_v12_minus_ref": float(stable_diff.mean()),
                "loss_win_count": loss_wins,
                "minimum_loss_wins": minimum_wins,
                "criterion": (
                    "diagnostic ablation: mean score should not lose material "
                    "adaptation utility to probability-only score"
                ),
            })

    gate = pd.DataFrame(rows)
    if gate.empty:
        return pd.DataFrame(columns=[
            "gate", "required", "passed", "n_pairs",
            "loss_diff_v12_minus_ref", "balanced_acc_diff_v12_minus_ref",
            "stable_acc_diff_v12_minus_ref", "loss_win_count",
            "minimum_loss_wins", "criterion", "overall_pass", "decision",
        ])
    required = gate[gate.required]
    overall_pass = bool((not required.empty) and required.passed.all())
    gate["overall_pass"] = overall_pass
    if overall_pass:
        decision = "FREEZE_FOR_CONFIRMATION"
    elif int(gate.n_pairs.max()) < 3:
        decision = "CONTINUE_SCREENING"
    else:
        decision = "DO_NOT_CONFIRM"
    gate["decision"] = decision
    return gate


def legacy_v12_fail_fast_screen_reason(
    summaries: pd.DataFrame,
    args,
) -> Optional[str]:
    """Return a predeclared stop reason after completed paired seed(s)."""
    label = epsilon_label(args.primary_epsilon)
    proposed = summaries[
        summaries.policy == f"pspa_mean_ogpm_{label}"
    ]
    reference = summaries[
        summaries.policy == f"cusum_ogpm_{label}"
    ]
    merged = proposed.merge(
        reference, on=["changed_fraction", "seed"],
        suffixes=("_v10", "_cusum"),
    )
    if merged.empty:
        return None
    primary_rows = merged[np.isclose(
        merged.changed_fraction, args.primary_fraction,
    )]
    if primary_rows.empty:
        return None
    loss_ratio = (
        primary_rows.mean_changed_loss_post_v10
        / np.maximum(primary_rows.mean_changed_loss_post_cusum, 1e-12)
    )
    balanced_diff = (
        primary_rows.balanced_group_acc_post_v10
        - primary_rows.balanced_group_acc_post_cusum
    )
    if len(primary_rows) == 1 and (
        float(loss_ratio.iloc[0]) > 1.02
        or float(balanced_diff.iloc[0]) < -0.005
    ):
        return (
            "first paired seed exceeded the fixed 2% changed-loss or "
            "-0.5pp balanced-accuracy boundary"
        )
    if len(primary_rows) >= 2:
        wins = int((
            primary_rows.mean_changed_loss_post_v10
            < primary_rows.mean_changed_loss_post_cusum
        ).sum())
        if wins == 0:
            return "PSPA lost changed loss on every completed paired seed"
    return None


def predeclared_pretest_gate(primary: pd.DataFrame, args) -> pd.DataFrame:
    """Freeze only an MC-PrivGLR variant that passes fresh-seed constraints."""
    rows: List[dict] = []
    primary_label = epsilon_label(args.primary_epsilon)
    for epsilon in args.epsilons:
        label = epsilon_label(epsilon)
        proposed_name = choose_mcglr_candidate(primary, epsilon)
        reference_name = strongest_private_reference(primary, epsilon)
        if proposed_name is None:
            continue
        proposed = primary[primary.policy == proposed_name]
        reference = primary[primary.policy == reference_name]
        merged = proposed.merge(
            reference, on="seed", suffixes=("_v13", "_ref"),
        )
        if merged.empty:
            continue
        loss_diff = (
            merged.mean_changed_loss_post_v13
            - merged.mean_changed_loss_post_ref
        )
        balanced_diff = (
            merged.balanced_group_acc_post_v13
            - merged.balanced_group_acc_post_ref
        )
        stable_diff = (
            merged.mean_stable_acc_post_v13
            - merged.mean_stable_acc_post_ref
        )
        n_pairs = int(len(merged))
        wins = int((loss_diff < 0.0).sum())
        minimum_wins = int(math.ceil(2.0 * n_pairs / 3.0))
        is_primary = label == primary_label
        if is_primary:
            passed = bool(
                n_pairs >= 3
                and float(loss_diff.mean()) < 0.0
                and wins >= minimum_wins
                and float(balanced_diff.mean()) >= -0.003
                and float(stable_diff.mean()) >= -0.010
            )
            criterion = (
                "n>=3; mean changed-loss diff < 0; wins >= ceil(2n/3); "
                "balanced diff >= -0.003; stable diff >= -0.010"
            )
        else:
            margin = 0.01 * abs(
                float(merged.mean_changed_loss_post_ref.mean())
            )
            passed = bool(
                n_pairs >= 3
                and float(loss_diff.mean()) <= margin
                and float(balanced_diff.mean()) >= -0.005
                and float(stable_diff.mean()) >= -0.010
            )
            criterion = (
                f"n>=3; changed-loss diff <= 1% margin ({margin:.6g}); "
                "balanced diff >= -0.005; stable diff >= -0.010"
            )
        rows.append({
            "gate": f"selected_mcglr_{label}_versus_{reference_name}",
            "reference": reference_name,
            "selected_candidate": proposed_name,
            "required": True,
            "passed": passed,
            "n_pairs": n_pairs,
            "loss_diff_v15_minus_ref": float(loss_diff.mean()),
            "balanced_acc_diff_v15_minus_ref": float(balanced_diff.mean()),
            "stable_acc_diff_v15_minus_ref": float(stable_diff.mean()),
            "loss_win_count": wins,
            "minimum_loss_wins": minimum_wins,
            "criterion": criterion,
        })

    gate = pd.DataFrame(rows)
    if gate.empty:
        return pd.DataFrame(columns=[
            "gate", "selected_candidate", "reference", "required", "passed", "n_pairs",
            "loss_diff_v15_minus_ref", "balanced_acc_diff_v15_minus_ref",
            "stable_acc_diff_v15_minus_ref", "loss_win_count",
            "minimum_loss_wins", "criterion", "overall_pass", "decision",
        ])
    required = gate[gate.required]
    overall_pass = bool((not required.empty) and required.passed.all())
    gate["overall_pass"] = overall_pass
    if overall_pass:
        decision = "FREEZE_FOR_CONFIRMATION"
    elif int(gate.n_pairs.max()) < 3:
        decision = "CONTINUE_SCREENING"
    else:
        decision = "DO_NOT_CONFIRM"
    gate["decision"] = decision
    return gate


def fail_fast_screen_reason(
    summaries: pd.DataFrame,
    args,
) -> Optional[str]:
    """Stop only when every admissible MC-PrivGLR variant loses after two seeds."""
    label = epsilon_label(args.primary_epsilon)
    primary = summaries[np.isclose(
        summaries.changed_fraction, args.primary_fraction,
    )]
    reference_name = strongest_private_reference(primary, args.primary_epsilon)
    reference = primary[primary.policy == reference_name]
    candidates = primary[
        primary.policy.str.startswith("mcglr_")
        & primary.policy.str.endswith(f"_ogpm_{label}")
    ]
    if reference.empty or candidates.empty:
        return None
    seeds = sorted(set(reference.seed) & set(candidates.seed))
    if not seeds:
        return None

    candidate_names = sorted(candidates.policy.unique())
    comparisons = []
    for name in candidate_names:
        merged = candidates[candidates.policy == name].merge(
            reference, on="seed", suffixes=("_v13", "_ref"),
        )
        if merged.empty:
            continue
        comparisons.append({
            "policy": name,
            "n": len(merged),
            "loss_ratio": float(
                merged.mean_changed_loss_post_v13.mean()
                / max(merged.mean_changed_loss_post_ref.mean(), 1e-12)
            ),
            "balanced_diff": float(
                (merged.balanced_group_acc_post_v13
                 - merged.balanced_group_acc_post_ref).mean()
            ),
            "wins": int((
                merged.mean_changed_loss_post_v13
                < merged.mean_changed_loss_post_ref
            ).sum()),
        })
    if not comparisons:
        return None
    completed = max(item["n"] for item in comparisons)
    admissible = [
        item for item in comparisons if item["balanced_diff"] >= -0.005
    ]
    if admissible and completed >= 2 and all(item["wins"] == 0 for item in admissible):
        return "no admissible MC-PrivGLR variant has a changed-loss win after two seeds"
    return None


def scientific_config(args) -> dict:
    excluded = {
        "gpu", "out_dir", "rerun_completed", "allow_config_mismatch",
        "save_decisions", "checkpoint_policies", "num_workers", "self_test",
        "rebuild_warmup", "rebuild_calibration_cache", "rebuild_femnist_cache",
        "policy_subset",
        "fail_fast_screen", "num_stream_seeds", "include_ablations",
        "latent_process_sigma", "latent_hazard", "need_margin",
        "response_prior_mean", "response_prior_strength",
        "response_forgetting", "response_need_floor",
        "response_confidence_scale",
        "pspa_shifts", "pspa_prior", "pspa_persistence", "pspa_hazard",
        "mure_decays", "mure_drift_margin", "mure_consistency_penalty",
        "mure_softmin_beta",
        "caps_retentions",
    }
    config = {k: json_safe(v) for k, v in vars(args).items() if k not in excluded}
    config.update({
        "version": VERSION,
        "observation": "bounded loss l/(l+loss_scale)",
        "privacy_mechanism": "classical_OGPM_01",
        "privacy_mechanism_source": "official closed_form_mechanism.classical_mechanism_01",
        "quantization": "equal-width output bins only",
        "inference": (
            "parallel reflected private log-likelihood charts for a fixed public "
            "grid of positive shifts through the exact OGPM channel"
        ),
        "scheduler": "select the B largest multi-chart private GLR scores",
        "action_response": "none; the next private report closes the model loop",
        "adaptation": "independent local models + multi-step weighted FedAvg",
        "published_baselines": {
            "source_repo": "https://github.com/CityChan/HiCS-FL",
            "expected_commit": args.official_hics_commit,
            "oort_source": "UCBsampler.py direct call; OGPM bin replaces raw loss utility",
            "hics_source": "clustering.py direct primitives; V15 model/data-loop adapter",
            "power_of_choice_source": "server/server_poc.py rule; OGPM bin replaces raw loss",
            "ucb_sampler_sha256": file_sha256(
                Path(args.official_hics_root) / "UCBsampler.py"
            ),
            "clustering_sha256": file_sha256(
                Path(args.official_hics_root) / "clustering.py"
            ),
            "fedgcs_repo": "https://github.com/zhiyuan-ning/GenerativeFL",
            "fedgcs_expected_commit": args.official_fedgcs_commit,
            "fedgcs_model_sha256": file_sha256(
                Path(args.official_fedgcs_root) / "autos" / "model.py"
            ),
            "fedgcs_interface": (
                "official AUTOS architecture and loss; candidates/utility records "
                "built only from common OGPM reports"
            ),
        },
        "privacy_unit": "one client monitoring-window event",
        "trajectory_privacy_claim": False,
    })
    return config


def write_or_validate_config(args, output: Path) -> None:
    config = scientific_config(args)
    path = output / "v15_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if stable_hash(old) != stable_hash(config) and not args.allow_config_mismatch:
            raise RuntimeError(
                "Output directory contains a different scientific configuration. "
                "Use a new --out_dir or explicitly pass --allow_config_mismatch."
            )
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_args(args) -> None:
    if str(args.dataset).lower() not in {"cifar10", "cifar100", "femnist"}:
        raise ValueError("--dataset must be cifar10, cifar100, or femnist")
    if not (0 < args.budget_b <= args.num_clients):
        raise ValueError("--budget_b must be in [1, num_clients]")
    if not (1 <= args.calib_end <= args.change_time < args.horizon):
        raise ValueError("Require 1 <= calib_end <= change_time < horizon")
    if args.report_levels < 2:
        raise ValueError("--report_levels must be at least 2")
    if args.grid_size < args.report_levels:
        raise ValueError("--grid_size should be >= report_levels")
    if not (0 < args.persist_if_untrained <= 1):
        raise ValueError("--persist_if_untrained must be in (0,1]")
    if args.local_steps < 1 or args.warmup_local_steps < 1:
        raise ValueError("local step counts must be positive")
    if args.poc_candidate_multiplier < 1 or args.oort_warmup_rounds < 1:
        raise ValueError("PoC multiplier and Oort warm-up rounds must be positive")
    if args.hics_temperature <= 0 or args.hics_groups < 2:
        raise ValueError("HiCS temperature must be positive and groups >= 2")
    if args.fedgcs_candidates_per_round < 4:
        raise ValueError("--fedgcs_candidates_per_round must be at least 4")
    if args.fedgcs_replay_size < args.fedgcs_candidates_per_round:
        raise ValueError("--fedgcs_replay_size must cover at least one candidate batch")
    if min(args.fedgcs_train_every, args.fedgcs_train_epochs,
           args.fedgcs_batch_size, args.fedgcs_top_k) < 1:
        raise ValueError("FedGCS integer training controls must be positive")
    if args.fedgcs_lr <= 0 or not (0 <= args.fedgcs_trade_off <= 1):
        raise ValueError("FedGCS lr must be positive and trade_off in [0,1]")
    if not args.fedgcs_gradient_steps or any(x <= 0 for x in args.fedgcs_gradient_steps):
        raise ValueError("--fedgcs_gradient_steps must contain positive values")
    if args.loss_scale <= 0 or args.latent_process_sigma <= 0:
        raise ValueError("--loss_scale and --latent_process_sigma must be positive")
    if not (0 <= args.latent_hazard <= 1):
        raise ValueError("--latent_hazard must be in [0,1]")
    if args.need_margin < 0:
        raise ValueError("--need_margin must be nonnegative")
    if not (0 < args.response_prior_mean < 1):
        raise ValueError("--response_prior_mean must be in (0,1)")
    if args.response_prior_strength <= 0:
        raise ValueError("--response_prior_strength must be positive")
    if not (0 < args.response_forgetting <= 1):
        raise ValueError("--response_forgetting must be in (0,1]")
    if args.response_need_floor <= 0 or args.response_confidence_scale <= 0:
        raise ValueError("response scales must be positive")
    if not (0 < args.pspa_prior < 1):
        raise ValueError("--pspa_prior must be in (0,1)")
    if not (0 < args.pspa_persistence <= 1):
        raise ValueError("--pspa_persistence must be in (0,1]")
    if not (0 <= args.pspa_hazard < 1):
        raise ValueError("--pspa_hazard must be in [0,1)")
    shifts = np.asarray(args.pspa_shifts, dtype=np.float64)
    if shifts.size == 0 or np.any(shifts <= 0) or np.any(shifts >= 1):
        raise ValueError("--pspa_shifts must contain values strictly in (0,1)")
    if np.any(np.diff(shifts) <= 0):
        raise ValueError("--pspa_shifts must be strictly increasing")
    if not args.policy_subset:
        raise ValueError("--policy_subset cannot be empty")
    mcglr_shifts = np.asarray(args.mcglr_shifts, dtype=np.float64)
    if (
        mcglr_shifts.size < 2
        or np.any(mcglr_shifts <= 0)
        or np.any(mcglr_shifts >= 1)
        or np.any(np.diff(mcglr_shifts) <= 0)
    ):
        raise ValueError("--mcglr_shifts must be strictly increasing values in (0,1)")
    if args.mcglr_drift < 0 or args.mcglr_softmax_beta <= 0:
        raise ValueError("MC-PrivGLR drift must be nonnegative and beta positive")
    if args.avs_candidate_multiplier < 1:
        raise ValueError("--avs_candidate_multiplier must be at least 1")
    if args.warmup_mode not in {"fedavg", "central"}:
        raise ValueError("--warmup_mode must be fedavg or central")
    if not (0.05 <= args.femnist_test_fraction < 0.5):
        raise ValueError("--femnist_test_fraction must be in [0.05, 0.5)")
    if args.femnist_min_writer_samples <= args.min_client_samples:
        raise ValueError(
            "--femnist_min_writer_samples must exceed --min_client_samples"
        )
    if any(not (0 < f < 1) for f in args.changed_fractions):
        raise ValueError("Every changed fraction must be in (0,1)")
    if not any(
        (np.isinf(e) and np.isinf(args.primary_epsilon))
        or (np.isfinite(e) and np.isclose(e, args.primary_epsilon))
        for e in args.epsilons
    ):
        raise ValueError("--primary_epsilon must also appear in --epsilons")
    args.post_start = min(
        args.horizon - 1,
        args.change_time + max(0, args.async_window - 1) + max(0, args.ramp_windows - 1),
    )


def run_self_tests() -> None:
    grid = np.linspace(0.0, 1.0, 65)
    for epsilon in (4.0, 1.0, 0.5):
        channel = ogpm_discrete_channel(epsilon, 16, grid)
        diagnostic = validate_channel(epsilon, channel)
        assert diagnostic["valid"], diagnostic
    rng = np.random.default_rng(7)
    x = np.full(200000, 0.37)
    reports = ogpm_report_quantized(x, 1.0, 16, rng.random(x.size))
    empirical = np.bincount(reports, minlength=16) / reports.size
    exact = ogpm_discrete_channel(1.0, 16, np.array([0.37]))[0]
    assert np.max(np.abs(empirical - exact)) < 0.01
    mapped = normalize_losses(np.array([0.0, 0.1, 1.0, 10.0]), math.log(10.0))
    assert np.all(np.diff(mapped) > 0) and mapped.min() >= 0 and mapped.max() < 1

    args = argparse.Namespace(
        num_clients=4, belief_prior=0.05, response_prior_strength=4.0,
        response_prior_mean=0.5, calib_end=8, baseline_shrinkage=4.0,
        baseline_sigma=0.05, drift_shifts=[0.10, 0.20, 0.35],
        emission_floor=1e-10, need_margin=0.02,
        latent_process_sigma=0.03, latent_hazard=0.0025,
        response_need_floor=0.02, response_confidence_scale=0.10,
        response_forgetting=0.98, budget_b=2, report_levels=16,
        persist_if_untrained=0.995, belief_hazard=0.0025,
        severity_ema=0.25, severity_weight=0.75,
        age_weight=0.0, age_cap=20.0, cusum_drift=0.05,
        avs_candidate_multiplier=4,
        pspa_shifts=[0.05, 0.10, 0.20, 0.35], pspa_prior=0.05,
        pspa_persistence=0.995, pspa_hazard=0.0025,
        mure_decays=[0.50, 0.80, 0.95], mure_drift_margin=0.02,
        mure_consistency_penalty=0.50, mure_softmin_beta=2.0,
        caps_retentions=[0.98, 0.95, 0.90, 0.80],
        mcglr_shifts=[0.05, 0.10, 0.20, 0.35],
        mcglr_drift=0.02, mcglr_softmax_beta=2.0,
    )
    controller = OnlineController(
        "mcglr_max", 1.0, args, grid,
        ogpm_discrete_channel(1.0, 16, grid),
    )
    for t in range(args.calib_end):
        controller.record_calibration(
            np.array([(t + client) % 16 for client in range(args.num_clients)])
        )
    ids, scores = controller.select(
        args.calib_end, np.array([4, 7, 10, 13]),
        np.linspace(0.1, 0.4, 4), np.array([0, 1]),
    )
    assert ids.size == 2 and np.all(np.isfinite(scores))
    assert np.all(np.isfinite(controller.mcglr_charts))
    assert np.all(controller.mcglr_charts >= 0.0)
    controller.select(
        args.calib_end + 1, np.array([3, 6, 9, 12]),
        np.linspace(0.4, 0.1, 4), np.array([2, 3]),
    )
    assert np.all(np.isfinite(controller.mcglr_score))

    tiny = nn.Linear(2, 1, bias=False)
    first = {"weight": torch.ones_like(tiny.weight)}
    second = {"weight": 3.0 * torch.ones_like(tiny.weight)}
    aggregate_local_states(tiny, [first, second], np.array([1.0, 3.0]))
    assert torch.allclose(tiny.weight, torch.full_like(tiny.weight, 2.5))
    print(
        "V13 self-test passed: official OGPM, exact multi-chart private GLR, "
        "bounded loss, and weighted FedAvg."
    )


def main(args) -> None:
    args.epsilons = [parse_epsilon(x) for x in args.epsilons]
    args.primary_epsilon = parse_epsilon(args.primary_epsilon)
    args.drift_shifts = [float(x) for x in args.drift_shifts]
    args.pspa_shifts = [float(x) for x in args.pspa_shifts]
    args.mure_decays = [float(x) for x in args.mure_decays]
    args.caps_retentions = [float(x) for x in args.caps_retentions]
    args.mcglr_shifts = [float(x) for x in args.mcglr_shifts]
    if args.self_test:
        run_self_tests()
        return
    require_official_ogpm()
    args.dataset = str(args.dataset).lower()
    args.num_classes = {"cifar10": 10, "cifar100": 100, "femnist": 62}[args.dataset]
    if args.loss_scale is None:
        args.loss_scale = math.log(float(args.num_classes))
    set_dataset_statistics(args.dataset)
    validate_args(args)
    seed_everything(args.seed, args.deterministic)

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    output = Path(args.out_dir)
    per_seed_dir = output / "per_seed"
    temporary_model = make_model(args.num_classes)
    args.model_bytes = int(
        sum(p.numel() * p.element_size() for p in temporary_model.parameters())
    )
    if not hasattr(temporary_model, "fc") or not hasattr(temporary_model.fc, "in_features"):
        raise RuntimeError("Every benchmark model must expose final classifier model.fc")
    args.fc_features = int(temporary_model.fc.in_features)
    del temporary_model
    config_hash = stable_hash(scientific_config(args))
    per_policy_dir = output / "per_policy" / config_hash
    output.mkdir(parents=True, exist_ok=True)
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_policies:
        per_policy_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(args, output)

    print("=" * 112)
    print("V15: frozen MC-PrivGLR versus source-backed published baselines + FedGCS")
    print(f"dataset={args.dataset.upper()} | TRUE multi-step FedAvg")
    print(f"device={device} | N={args.num_clients} | report levels L={args.report_levels} ({math.ceil(math.log2(args.report_levels))} bits)")
    print(f"budget B={args.budget_b} | changed={args.changed_fractions} | eps={[epsilon_label(e) for e in args.epsilons]}")
    print(f"horizon={args.horizon} | calibration={args.calib_end} | change={args.change_time} | post_start={args.post_start}")
    print(
        f"MC-PrivGLR shifts={args.mcglr_shifts} | "
        f"chart drift={args.mcglr_drift:g} | softmax beta={args.mcglr_softmax_beta:g}"
    )
    print("Privacy: epsilon is per monitoring-window event; repeated reports compose across time.")
    print("Only Oracle receives active identities/raw pre-selection monitoring loss.")
    print(f"Official baseline source root: {Path(args.official_hics_root).resolve()}")
    print(f"Official FedGCS source root: {Path(args.official_fedgcs_root).resolve()}")
    print(f"Official OGPM source: {Path(inspect.getsourcefile(official_classical_ogpm_01)).resolve()}")
    print("=" * 112)

    print(f"Loading {args.dataset.upper()} TRAIN/TEST...")
    if args.dataset == "femnist":
        (
            train_x, train_y, test_x, test_y, loaded_classes,
            train_clients, test_clients, private_idx,
        ) = load_femnist_federated(args)
    else:
        train_x, train_y, test_x, test_y, loaded_classes = load_cifar(
            args.root, args.dataset,
        )
        _public_idx, private_idx = stratified_public_split(
            train_y, args.public_calib_size, args.data_seed,
        )
        train_clients = dirichlet_partition(
            train_y, private_idx, args.num_clients, args.dirichlet_alpha,
            args.data_seed + 11, args.min_client_samples,
        )
        test_clients = partition_test_by_train_priors(
            train_y, train_clients, test_y, args.num_clients, args.data_seed + 29,
            args.num_classes,
        )
    if loaded_classes != args.num_classes:
        raise RuntimeError("Dataset class-count mismatch")

    initial_state, warmup_acc = train_or_load_warmup(
        args, train_x, train_y, private_idx, train_clients,
        test_x, test_y, device,
    )
    print(f"warm-up ({args.warmup_mode}) {args.dataset.upper()} test accuracy={warmup_acc:.4f}")
    print(f"bounded loss mapping: x=loss/(loss+{args.loss_scale:.6f})")

    grid = np.linspace(0.0, 1.0, args.grid_size)
    channels: Dict[str, np.ndarray] = {}
    diagnostics = []
    for epsilon in args.epsilons:
        label = epsilon_label(epsilon)
        channel = ogpm_discrete_channel(epsilon, args.report_levels, grid)
        diagnostic = validate_channel(epsilon, channel, args.ogpm_ldp_tolerance)
        if not diagnostic["valid"]:
            raise RuntimeError(f"OGPM channel validation failed: {diagnostic}")
        channels[label] = channel
        diagnostics.append(diagnostic)
    atomic_to_csv(
        pd.DataFrame(diagnostics),
        output / "v15_privacy_channel_diagnostics.csv",
    )

    all_curves: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []
    all_decisions: List[pd.DataFrame] = []
    specs = policy_specs(args)
    expected_policy_names = {name for name, _, _ in specs}
    method_registry = []
    for name, kind, epsilon in specs:
        status, detail = policy_provenance(kind)
        method_registry.append({
            "policy": name,
            "controller_kind": kind,
            "epsilon": "none" if epsilon is None else epsilon_label(epsilon),
            "source_status": status,
            "source_detail": detail,
            "source_repo": (
                "https://github.com/CityChan/HiCS-FL"
                if kind in {"poc_private", "oort_private", "hics2024"}
                else (
                    "https://github.com/zhiyuan-ning/GenerativeFL"
                    if kind == "fedgcs_private" else ""
                )
            ),
        })
    atomic_to_csv(pd.DataFrame(method_registry), output / "v15_method_registry.csv")
    private_epsilons = sorted({
        float(epsilon) for _, kind, epsilon in specs
        if epsilon is not None and (
            kind in {
                "topb", "cusum", "belief", "poc_private",
                "oort_private", "hics2024", "fedgcs_private",
            }
            or kind.startswith("pspa")
            or kind.startswith("privmav")
            or kind.startswith("mure_")
            or kind.startswith("caps_")
            or kind.startswith("mcglr_")
        )
    })
    print(f"Policies per seed: {len(specs)} -> {[name for name, _, _ in specs]}")

    for fraction in args.changed_fractions:
        fraction_tag = int(round(10000 * fraction))
        for seed_index, seed in enumerate(range(args.seed, args.seed + args.num_stream_seeds), start=1):
            curve_path = per_seed_dir / f"v15_curve_f{fraction_tag:04d}_s{seed}.csv"
            summary_path = per_seed_dir / f"v15_summary_f{fraction_tag:04d}_s{seed}.csv"
            decision_path = per_seed_dir / f"v15_decisions_f{fraction_tag:04d}_s{seed}.csv"
            cached = False
            if curve_path.exists() and summary_path.exists() and (
                (not args.save_decisions) or decision_path.exists()
            ):
                cached_summary_probe = pd.read_csv(summary_path)
                cached = set(cached_summary_probe.policy) == expected_policy_names
            if cached and not args.rerun_completed:
                all_curves.append(pd.read_csv(curve_path))
                all_summaries.append(pd.read_csv(summary_path))
                if args.save_decisions:
                    all_decisions.append(pd.read_csv(decision_path))
                print(f"fraction={fraction:g} seed={seed} cached ({seed_index}/{args.num_stream_seeds})")
                if args.fail_fast_screen and seed_index < args.num_stream_seeds:
                    reason = fail_fast_screen_reason(
                        pd.concat(all_summaries, ignore_index=True), args,
                    )
                    if reason is not None:
                        print(
                            f"FAIL-FAST: {reason}. Remaining requested seeds are skipped.",
                            flush=True,
                        )
                        break
                continue

            started = time.time()
            plan = make_stream_plan(args, fraction, seed, train_clients, test_clients)
            calibration_report_cache = build_or_load_calibration_report_cache(
                args,
                fraction,
                seed,
                initial_state,
                plan,
                train_x,
                train_y,
                private_epsilons,
                device,
                output / "calibration_cache",
            )
            seed_curves: List[pd.DataFrame] = []
            seed_summaries: List[dict] = []
            seed_decisions: List[pd.DataFrame] = []
            for policy_index, (name, kind, epsilon) in enumerate(specs, start=1):
                policy_tag = safe_filename_token(name)
                policy_prefix = f"v15_f{fraction_tag:04d}_s{seed}_{policy_tag}"
                policy_curve_path = per_policy_dir / f"{policy_prefix}_curve.csv"
                policy_summary_path = per_policy_dir / f"{policy_prefix}_summary.csv"
                policy_decision_path = per_policy_dir / f"{policy_prefix}_decisions.csv"
                policy_cached = (
                    args.checkpoint_policies
                    and policy_curve_path.exists()
                    and policy_summary_path.exists()
                    and ((not args.save_decisions) or policy_decision_path.exists())
                )
                if policy_cached and not args.rerun_completed:
                    seed_curves.append(pd.read_csv(policy_curve_path))
                    cached_summary = pd.read_csv(policy_summary_path)
                    if len(cached_summary) != 1:
                        raise RuntimeError(
                            f"Invalid policy summary cache: {policy_summary_path}"
                        )
                    seed_summaries.append(cached_summary.iloc[0].to_dict())
                    if args.save_decisions:
                        seed_decisions.append(pd.read_csv(policy_decision_path))
                    print(
                        f"  fraction={fraction:g} seed={seed} policy "
                        f"{policy_index:02d}/{len(specs)}: {name} [cached]",
                        flush=True,
                    )
                    continue

                print(
                    f"  fraction={fraction:g} seed={seed} policy {policy_index:02d}/{len(specs)}: {name}",
                    flush=True,
                )
                curves, summary, decisions = run_one_policy(
                    name, kind, epsilon, args, fraction, seed, initial_state,
                    plan, train_x, train_y, test_x, test_y,
                    args.loss_scale, channels, grid, device,
                    calibration_report_cache,
                )
                seed_curves.append(curves)
                seed_summaries.append(summary)
                if args.save_decisions:
                    seed_decisions.append(decisions)
                if args.checkpoint_policies:
                    atomic_to_csv(curves, policy_curve_path)
                    atomic_to_csv(pd.DataFrame([summary]), policy_summary_path)
                    if args.save_decisions:
                        atomic_to_csv(decisions, policy_decision_path)
                    print(
                        f"    checkpoint saved: {policy_prefix}",
                        flush=True,
                    )

            seed_curve_df = pd.concat(seed_curves, ignore_index=True)
            seed_summary_df = add_recovery_metrics(
                seed_curve_df, pd.DataFrame(seed_summaries), args,
            )
            atomic_to_csv(seed_curve_df, curve_path)
            atomic_to_csv(seed_summary_df, summary_path)
            all_curves.append(seed_curve_df)
            all_summaries.append(seed_summary_df)
            if args.save_decisions:
                seed_decision_df = pd.concat(seed_decisions, ignore_index=True)
                atomic_to_csv(seed_decision_df, decision_path)
                all_decisions.append(seed_decision_df)
            print(
                f"fraction={fraction:g} seed={seed} complete in {(time.time() - started) / 60:.1f} min",
                flush=True,
            )
            if args.fail_fast_screen and seed_index < args.num_stream_seeds:
                reason = fail_fast_screen_reason(
                    pd.concat(all_summaries, ignore_index=True), args,
                )
                if reason is not None:
                    print(
                        f"FAIL-FAST: {reason}. Remaining requested seeds are skipped.",
                        flush=True,
                    )
                    break

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    atomic_to_csv(curves_df, output / "v15_all_curves.csv")
    atomic_to_csv(summary_df, output / "v15_per_seed_summary.csv")
    if args.save_decisions:
        atomic_to_csv(
            pd.concat(all_decisions, ignore_index=True),
            output / "v15_all_decisions.csv",
        )

    primary = summary_df[np.isclose(summary_df.changed_fraction, args.primary_fraction)].copy()
    if primary.empty:
        raise RuntimeError("No rows match --primary_fraction")
    metrics = [
        "identity_auprc_post", "identity_auroc_post", "recall_at_b_post",
        "false_selection_rate_post", "mean_first_selection_delay",
        "mean_global_acc_post", "mean_changed_acc_post", "mean_stable_acc_post",
        "balanced_group_acc_post", "worst_group_acc_post",
        "mean_changed_loss_post", "mean_stable_loss_post", "recovered",
        "client_rounds_to_recovery", "total_bits_to_recovery",
        "uplink_bits_to_recovery", "downlink_bits_to_recovery",
        "bidirectional_bits_to_recovery", "final_uplink_bits",
        "final_bidirectional_bits",
        "total_selection_seconds", "mean_selection_milliseconds",
        "total_monitoring_seconds", "total_adaptation_seconds",
        "oracle_normalized_loss_recovery",
        "oracle_normalized_balanced_recovery",
        "changed_loss_regret_vs_oracle",
        "balanced_acc_gap_vs_oracle",
        "stable_acc_drop_vs_random",
        "cumulative_changed_loss_regret_vs_oracle",
    ]
    primary_summary = summarize_primary(primary, metrics, args.bootstrap_runs)
    pairwise = paired_comparisons(primary, args)
    atomic_to_csv(primary_summary, output / "v15_primary_policy_summary.csv")
    atomic_to_csv(pairwise, output / "v15_pairwise_primary.csv")
    leaderboard = development_leaderboard(primary, args)
    atomic_to_csv(leaderboard, output / "v15_formal_leaderboard.csv")

    label = epsilon_label(args.primary_epsilon)
    proposed_name = f"mcglr_top2_ogpm_{label}"
    proposed_row = primary[primary.policy == proposed_name]
    report_lines = [
        "V15 MC-PrivGLR 正式公开基线比较说明",
        "=" * 88,
        f"版本: {VERSION}",
        f"数据集: {args.dataset.upper()} | paired seeds={args.seed}--{args.seed + args.num_stream_seeds - 1}",
        f"主设置: changed={100 * args.primary_fraction:g}%, epsilon={epsilon_label(args.primary_epsilon)}, N={args.num_clients}, B={args.budget_b}",
        f"联邦: {args.local_steps}步独立本地SGD -> 按客户端样本量加权FedAvg",
        f"监控: 连续窗口平均损失 -> l/(l+{args.loss_scale:g}) -> classical OGPM -> {args.report_levels}级输出量化",
        f"私有证据: 对预先固定的变化幅度{args.mcglr_shifts}分别使用exact OGPM emission计算LLR",
        "序贯状态: 每个客户端维护多条反射GLR图表，避免先混合变化幅度稀释证据",
        "冻结方法: MC-PrivGLR-top2；正式实验不再比较或选择其他MC-PrivGLR变体",
        "闭环: 每个策略拥有独立动态模型，当前选择会改变模型、下一轮监控报告和状态转移",
        f"本次运行策略: {', '.join(name for name, _, _ in specs)}",
        "公开基线: Power-of-Choice-OGPM、Oort-OGPM、HiCS-FL(source-adapted)",
        "边界控制: Random 与 Oracle；它们不是文献SOTA基线",
        "",
        "隐私口径: 每个监控窗口报告分别满足 event-level epsilon-LDP；时间轨迹会组合，不宣称总预算仍为epsilon。",
        "统一接口: 所有非Oracle策略收到相同OGPM报告；只有已选客户端上传标准FedAvg模型更新。",
        "主指标: changed loss、balanced accuracy、Oracle-normalized recovery、相对Oracle适应遗憾和恢复上传量。",
        "身份AUPRC/Recall@B是诊断指标；主判定仍是changed loss和balanced accuracy。",
        "",
    ]
    if proposed_name is not None and not proposed_row.empty:
        report_lines.extend([
            f"冻结的正式方法: {proposed_name}",
            f"当前已完成 paired seeds: {proposed_row.seed.nunique()}",
            f"Proposed mean identity AUPRC: {proposed_row.identity_auprc_post.mean():.6f}",
            f"Proposed mean Recall@B: {proposed_row.recall_at_b_post.mean():.6f}",
            f"Proposed mean balanced accuracy: {proposed_row.balanced_group_acc_post.mean():.6f}",
            f"Proposed mean changed loss: {proposed_row.mean_changed_loss_post.mean():.6f}",
        ])
    report_path = output / "V15_正式公开基线比较说明.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n" + "\n".join(report_lines))
    print("\nKey outputs:")
    for path in [
        report_path,
        output / "v15_primary_policy_summary.csv",
        output / "v15_pairwise_primary.csv",
        output / "v15_method_registry.csv",
        output / "v15_formal_leaderboard.csv",
        output / "v15_per_seed_summary.csv",
        output / "v15_all_curves.csv",
        output / "v15_privacy_channel_diagnostics.csv",
        output / "v15_config.json",
    ]:
        print(" ", path.resolve())
    print("\nV15 formal leaderboard (lower changed loss is better):")
    display_columns = [
        "screen_rank", "policy", "mean_changed_loss_post",
        "mean_changed_acc_post", "balanced_group_acc_post",
        "mean_stable_acc_post", "recall_at_b_post",
    ]
    print(leaderboard[display_columns].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "V15 frozen MC-PrivGLR comparison with source-backed published "
            "client-selection baselines under one OGPM/FedAvg protocol"
        ),
    )
    # Environment and output
    parser.add_argument("--root", type=str, default=r".\data")
    parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "femnist"],
        default="cifar10",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument(
        "--data_seed", type=int, default=2026,
        help=(
            "Fixed client-partition/writer-selection seed. Keep this identical "
            "across policies and stream seeds."
        ),
    )
    parser.add_argument("--num_stream_seeds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--out_dir", type=str, default=r"results\v15_official_cifar10")
    parser.add_argument("--warmup_ckpt", type=str, default=r"results\v15_shared\cifar10_resnet18_fedavg.pt")
    parser.add_argument(
        "--official_hics_root", type=str, default=r"baselines\HiCS-FL",
        help="Clone of https://github.com/CityChan/HiCS-FL at a recorded commit.",
    )
    parser.add_argument(
        "--official_hics_commit", type=str, default="f1dc34e",
        help="Commit recorded in v15_config.json; verify with git rev-parse HEAD.",
    )
    parser.add_argument(
        "--official_fedgcs_root", type=str, default=r"baselines\FedGCS",
        help="Clone of https://github.com/zhiyuan-ning/GenerativeFL.",
    )
    parser.add_argument(
        "--official_fedgcs_commit", type=str,
        default="785773b2a6675048aa3ab2ac7aaf0324b802f368",
        help="Exact official FedGCS commit required by the adapter.",
    )
    parser.add_argument("--rerun_completed", action="store_true")
    parser.add_argument(
        "--policy_subset", nargs="+", default=["paper"],
        help=(
            "paper, screen, all, or exact policy/controller tokens. The paper "
            "set contains Random, Oracle, PoC-OGPM, Oort-OGPM, HiCS-FL, "
            "FedGCS-OGPM, and "
            "the frozen MC-PrivGLR-top2 proposal."
        ),
    )
    parser.add_argument(
        "--fail_fast_screen", action="store_true",
        help="Stop later requested seeds after a predeclared large early loss.",
    )
    parser.add_argument("--checkpoint_policies", action="store_true", default=True)
    parser.add_argument(
        "--no_checkpoint_policies",
        action="store_false",
        dest="checkpoint_policies",
    )
    parser.add_argument("--allow_config_mismatch", action="store_true")
    parser.add_argument("--rebuild_warmup", action="store_true")
    parser.add_argument("--rebuild_calibration_cache", action="store_true")
    parser.add_argument("--rebuild_femnist_cache", action="store_true")
    parser.add_argument("--save_decisions", action="store_true", default=True)
    parser.add_argument("--no_save_decisions", action="store_false", dest="save_decisions")
    parser.add_argument("--self_test", action="store_true")

    # FL stream
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--change_time", type=int, default=30)
    parser.add_argument("--calib_end", type=int, default=30)
    parser.add_argument("--async_window", type=int, default=15)
    parser.add_argument("--ramp_windows", type=int, default=10)
    parser.add_argument("--changed_fractions", type=float, nargs="+", default=[0.05])
    parser.add_argument("--primary_fraction", type=float, default=0.05)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--min_client_samples", type=int, default=40)
    parser.add_argument(
        "--femnist_path", type=str, default="",
        help=(
            "Optional local save_to_disk directory or parquet path. If empty, "
            "the runner searches root/femnist_hf, femnist_processed, hf_cache."
        ),
    )
    parser.add_argument("--femnist_min_writer_samples", type=int, default=120)
    parser.add_argument("--femnist_test_fraction", type=float, default=0.20)
    parser.add_argument("--window_batch", type=int, default=32)
    parser.add_argument("--local_batch", type=int, default=64)
    parser.add_argument("--eval_client_batch", type=int, default=64)
    parser.add_argument("--budget_b", "--budget_k", dest="budget_b", type=int, default=5)
    parser.add_argument("--local_steps", type=int, default=5)
    parser.add_argument(
        "--fedavg_weighting", choices=["samples", "uniform"], default="samples",
    )
    parser.add_argument(
        "--adapt_during_calibration", action="store_true",
        help="Train during the calibration prefix (off by default for a stationary baseline).",
    )

    # Warm-up.  fedavg is the formal default; central is smoke-test only.
    parser.add_argument("--public_calib_size", type=int, default=2000)
    parser.add_argument("--warmup_mode", choices=["fedavg", "central"], default="fedavg")
    parser.add_argument("--warmup_rounds", type=int, default=80)
    parser.add_argument("--warmup_clients_per_round", type=int, default=10)
    parser.add_argument("--warmup_local_steps", type=int, default=5)
    parser.add_argument("--warmup_log_every", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--warmup_batch_size", type=int, default=128)
    parser.add_argument("--warmup_lr", type=float, default=0.05)
    parser.add_argument("--warmup_momentum", type=float, default=0.9)
    parser.add_argument("--warmup_weight_decay", type=float, default=5e-4)
    parser.add_argument("--infer_batch_size", type=int, default=2048)
    parser.add_argument(
        "--loss_scale", type=float, default=None,
        help="Bounded-loss scale; default is log(number of classes).",
    )

    # OGPM report
    parser.add_argument("--epsilons", type=str, nargs="+", default=["1.0"])
    parser.add_argument("--primary_epsilon", type=str, default="1.0")
    parser.add_argument("--report_levels", "--num_categories", dest="report_levels", type=int, default=16)
    parser.add_argument("--grid_size", type=int, default=65)
    parser.add_argument("--ogpm_ldp_tolerance", type=float, default=1e-9)
    parser.add_argument("--emission_floor", type=float, default=1e-10)
    parser.add_argument("--baseline_shrinkage", type=float, default=8.0)
    parser.add_argument("--baseline_sigma", type=float, default=0.05)
    parser.add_argument("--drift_shifts", type=float, nargs="+", default=[0.10, 0.20, 0.35])

    # Historical likelihood baselines only. PrivMAV does not use drift_shifts.
    parser.add_argument("--belief_prior", type=float, default=0.05)
    parser.add_argument("--belief_hazard", type=float, default=0.0025)
    parser.add_argument("--persist_if_untrained", type=float, default=0.995)
    parser.add_argument("--severity_ema", type=float, default=0.25)
    parser.add_argument("--severity_weight", type=float, default=0.75)
    parser.add_argument("--age_weight", type=float, default=0.0)
    parser.add_argument("--age_cap", type=float, default=20.0)
    parser.add_argument("--cusum_drift", type=float, default=0.05)
    parser.add_argument("--avs_candidate_multiplier", type=int, default=4)
    parser.add_argument("--poc_candidate_multiplier", type=int, default=2)
    parser.add_argument(
        "--oort_warmup_rounds", type=int, default=25,
        help="Official HiCS-FL Oort benchmark uses random selection for 25 rounds.",
    )
    parser.add_argument("--hics_temperature", type=float, default=0.001)
    parser.add_argument("--hics_lambda", type=float, default=10.0)
    parser.add_argument("--hics_gamma", type=float, default=4.0)
    parser.add_argument("--hics_groups", type=int, default=10)

    # FedGCS: exact official AUTOS encoder/predictor/decoder, with a common
    # private-report record interface replacing the authors' Plato evaluator.
    parser.add_argument("--fedgcs_candidates_per_round", type=int, default=32)
    parser.add_argument("--fedgcs_replay_size", type=int, default=512)
    parser.add_argument("--fedgcs_train_every", type=int, default=10)
    parser.add_argument("--fedgcs_train_epochs", type=int, default=3)
    parser.add_argument("--fedgcs_batch_size", type=int, default=64)
    parser.add_argument("--fedgcs_lr", type=float, default=1e-3)
    parser.add_argument("--fedgcs_trade_off", type=float, default=0.8)
    parser.add_argument("--fedgcs_hidden_size", type=int, default=64)
    parser.add_argument("--fedgcs_mlp_hidden_size", type=int, default=128)
    parser.add_argument("--fedgcs_top_k", type=int, default=16)
    parser.add_argument(
        "--fedgcs_gradient_steps", type=float, nargs="+",
        default=[1.0, 2.0, 3.0],
    )

    # PrivMAV continuous posterior and action-value model
    parser.add_argument("--latent_process_sigma", type=float, default=0.025)
    parser.add_argument("--latent_hazard", type=float, default=0.0025)
    parser.add_argument("--need_margin", type=float, default=0.02)
    parser.add_argument("--response_prior_mean", type=float, default=0.50)
    parser.add_argument("--response_prior_strength", type=float, default=4.0)
    parser.add_argument("--response_forgetting", type=float, default=0.98)
    parser.add_argument("--response_need_floor", type=float, default=0.02)
    parser.add_argument("--response_confidence_scale", type=float, default=0.10)
    parser.add_argument("--include_ablations", action="store_true")

    # PSPA: one stable state plus persistent positive-shift hypotheses.
    parser.add_argument(
        "--pspa_shifts", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
    )
    parser.add_argument("--pspa_prior", type=float, default=0.05)
    parser.add_argument("--pspa_persistence", type=float, default=0.995)
    parser.add_argument("--pspa_hazard", type=float, default=0.0025)

    # MURE: our multi-timescale private residual-evidence state recursion.
    parser.add_argument(
        "--mure_decays", type=float, nargs="+",
        default=[0.50, 0.80, 0.95],
        help="Short/medium/long public evidence-memory factors.",
    )
    parser.add_argument(
        "--mure_drift_margin", type=float, default=0.02,
        help="Small positive margin subtracted from every centered innovation.",
    )
    parser.add_argument(
        "--mure_consistency_penalty", type=float, default=0.50,
        help="Penalty on disagreement between standardized memory scales.",
    )
    parser.add_argument(
        "--mure_softmin_beta", type=float, default=2.0,
        help="Temperature inverse for the soft-min development candidate.",
    )

    # CAPS: fixed action-controlled persistence values screened as one family.
    parser.add_argument(
        "--caps_retentions", type=float, nargs="+",
        default=[0.98, 0.95, 0.90, 0.80],
        help=(
            "Fraction of previous unresolved-state probability retained after "
            "a client is trained. The value is fixed before confirmation."
        ),
    )

    # MC-PrivGLR: public shift grid, exact OGPM emissions, parallel charts.
    parser.add_argument(
        "--mcglr_shifts", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
        help="Predeclared positive normalized-loss shifts for the GLR charts.",
    )
    parser.add_argument(
        "--mcglr_drift", type=float, default=0.02,
        help="Public nonnegative drift margin subtracted from every chart update.",
    )
    parser.add_argument(
        "--mcglr_softmax_beta", type=float, default=2.0,
        help="Inverse temperature for the softmax chart aggregation candidate.",
    )

    # Adaptation and evaluation
    parser.add_argument("--adapt_lr", type=float, default=0.01)
    parser.add_argument("--adapt_momentum", type=float, default=0.0)
    parser.add_argument("--adapt_weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--monitor_chunk_clients", type=int, default=20)
    parser.add_argument("--eval_chunk_clients", type=int, default=10)
    parser.add_argument("--recovery_ratio", type=float, default=0.80)
    parser.add_argument("--recovery_patience", type=int, default=2)
    parser.add_argument("--oracle_final_points", type=int, default=3)
    parser.add_argument("--bootstrap_runs", type=int, default=500)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
