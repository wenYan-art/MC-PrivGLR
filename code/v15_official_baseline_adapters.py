# -*- coding: utf-8 -*-
"""Adapters for source-backed client-selection baselines.

The adapters deliberately contain no data loading, local training, or privacy
mechanism.  They consume only information supplied by the common V14 runner:

* ``OortAdapter`` calls ``UCBsampler.py`` from the official HiCS-FL repository.
  That file is the Oort baseline shipped by the NeurIPS-2024 HiCS-FL artifact.
* ``HiCSAdapter`` imports clustering primitives from the same official artifact
  and follows ``server/server_hics.py`` while keeping the V14 FedAvg/data loop.

Consequently Oort is labelled ``official-benchmark-code`` and HiCS is labelled
``source-adapted`` in every output.  Neither label should be shortened to
"official implementation" in a paper without the qualifier.
"""

from __future__ import annotations

import importlib.util
import importlib
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _load_module(path: Path, module_name: str):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required official source file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official source module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_head(root: Path) -> str:
    """Return HEAD without changing the caller's working directory."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        marker = root / "OFFICIAL_COMMIT.txt"
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip()
        raise RuntimeError(
            f"Cannot verify official source commit in {root}: {exc}"
        ) from exc


@dataclass
class _Node:
    """Minimal attribute namespace expected by the official Plato config."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FedGCSAdapter:
    """Official FedGCS AUTOS architecture on the common private interface.

    FedGCS was published for a static multi-objective client-selection task.
    Its repository is coupled to Plato and an offline ``device_env.pkl``.  That
    environment is not interchangeable with our CIFAR/FEMNIST drift stream.
    This adapter therefore imports the *unchanged* official ``autos.model``
    encoder/predictor/decoder and keeps its two training losses (utility MSE +
    sequence reconstruction NLL), while constructing records from the same
    epsilon-LDP reports available to every private selector.

    This is deliberately reported as ``official-architecture/source-adapted``;
    it is not reported as a bit-for-bit run of the authors' MNIST Plato script.
    No raw monitoring loss, drift identity, test label, or unselected gradient
    is passed to this class.
    """

    EXPECTED_FILES = (
        "autos/model.py",
        "autos/encoder.py",
        "autos/decoder.py",
        "autos/train_utils.py",
    )

    def __init__(
        self,
        source_root: str | Path,
        expected_commit: str,
        num_clients: int,
        budget: int,
        report_levels: int,
        seed: int,
        device: torch.device,
        *,
        candidates_per_round: int = 32,
        replay_size: int = 512,
        train_every: int = 10,
        train_epochs: int = 3,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        trade_off: float = 0.8,
        hidden_size: int = 64,
        mlp_hidden_size: int = 128,
        top_k: int = 16,
        gradient_steps: Sequence[float] = (1.0, 2.0, 3.0),
    ) -> None:
        self.root = Path(source_root).resolve()
        missing = [str(self.root / item) for item in self.EXPECTED_FILES
                   if not (self.root / item).is_file()]
        if missing:
            raise FileNotFoundError(
                "FedGCS official source is incomplete; missing: " + ", ".join(missing)
            )
        actual = _git_head(self.root)
        expected = str(expected_commit).strip()
        if expected and not actual.startswith(expected):
            raise RuntimeError(
                f"FedGCS commit mismatch: expected {expected}, found {actual}"
            )
        if device.type != "cuda":
            raise RuntimeError(
                "The official FedGCS decoder creates CUDA tensors during latent "
                "generation. Run this policy with --gpu >= 0 on a CUDA device."
            )

        self.num_clients = int(num_clients)
        self.budget = int(budget)
        self.report_levels = int(report_levels)
        self.device = device
        self.gpu = int(device.index or 0)
        self.rng = np.random.default_rng(int(seed))
        self.candidates_per_round = int(candidates_per_round)
        self.replay_size = int(replay_size)
        self.train_every = int(train_every)
        self.train_epochs = int(train_epochs)
        self.batch_size = int(batch_size)
        self.trade_off = float(trade_off)
        self.top_k = int(top_k)
        self.gradient_steps = tuple(float(x) for x in gradient_steps)
        self.round = 0
        self.trained_updates = 0
        self.sequences: list[np.ndarray] = []
        self.targets: list[float] = []
        self.last_scores = np.zeros(self.num_clients, dtype=np.float64)

        # The official modules use absolute imports such as ``autos.decoder``.
        # Put the verified source root first only for this import operation.
        source_text = str(self.root)
        inserted = source_text not in sys.path
        if inserted:
            sys.path.insert(0, source_text)
        try:
            official_model = importlib.import_module("autos.model")
        finally:
            if inserted and sys.path and sys.path[0] == source_text:
                sys.path.pop(0)

        autos_cfg = _Node(
            method_name="rnn",
            gpu=self.gpu,
            encoder_layers=1,
            encoder_hidden_size=int(hidden_size),
            encoder_dropout=0.0,
            mlp_layers=2,
            mlp_hidden_size=int(mlp_hidden_size),
            decoder_layers=1,
            decoder_hidden_size=int(hidden_size),
            decoder_dropout=0.0,
        )
        cfg = _Node(
            clients=_Node(total_clients=self.num_clients),
            server=_Node(autos=autos_cfg),
        )

        def config_factory():
            return cfg

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        self.model = official_model.AUTOS(config_factory).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(learning_rate), weight_decay=0.0,
        )

    def _sequence(self, selected: Sequence[int]) -> np.ndarray:
        selected = np.asarray(selected, dtype=np.int64)
        selected = np.asarray(list(dict.fromkeys(selected.tolist())), dtype=np.int64)
        selected = selected[(selected >= 0) & (selected < self.num_clients)]
        if selected.size < self.budget:
            remaining = np.setdiff1d(np.arange(self.num_clients), selected)
            selected = np.concatenate([
                selected,
                self.rng.choice(
                    remaining, size=self.budget - selected.size, replace=False,
                ),
            ])
        out = np.full(self.num_clients, self.num_clients, dtype=np.int64)
        out[: self.budget] = selected[: self.budget]
        return out

    def _subset_from_sequence(self, sequence: Sequence[int]) -> np.ndarray:
        unique: list[int] = []
        for value in np.asarray(sequence, dtype=np.int64).reshape(-1):
            item = int(value)
            if 0 <= item < self.num_clients and item not in unique:
                unique.append(item)
            if len(unique) == self.budget:
                break
        return self._sequence(unique)[: self.budget]

    def _candidate_subsets(self, private_scores: np.ndarray, tie: np.ndarray) -> list[np.ndarray]:
        scores = np.asarray(private_scores, dtype=np.float64)
        tie = np.asarray(tie, dtype=np.float64)
        candidates: list[np.ndarray] = []
        candidates.append(np.lexsort((tie, -scores))[: self.budget])
        candidates.append(np.argsort(tie)[: self.budget])
        ranks = np.argsort(np.argsort(-scores)).astype(np.float64)
        weights = np.exp(-ranks / max(1.0, self.num_clients / 10.0)) + 1e-6
        weights /= weights.sum()
        for _ in range(max(0, self.candidates_per_round - 2)):
            if self.rng.random() < 0.7:
                chosen = self.rng.choice(
                    self.num_clients, self.budget, replace=False, p=weights,
                )
            else:
                chosen = self.rng.choice(
                    self.num_clients, self.budget, replace=False,
                )
            candidates.append(np.asarray(chosen, dtype=np.int64))
        return candidates

    def _utility(self, selected: np.ndarray, private_scores: np.ndarray) -> float:
        # Fixed-latency/fixed-energy benchmark: the only varying statistical
        # utility is the mean common private report of the proposed subset.
        return float(np.mean(np.asarray(private_scores)[selected]))

    def _append_records(self, candidates: list[np.ndarray], scores: np.ndarray) -> None:
        for selected in candidates:
            self.sequences.append(self._sequence(selected))
            self.targets.append(self._utility(selected, scores))
        overflow = len(self.sequences) - self.replay_size
        if overflow > 0:
            del self.sequences[:overflow]
            del self.targets[:overflow]

    def _train(self) -> None:
        if len(self.sequences) < max(16, self.batch_size // 2):
            return
        sequence = torch.as_tensor(
            np.stack(self.sequences), dtype=torch.long, device=self.device,
        )
        target = torch.as_tensor(
            np.asarray(self.targets), dtype=torch.float32, device=self.device,
        )
        order = torch.arange(sequence.size(0), device=self.device)
        self.model.train()
        for _ in range(self.train_epochs):
            order = order[torch.randperm(order.numel(), device=self.device)]
            for start in range(0, order.numel(), self.batch_size):
                ids = order[start:start + self.batch_size]
                encoder_input = sequence[ids]
                decoder_input = torch.cat([
                    torch.full(
                        (ids.numel(), 1), self.num_clients,
                        dtype=torch.long, device=self.device,
                    ),
                    encoder_input[:, :-1],
                ], dim=1)
                predict, log_prob, _ = self.model(encoder_input, decoder_input)
                mse = F.mse_loss(predict.squeeze(-1), target[ids])
                nll = F.nll_loss(
                    log_prob.reshape(-1, log_prob.size(-1)),
                    encoder_input.reshape(-1),
                )
                loss = self.trade_off * mse + (1.0 - self.trade_off) * nll
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
        self.trained_updates += 1

    @torch.no_grad()
    def _predict(self, sequences: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(sequences, dtype=torch.long, device=self.device)
        decoder_input = torch.cat([
            torch.full(
                (x.size(0), 1), self.num_clients,
                dtype=torch.long, device=self.device,
            ), x[:, :-1],
        ], dim=1)
        value, _, _ = self.model(x, decoder_input)
        return value.squeeze(-1).detach().cpu().numpy().astype(np.float64)

    def _generate(self, seeds: np.ndarray) -> list[np.ndarray]:
        if self.trained_updates == 0:
            return []
        self.model.eval()
        x = torch.as_tensor(seeds, dtype=torch.long, device=self.device)
        generated: list[np.ndarray] = []
        # Official gradient-based latent search and decoder are called here.
        for step in self.gradient_steps:
            self.model.zero_grad(set_to_none=True)
            decoded = self.model.generate_new_device(
                x, predict_lambda=float(step), direction="+",
            )
            decoded_np = decoded.squeeze(-1).detach().cpu().numpy()
            for row in decoded_np:
                generated.append(self._sequence(self._subset_from_sequence(row)))
        return generated

    def select(
        self,
        report_bins: np.ndarray,
        tie: np.ndarray,
        common_random: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = (
            np.asarray(report_bins, dtype=np.float64) + 0.5
        ) / float(self.report_levels)
        self.last_scores = scores.copy()
        candidates = self._candidate_subsets(scores, np.asarray(tie))
        self._append_records(candidates, scores)
        if self.round == 0 or (self.round + 1) % self.train_every == 0:
            self._train()

        base_sequences = np.stack([self._sequence(x) for x in candidates])
        pool = [row for row in base_sequences]
        if self.trained_updates:
            predicted = self._predict(base_sequences)
            seed_ids = np.argsort(-predicted)[: min(self.top_k, predicted.size)]
            pool.extend(self._generate(base_sequences[seed_ids]))
        pool_array = np.stack(pool)
        if self.trained_updates:
            best = int(np.argmax(self._predict(pool_array)))
        else:
            utilities = [
                self._utility(self._subset_from_sequence(row), scores)
                for row in pool_array
            ]
            best = int(np.argmax(utilities))
        selected = self._subset_from_sequence(pool_array[best])
        self.round += 1
        return selected, scores


class OortAdapter:
    """Direct wrapper around HiCS-FL's bundled Oort UCB sampler.

    The UCB selection code is not rewritten here.  V14 supplies an OGPM report
    bin as the statistical reward, because raw pre-selection losses are outside
    our privacy interface.  Duration is fixed to one because the benchmark is
    statistical/communication constrained rather than straggler constrained.
    """

    def __init__(
        self,
        source_root: str | Path,
        client_sizes: Sequence[float],
        seed: int,
        config: Optional[Dict[str, float]] = None,
    ) -> None:
        root = Path(source_root)
        module = _load_module(root / "UCBsampler.py", f"hics_oort_{seed}")
        defaults: Dict[str, float] = {
            "exploration_factor": 0.9,
            "exploration_decay": 0.8,
            "exploration_min": 0.2,
            "exploration_alpha": 0.3,
            "round_threshold": 10,
            "sample_window": 5.0,
            "pacer_step": 20,
            "pacer_delta": 5,
            "blacklist_rounds": -1,
            "blacklist_max_len": 0.3,
            "clip_bound": 0.98,
            "round_penalty": 2.0,
            "cut_off_util": 0.7,
        }
        if config:
            defaults.update(config)
        self.sampler = module.UCBsampler(defaults, sample_seed=int(seed))
        self.client_sizes = np.asarray(client_sizes, dtype=np.float64)
        for client, size in enumerate(self.client_sizes.tolist()):
            self.sampler.register_client(
                int(client), {"reward": max(float(size), 1.0), "duration": 1.0},
            )
            self.sampler.update_duration(int(client), 1.0)

    def select(self, budget: int) -> np.ndarray:
        selected = self.sampler.select_participant(
            int(budget), feasible_clients=range(self.client_sizes.size),
        )
        return np.asarray(selected, dtype=np.int64)

    def observe(
        self,
        t: int,
        selected: Sequence[int],
        report_bins: np.ndarray,
        report_levels: int,
    ) -> None:
        reports = np.asarray(report_bins, dtype=np.float64)
        for client in np.asarray(selected, dtype=np.int64):
            private_loss_proxy = (reports[client] + 0.5) / float(report_levels)
            reward = math.sqrt(max(private_loss_proxy, 1e-12)) * max(
                float(self.client_sizes[client]), 1.0,
            )
            self.sampler.update_client_util(
                int(client),
                {
                    "reward": float(reward),
                    "duration": 1.0,
                    "status": True,
                    "time_stamp": int(t) + 1,
                },
            )

    def score_vector(self) -> np.ndarray:
        metrics = self.sampler.getAllMetrics()
        scores = np.full(self.client_sizes.size, -np.inf, dtype=np.float64)
        for client, state in metrics.items():
            count = max(int(state.get("count", 0)), 1)
            staleness = max(
                0, int(self.sampler.training_round) - int(state.get("time_stamp", 0)),
            )
            scores[int(client)] = float(state.get("reward", 0.0)) + math.sqrt(
                math.log(max(2, int(self.sampler.training_round) + 1)) / count,
            ) + 1e-6 * staleness
        return scores


class HiCSAdapter:
    """Source-adapted HiCS-FL selector using official clustering primitives.

    The common runner supplies stale final-layer model deltas only after a
    client has been selected and trained.  This matches the information timing
    of the source method and does not expose any unselected-client gradient.
    """

    def __init__(
        self,
        source_root: str | Path,
        client_sizes: Sequence[float],
        num_classes: int,
        fc_features: int,
        budget: int,
        seed: int,
        temperature: float = 0.001,
        lambda_entropy: float = 10.0,
        gamma: float = 4.0,
        num_groups: int = 10,
        horizon: int = 200,
    ) -> None:
        root = Path(source_root)
        self.clustering = _load_module(
            root / "clustering.py", f"hics_clustering_{seed}",
        )
        self.client_sizes = np.asarray(client_sizes, dtype=np.float64)
        self.weights = self.client_sizes / np.maximum(self.client_sizes.sum(), 1e-15)
        self.num_clients = int(self.client_sizes.size)
        self.num_classes = int(num_classes)
        self.fc_features = int(fc_features)
        self.budget = int(budget)
        self.temperature = float(temperature)
        self.lambda_entropy = float(lambda_entropy)
        self.gamma = float(gamma)
        self.num_groups = int(min(max(2, num_groups), self.num_clients))
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(int(seed))
        self.weight_delta = np.zeros(
            (self.num_clients, self.num_classes, self.fc_features), dtype=np.float64,
        )
        self.bias_delta = np.zeros(
            (self.num_clients, self.num_classes), dtype=np.float64,
        )
        self.observed = np.zeros(self.num_clients, dtype=bool)
        self.last_scores = np.zeros(self.num_clients, dtype=np.float64)
        self.online_round = 0
        self.warmup_rounds = int(math.ceil(self.num_clients / max(1, self.budget)))

    def observe(
        self,
        selected: Sequence[int],
        fc_weight_delta: np.ndarray,
        fc_bias_delta: np.ndarray,
    ) -> None:
        selected = np.asarray(selected, dtype=np.int64)
        weight = np.asarray(fc_weight_delta, dtype=np.float64)
        bias = np.asarray(fc_bias_delta, dtype=np.float64)
        if weight.shape != (selected.size, self.num_classes, self.fc_features):
            raise ValueError(
                "Unexpected fc_weight_delta shape: "
                f"{weight.shape}; expected {(selected.size, self.num_classes, self.fc_features)}"
            )
        if bias.shape != (selected.size, self.num_classes):
            raise ValueError(
                "Unexpected fc_bias_delta shape: "
                f"{bias.shape}; expected {(selected.size, self.num_classes)}"
            )
        self.weight_delta[selected] = weight
        self.bias_delta[selected] = bias
        self.observed[selected] = True

    def _entropy(self) -> np.ndarray:
        # server_hics.py averages each classifier row before applying softmax.
        magnitude = self.weight_delta.mean(axis=2)
        z = magnitude / max(self.temperature, 1e-12)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-15)
        return -np.sum(p * np.log(np.maximum(p, 1e-15)), axis=1)

    def _distance_matrix(self, entropy: np.ndarray) -> np.ndarray:
        gradients = [
            [self.weight_delta[i], self.bias_delta[i]]
            for i in range(self.num_clients)
        ]
        matrix = np.zeros((self.num_clients, self.num_clients), dtype=np.float64)
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                matrix[i, j] = self.clustering.get_similarity(
                    gradients[i], gradients[j], "cosine",
                ) + self.lambda_entropy * abs(float(entropy[i] - entropy[j]))
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=0.0)
        matrix = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def select(
        self,
        common_random: Sequence[int],
        tie: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # The official method first covers all clients once.  Reusing the
        # paired common-random schedule keeps that warm-up identical by policy.
        if self.online_round < self.warmup_rounds or not np.all(self.observed):
            unobserved = np.flatnonzero(~self.observed)
            if unobserved.size:
                if tie is None:
                    selected = unobserved[: self.budget]
                else:
                    selected = unobserved[
                        np.argsort(np.asarray(tie, dtype=np.float64)[unobserved])
                        [: self.budget]
                    ]
                if selected.size < self.budget:
                    remaining = np.setdiff1d(
                        np.asarray(common_random, dtype=np.int64), selected,
                    )
                    selected = np.concatenate(
                        [selected, remaining[: self.budget - selected.size]]
                    )
            else:
                selected = np.asarray(common_random, dtype=np.int64)[: self.budget]
            scores = np.zeros(self.num_clients, dtype=np.float64)
            scores[selected] = 1.0
            self.last_scores = scores
            self.online_round += 1
            return selected, scores

        from scipy.cluster.hierarchy import linkage
        from sklearn.cluster import AgglomerativeClustering

        entropy = self._entropy()
        distance = self._distance_matrix(entropy)
        selected: list[int] = []

        if float(np.var(entropy)) < 0.1:
            groups = min(self.num_groups, self.num_clients)
            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=groups, metric="precomputed", linkage="average",
                )
            except TypeError:  # scikit-learn < 1.2
                clusterer = AgglomerativeClustering(
                    n_clusters=groups, affinity="precomputed", linkage="average",
                )
            labels = clusterer.fit_predict(distance)
            clusters = [np.flatnonzero(labels == group) for group in range(groups)]
            cluster_entropy = np.asarray(
                [entropy[c].mean() if c.size else -np.inf for c in clusters],
                dtype=np.float64,
            )
            progress = self.online_round / max(1, self.horizon - 1)
            logits = self.gamma * (1.0 - progress) * np.nan_to_num(
                cluster_entropy, nan=-1e6, neginf=-1e6,
            )
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= np.maximum(probabilities.sum(), 1e-15)
            capacity = np.zeros(groups, dtype=np.int64)
            for _ in range(self.budget):
                available = np.asarray(
                    [capacity[g] < clusters[g].size for g in range(groups)], dtype=bool,
                )
                p = probabilities * available
                p /= np.maximum(p.sum(), 1e-15)
                group = int(self.rng.choice(groups, p=p))
                choices = np.setdiff1d(clusters[group], np.asarray(selected), assume_unique=False)
                client_weights = self.client_sizes[choices]
                client_weights /= np.maximum(client_weights.sum(), 1e-15)
                selected.append(int(self.rng.choice(choices, p=client_weights)))
                capacity[group] += 1
            scores = entropy.copy()
        else:
            # This is Algorithm 2 + sample_clients from clustering.py.  The
            # official helper uses NumPy's global RNG, so save/restore its state
            # and seed it deterministically for paired reproducibility.
            link = linkage(distance, method="ward")
            distributions = self.clustering.get_clusters_with_alg2(
                link, self.budget, self.weights,
            )
            state = np.random.get_state()
            np.random.seed(int(self.rng.integers(0, 2**31 - 1)))
            try:
                selected = [int(x) for x in self.clustering.sample_clients(distributions)]
            finally:
                np.random.set_state(state)
            scores = entropy.copy()

        # Defensive deduplication without silently changing the budget.
        selected = list(dict.fromkeys(selected))
        if len(selected) < self.budget:
            remaining = np.setdiff1d(
                np.arange(self.num_clients), np.asarray(selected, dtype=np.int64),
            )
            fill = remaining[np.argsort(-entropy[remaining])[: self.budget - len(selected)]]
            selected.extend(int(x) for x in fill)
        selected_array = np.asarray(selected[: self.budget], dtype=np.int64)
        self.last_scores = np.asarray(scores, dtype=np.float64)
        self.online_round += 1
        return selected_array, self.last_scores.copy()
