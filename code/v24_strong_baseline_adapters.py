# -*- coding: utf-8 -*-
"""
V24 stronger-but-principled recent baseline adapters.

This file freezes two baselines BEFORE looking at V24 formal results.

1) Adaptive-OSMD-NativeGrad-OGPM (JMLR 2025, source-backed)
   - verifies the frozen official FL-Client-Sampling source commit;
   - directly extracts and executes the official `lr_set_fun` and
     `Adaptive_OSMD_Expert` functions;
   - keeps the source feedback semantics
         a_i = lambda_i^2 * ||g_i||_2^2,
     but uses the common public gradient clipping bound C to form the bounded
     local statistic min(||g_i||_2^2 / C^2, 1);
   - ONLY selected-client native feedback is privatized with the same official
     epsilon-LDP Optimal-GPM channel before entering the OSMD recursion;
   - no raw unselected-client gradients, raw losses, or drift identities are
     exposed to the selector;
   - fixed B uses distinct-client PPS sampling without replacement because the
     benchmark counts distinct model uploads.

2) FEROMA-Cal-OGPM-EMA (ICLR 2026, official-source-grounded adaptation)
   - checks the official FEROMA source for the latent mean/std descriptor and
     Euclidean-distance primitives used by the adaptation;
   - each client locally forms an extended latent profile [mean, std];
   - the first `calib_end` windows are local-only calibration: the profile
     reference is the calibration mean and q_i is the per-client 95th
     percentile of Euclidean distance to that frozen reference;
   - after calibration, the bounded local profile-shift statistic is
         s_i,t = clip(||z_i,t-r_i||_2 / q_i, 0, 1);
   - ONLY this one bounded scalar per client/window is released through the
     same official epsilon-LDP Optimal-GPM channel;
   - the server applies a predeclared EMA (alpha=0.20) to the private reports
     and selects Top-B;
   - raw profiles, raw distances, references, and q_i remain local simulation
     state and are never passed to the server-side selector.

Both transformations are source-adapted and must be labelled as such in the
paper.  V24 is the final frozen strong-baseline version: no result-dependent
retuning is intended after the formal five-seed run.
"""

from __future__ import annotations

import ast
import math
import subprocess
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


def _git_head(root: Path) -> str:
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
        raise RuntimeError(f"Cannot verify source commit in {root}: {exc}") from exc


def _verify_commit(root: str | Path, expected_commit: str) -> Path:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Source repository not found: {root}")
    actual = _git_head(root)
    expected = str(expected_commit).strip()
    if expected and not actual.startswith(expected):
        raise RuntimeError(
            f"Source commit mismatch for {root.name}: expected {expected}, found {actual}"
        )
    return root


def _extract_source_functions(path: Path, names: Sequence[str]) -> dict:
    """Compile only named top-level functions from a verified source file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    wanted = set(names)
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    found = {node.name for node in nodes}
    missing = wanted - found
    if missing:
        raise RuntimeError(
            f"Official source {path} is missing function(s): {sorted(missing)}"
        )
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "math": math}
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class AdaptiveOSMDNativeSignalLDPAdapter:
    """JMLR-2025 Adaptive-OSMD with native selected-client feedback under LDP."""

    EXPECTED_FILE = Path("Real_Data") / "algorithms_cv.py"

    def __init__(
        self,
        source_root: str | Path,
        expected_commit: str,
        num_clients: int,
        budget: int,
        report_levels: int,
        client_sizes: Sequence[float],
        horizon: int,
        seed: int,
        *,
        alpha: float = 0.4,
    ) -> None:
        self.root = _verify_commit(source_root, expected_commit)
        source_file = self.root / self.EXPECTED_FILE
        if not source_file.is_file():
            raise FileNotFoundError(f"Adaptive-OSMD source missing: {source_file}")

        source_text = source_file.read_text(encoding="utf-8", errors="ignore")
        # Fail loudly if the frozen source no longer contains the feedback
        # structure audited for V24.
        for token in ("a_choose", "grad_norm", "lambda_train", "Adaptive_OSMD_Expert"):
            if token not in source_text:
                raise RuntimeError(
                    f"Adaptive-OSMD source no longer contains audited token: {token}"
                )

        funcs = _extract_source_functions(
            source_file, ("lr_set_fun", "Adaptive_OSMD_Expert")
        )
        self._lr_set_fun = funcs["lr_set_fun"]
        self._expert_update = funcs["Adaptive_OSMD_Expert"]

        self.num_clients = int(num_clients)
        self.budget = int(budget)
        self.report_levels = int(report_levels)
        self.horizon = int(max(1, horizon))
        self.alpha = float(alpha)
        if not (1 <= self.budget <= self.num_clients):
            raise ValueError("Invalid fixed client budget")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("Adaptive-OSMD alpha must be in (0,1]")

        sizes = np.asarray(client_sizes, dtype=np.float64)
        if sizes.shape != (self.num_clients,):
            raise ValueError("client_sizes has unexpected shape")
        if np.any(sizes <= 0):
            raise ValueError("Every client must have positive training size")
        self.client_weight = sizes / sizes.sum()
        self.rng = np.random.default_rng(int(seed))

        # After public clipping and normalization, z_i in [0,1]. Therefore the
        # exact source coefficient a_i=lambda_i^2*z_i is bounded by max lambda_i^2.
        self.bar_a1 = float(max(np.max(self.client_weight ** 2), 1e-15))
        self.lr_set = np.asarray(
            self._lr_set_fun(
                self.budget,
                self.alpha,
                self.num_clients,
                self.bar_a1,
                self.horizon,
            ),
            dtype=np.float64,
        )
        if self.lr_set.size == 0 or not np.all(np.isfinite(self.lr_set)):
            raise RuntimeError("Official Adaptive-OSMD learning-rate grid is invalid")

        self.num_experts = int(self.lr_set.size)
        self.theta = (
            (1.0 + 1.0 / self.num_experts)
            / (
                np.arange(1, self.num_experts + 1, dtype=np.float64)
                * np.arange(2, self.num_experts + 2, dtype=np.float64)
            )
        )
        self.theta /= self.theta.sum()
        self.prob_experts = np.full(
            (self.num_experts, self.num_clients),
            1.0 / self.num_clients,
            dtype=np.float64,
        )
        self.gamma = (
            (self.alpha / self.num_clients)
            * math.sqrt(
                8.0 * self.budget
                / max(self.horizon * self.bar_a1, 1e-15)
            )
        )
        self.last_prob = np.full(
            self.num_clients, 1.0 / self.num_clients, dtype=np.float64
        )
        self.round = 0

    def probability_vector(self) -> np.ndarray:
        p = (self.theta[:, None] * self.prob_experts).sum(axis=0)
        p = np.asarray(p, dtype=np.float64)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.maximum(p, 1e-15)
        p /= p.sum()
        self.last_prob = p.copy()
        return p

    def select(
        self,
        tie: np.ndarray,
        common_random: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        del tie, common_random
        p = self.probability_vector()
        # Fixed-B FL requires B distinct model uploads.  This is the only
        # sampling-interface adaptation to the source np.random.choice call.
        selected = self.rng.choice(
            self.num_clients,
            size=self.budget,
            replace=False,
            p=p,
        ).astype(np.int64)
        self.round += 1
        return selected, p.copy()

    def observe_private_bins(
        self,
        selected: Sequence[int],
        private_bins: np.ndarray,
    ) -> None:
        """Run the official OSMD recursion on privatized native feedback."""
        selected = np.asarray(selected, dtype=np.int64)
        if selected.size == 0:
            return
        bins = np.asarray(private_bins, dtype=np.float64).reshape(-1)
        if bins.shape != (selected.size,):
            raise ValueError(
                f"private_bins shape {bins.shape} != ({selected.size},)"
            )
        z = np.clip(
            (bins + 0.5) / float(self.report_levels), 0.0, 1.0
        )

        # Exact source semantics after privacy preprocessing:
        #   a_i = lambda_i^2 * ||g_i||^2
        # with ||g_i||^2 replaced by its bounded, privatized version z_i.
        a_choose = (self.client_weight[selected] ** 2) * z
        a_choose = np.clip(a_choose, 0.0, self.bar_a1)

        p_mix = np.maximum(self.last_prob, 1e-15)
        l_hat = np.zeros(self.num_experts, dtype=np.float64)
        grad_l_hat = np.zeros(
            (self.num_experts, self.num_clients), dtype=np.float64
        )
        K = float(self.budget)
        for e in range(self.num_experts):
            pe = np.maximum(self.prob_experts[e], 1e-15)
            for k, client in enumerate(selected.tolist()):
                l_hat[e] += (
                    a_choose[k]
                    / max((K ** 2) * pe[client] * p_mix[client], 1e-15)
                )
                grad_l_hat[e, client] = (
                    -a_choose[k]
                    / max(
                        (K ** 2) * (pe[client] ** 2) * p_mix[client],
                        1e-15,
                    )
                )

        for e in range(self.num_experts):
            updated = self._expert_update(
                self.prob_experts[e].copy(),
                selected,
                grad_l_hat[e],
                float(self.lr_set[e]),
                self.budget,
                self.alpha,
            )
            updated = np.asarray(updated, dtype=np.float64)
            if updated.shape != (self.num_clients,) or not np.all(np.isfinite(updated)):
                raise RuntimeError(
                    "Official Adaptive_OSMD_Expert produced invalid probabilities"
                )
            updated = np.maximum(updated, 1e-15)
            self.prob_experts[e] = updated / updated.sum()

        # Exact source meta-expert exponential weighting, evaluated stably.
        log_weight = np.log(np.maximum(self.theta, 1e-300)) - self.gamma * l_hat
        log_weight -= np.max(log_weight)
        new_theta = np.exp(np.clip(log_weight, -700.0, 0.0))
        if not np.isfinite(new_theta).all() or new_theta.sum() <= 0:
            self.theta[:] = 1.0 / self.num_experts
        else:
            self.theta = new_theta / new_theta.sum()

    def score_vector(self) -> np.ndarray:
        return self.probability_vector()


class FeromaCalibratedProfileOGPMEMAAdapter:
    """Strong source-grounded FEROMA profile scheduler under common OGPM-LDP."""

    REQUIRED_FILES = (
        Path("feroma") / "server.py",
        Path("feroma") / "client.py",
        Path("public") / "models.py",
    )

    def __init__(
        self,
        source_root: str | Path,
        num_clients: int,
        budget: int,
        report_levels: int,
        *,
        calibration_quantile: float = 0.95,
        ema_alpha: float = 0.20,
    ) -> None:
        self.root = Path(source_root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"FEROMA source repository not found: {self.root}")
        for rel in self.REQUIRED_FILES:
            path = self.root / rel
            if not path.is_file():
                raise FileNotFoundError(f"FEROMA source file missing: {path}")

        server_text = (self.root / "feroma" / "server.py").read_text(
            encoding="utf-8", errors="ignore"
        )
        models_text = (self.root / "public" / "models.py").read_text(
            encoding="utf-8", errors="ignore"
        )
        for token in ("distance_function", "euclidean"):
            if token not in server_text:
                raise RuntimeError(
                    f"FEROMA source no longer exposes audited server primitive: {token}"
                )
        for token in ("latent_space_mean", "latent_space_std"):
            if token not in models_text:
                raise RuntimeError(
                    f"FEROMA source no longer exposes audited descriptor: {token}"
                )

        self.num_clients = int(num_clients)
        self.budget = int(budget)
        self.report_levels = int(report_levels)
        self.calibration_quantile = float(calibration_quantile)
        self.ema_alpha = float(ema_alpha)
        if not (1 <= self.budget <= self.num_clients):
            raise ValueError("Invalid fixed client budget")
        if not (0.5 <= self.calibration_quantile < 1.0):
            raise ValueError("FEROMA calibration quantile must be in [0.5,1)")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("FEROMA EMA alpha must be in (0,1]")

        self._calibration_profiles: list[np.ndarray] = []
        self.reference: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.calibration_finalized = False
        self.ema_initialized = False
        self.ema_score = np.zeros(self.num_clients, dtype=np.float64)
        self.last_private_instant = np.zeros(self.num_clients, dtype=np.float64)
        self.last_raw_shift = np.zeros(self.num_clients, dtype=np.float64)

    def _check_profiles(self, profiles: np.ndarray) -> np.ndarray:
        p = np.asarray(profiles, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] != self.num_clients:
            raise ValueError("profiles must have shape [num_clients, feature_dim]")
        if not np.all(np.isfinite(p)):
            raise ValueError("FEROMA profiles contain non-finite values")
        return p

    def record_calibration_profiles(self, profiles: np.ndarray) -> None:
        """Store client-local calibration profiles; nothing is released."""
        if self.calibration_finalized:
            raise RuntimeError("FEROMA calibration is already frozen")
        self._calibration_profiles.append(self._check_profiles(profiles).copy())

    def finalize_calibration(self) -> None:
        """Freeze r_i and q_i using only the calibration prefix."""
        if self.calibration_finalized:
            return
        if len(self._calibration_profiles) < 2:
            raise RuntimeError("FEROMA requires at least two calibration windows")
        stack = np.stack(self._calibration_profiles, axis=0)  # [Tc,N,D]
        self.reference = stack.mean(axis=0)
        distances = np.linalg.norm(stack - self.reference[None, :, :], axis=2)
        q = np.quantile(distances, self.calibration_quantile, axis=0)
        # Numerical floor only; it is not data/result tuning.
        self.scale = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
        self.calibration_finalized = True
        # Delete raw calibration descriptors once the local calibration state is frozen.
        self._calibration_profiles.clear()

    def bounded_profile_shift(self, profiles: np.ndarray) -> np.ndarray:
        """Client-local extended-profile Euclidean shift mapped to [0,1]."""
        if not self.calibration_finalized:
            self.finalize_calibration()
        assert self.reference is not None and self.scale is not None
        p = self._check_profiles(profiles)
        distance = np.linalg.norm(p - self.reference, axis=1)
        shift = np.clip(distance / self.scale, 0.0, 1.0)
        self.last_raw_shift = shift.copy()
        return shift

    def select(
        self,
        report_bins: np.ndarray,
        tie: np.ndarray,
        common_random: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        del common_random
        reports = np.asarray(report_bins, dtype=np.float64)
        if reports.shape != (self.num_clients,):
            raise ValueError("FEROMA-OGPM report vector has unexpected shape")
        instant = np.clip(
            (reports + 0.5) / float(self.report_levels), 0.0, 1.0
        )
        self.last_private_instant = instant.copy()
        if not self.ema_initialized:
            self.ema_score = instant.copy()
            self.ema_initialized = True
        else:
            a = self.ema_alpha
            self.ema_score = (1.0 - a) * self.ema_score + a * instant

        tie = np.asarray(tie, dtype=np.float64)
        order = np.lexsort((tie, -self.ema_score))
        selected = order[: self.budget].astype(np.int64)
        return selected, self.ema_score.copy()

    def score_vector(self) -> np.ndarray:
        return self.ema_score.copy()
