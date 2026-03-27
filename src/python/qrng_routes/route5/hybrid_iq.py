from __future__ import annotations

from itertools import combinations
import math
from dataclasses import dataclass

import numpy as np

from ..common import (
    SingleDeviceGuessingProblem,
    coherent_state,
    density_from_ket,
    kron,
    operator_span_rank,
    project_density_to_basis,
    support_basis,
)
from ..route3.cv_four_phase import dual_homodyne_probabilities


DEFAULT_RADIUS_VALUES = [0.0, 0.6, 1.2]
DEFAULT_PHASE_VALUES = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
DEFAULT_QUADRATURE_RANGES = [2.0, 3.0, 4.0]
DEFAULT_GAMMA_VALUES = [0.75, 1.0, 1.5]


@dataclass(frozen=True)
class AxisBoundsCandidate:
    num_bins: int
    finite_range: float
    gamma: float
    bounds: tuple[float, ...]


@dataclass(frozen=True)
class AlphabetCandidateSpec:
    radius_values: tuple[float, ...]
    phase_values: tuple[float, ...]
    alpha_values: tuple[complex, ...]


def _canonicalize_scalar(value: float) -> float:
    """Round tiny numerical noise to exact zero for cleaner JSON output."""
    if math.isfinite(value) and abs(value) < 1e-12:
        return 0.0
    return float(value)


def _serialize_complex(alpha: complex) -> dict:
    """JSON-friendly description of a coherent-state amplitude."""
    return {
        "real": float(np.real(alpha)),
        "imag": float(np.imag(alpha)),
        "abs": float(abs(alpha)),
        "phase": float(np.angle(alpha)),
    }


def serialize_complex_list(alphas: list[complex]) -> list[dict]:
    """JSON-friendly list of coherent amplitudes."""
    return [_serialize_complex(alpha) for alpha in alphas]


def _deduplicate_alphas(alphas: list[complex], tol: float = 1e-12) -> list[complex]:
    """Preserve order while removing duplicate coherent amplitudes."""
    unique: list[complex] = []
    for alpha in alphas:
        if any(abs(alpha - existing) <= tol for existing in unique):
            continue
        unique.append(alpha)
    return unique


def _unique_sorted_radii(radius_values: list[float], tol: float = 1e-12) -> list[float]:
    """Deduplicate and sort radii."""
    unique: list[float] = []
    for radius in sorted(float(value) for value in radius_values):
        if any(abs(radius - existing) <= tol for existing in unique):
            continue
        unique.append(radius)
    return unique


def _normalize_phase(phase: float) -> float:
    """Map phases to [0, 2pi) for stable subset enumeration."""
    two_pi = 2.0 * math.pi
    value = float(phase) % two_pi
    if abs(value - two_pi) < 1e-12:
        value = 0.0
    return value


def _unique_sorted_phases(phase_values: list[float], tol: float = 1e-12) -> list[float]:
    """Deduplicate phases modulo 2pi and sort them."""
    unique: list[float] = []
    for phase in sorted(_normalize_phase(value) for value in phase_values):
        if any(abs(phase - existing) <= tol for existing in unique):
            continue
        unique.append(phase)
    return unique


def generalized_coherent_alphabet(
    cutoff: int,
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
) -> tuple[list[np.ndarray], list[complex]]:
    """Local coherent alphabet with arbitrary amplitudes in phase space."""
    if alpha_values is not None and len(alpha_values) == 0:
        raise ValueError("alpha_values cannot be empty.")

    if alpha_values is None:
        radii = DEFAULT_RADIUS_VALUES if radius_values is None else list(radius_values)
        phases = DEFAULT_PHASE_VALUES if phase_values is None else list(phase_values)
        if len(radii) == 0 or len(phases) == 0:
            raise ValueError("radius_values and phase_values must be non-empty.")
        alpha_values = [radius * np.exp(1j * phase) for radius in radii for phase in phases]

    unique_alphas = _deduplicate_alphas([complex(alpha) for alpha in alpha_values])
    states = [density_from_ket(coherent_state(cutoff, alpha)) for alpha in unique_alphas]
    return states, unique_alphas


def _build_alpha_values(radius_values: tuple[float, ...], phase_values: tuple[float, ...]) -> tuple[complex, ...]:
    """Construct a coherent alphabet from a radius/phase specification."""
    alpha_values = [radius * np.exp(1j * phase) for radius in radius_values for phase in phase_values]
    return tuple(_deduplicate_alphas([complex(alpha) for alpha in alpha_values]))


def generate_radius_subsets(
    radius_values: list[float],
    num_radii_values: list[int],
    require_vacuum: bool = True,
) -> list[tuple[float, ...]]:
    """Generate systematic radius subsets for alphabet search."""
    unique_radii = _unique_sorted_radii(radius_values)
    if len(unique_radii) == 0:
        raise ValueError("radius_values cannot be empty.")

    zero_radii = [radius for radius in unique_radii if abs(radius) <= 1e-12]
    has_vacuum = len(zero_radii) > 0
    vacuum = zero_radii[0] if has_vacuum else None
    nonzero_radii = [radius for radius in unique_radii if abs(radius) > 1e-12]

    subsets: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for requested_count in sorted(set(int(value) for value in num_radii_values if int(value) > 0)):
        if require_vacuum and has_vacuum:
            if requested_count == 1:
                subset = (float(vacuum),)
                if subset not in seen:
                    seen.add(subset)
                    subsets.append(subset)
                continue
            choose_count = requested_count - 1
            if choose_count > len(nonzero_radii):
                continue
            for combo in combinations(nonzero_radii, choose_count):
                subset = (float(vacuum),) + tuple(float(value) for value in combo)
                if subset in seen:
                    continue
                seen.add(subset)
                subsets.append(subset)
        else:
            if requested_count > len(unique_radii):
                continue
            for combo in combinations(unique_radii, requested_count):
                subset = tuple(float(value) for value in combo)
                if subset in seen:
                    continue
                seen.add(subset)
                subsets.append(subset)
    return subsets


def _phase_subset_from_offset(phases: list[float], count: int, offset: int) -> tuple[float, ...]:
    """Approximate evenly spaced phase subset with a cyclic offset."""
    total = len(phases)
    chosen: list[int] = []
    used: set[int] = set()
    for index in range(count):
        raw = int(round(offset + index * total / count)) % total
        while raw in used:
            raw = (raw + 1) % total
        used.add(raw)
        chosen.append(raw)
    return tuple(float(phases[position]) for position in sorted(chosen))


def generate_phase_subsets(
    phase_values: list[float],
    num_phase_values: list[int],
) -> list[tuple[float, ...]]:
    """Generate systematic phase subsets for alphabet search."""
    unique_phases = _unique_sorted_phases(phase_values)
    if len(unique_phases) == 0:
        raise ValueError("phase_values cannot be empty.")

    subsets: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    total = len(unique_phases)
    for requested_count in sorted(set(int(value) for value in num_phase_values if int(value) > 0)):
        if requested_count > total:
            continue
        if requested_count == total:
            subset = tuple(unique_phases)
            if subset not in seen:
                seen.add(subset)
                subsets.append(subset)
            continue
        if total % requested_count == 0:
            step = total // requested_count
            for offset in range(step):
                subset = tuple(float(unique_phases[offset + index * step]) for index in range(requested_count))
                if subset in seen:
                    continue
                seen.add(subset)
                subsets.append(subset)
            continue
        for offset in range(total):
            subset = _phase_subset_from_offset(unique_phases, requested_count, offset)
            if subset in seen:
                continue
            seen.add(subset)
            subsets.append(subset)
    return subsets


def generate_alphabet_candidates_from_grid(
    radius_values: list[float],
    phase_values: list[float],
    num_radii_values: list[int],
    num_phase_values: list[int],
    require_vacuum: bool = True,
    max_local_states: int | None = None,
) -> list[AlphabetCandidateSpec]:
    """Systematically generate coherent-alphabet candidates from a radius/phase pool."""
    radius_subsets = generate_radius_subsets(radius_values, num_radii_values, require_vacuum=require_vacuum)
    phase_subsets = generate_phase_subsets(phase_values, num_phase_values)

    candidates: list[AlphabetCandidateSpec] = []
    seen: set[tuple[complex, ...]] = set()
    for radius_subset in radius_subsets:
        for phase_subset in phase_subsets:
            alpha_values = _build_alpha_values(radius_subset, phase_subset)
            if max_local_states is not None and len(alpha_values) > max_local_states:
                continue
            if alpha_values in seen:
                continue
            seen.add(alpha_values)
            candidates.append(
                AlphabetCandidateSpec(
                    radius_values=tuple(float(value) for value in radius_subset),
                    phase_values=tuple(float(value) for value in phase_subset),
                    alpha_values=alpha_values,
                )
            )
    return candidates


def reduced_joint_inputs_from_alphas(
    cutoff: int,
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray, list[complex], int, int, int]:
    """Product inputs projected to the exact support of the generalized alphabet."""
    local_states, local_alphas = generalized_coherent_alphabet(
        cutoff,
        alpha_values=alpha_values,
        radius_values=radius_values,
        phase_values=phase_values,
    )
    local_kets = []
    for rho in local_states:
        values, vectors = np.linalg.eigh(rho)
        local_kets.append(vectors[:, int(np.argmax(values))])

    local_basis = support_basis(local_kets)
    reduced_local_states = [project_density_to_basis(rho, local_basis) for rho in local_states]

    joint_states: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for x, rho_a in enumerate(reduced_local_states):
        for y, rho_b in enumerate(reduced_local_states):
            joint_states.append(kron(rho_a, rho_b))
            labels.append((x, y))

    joint_basis = kron(local_basis, local_basis)
    local_operator_span = operator_span_rank(reduced_local_states)
    return (
        joint_states,
        labels,
        joint_basis,
        local_alphas,
        local_basis.shape[1],
        joint_states[0].shape[0],
        local_operator_span,
    )


def power_spaced_bounds(num_bins: int, finite_range: float, gamma: float = 1.0) -> np.ndarray:
    """Symmetric axis-aligned bin boundaries for physically constrained IQ partitioning."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    if finite_range <= 0:
        raise ValueError("finite_range must be positive.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")

    if num_bins == 2:
        return np.array([-np.inf, 0.0, np.inf], dtype=float)

    normalized = np.linspace(-1.0, 1.0, num_bins + 1, dtype=float)
    edges = np.sign(normalized) * (np.abs(normalized) ** gamma) * finite_range
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.array([_canonicalize_scalar(value) for value in edges], dtype=float)


def generate_axis_bound_candidates(
    num_bins_values: list[int],
    quadrature_ranges: list[float],
    gamma_values: list[float],
) -> list[AxisBoundsCandidate]:
    """Candidate boundary families for one quadrature axis."""
    if len(num_bins_values) == 0:
        raise ValueError("num_bins_values cannot be empty.")

    candidates: list[AxisBoundsCandidate] = []
    seen: set[tuple[int, tuple[float, ...]]] = set()
    for num_bins in num_bins_values:
        if num_bins == 2:
            bounds = tuple(power_spaced_bounds(2, 1.0, gamma=1.0).tolist())
            key = (2, bounds)
            if key not in seen:
                seen.add(key)
                candidates.append(
                    AxisBoundsCandidate(
                        num_bins=2,
                        finite_range=0.0,
                        gamma=1.0,
                        bounds=bounds,
                    )
                )
            continue

        for finite_range in quadrature_ranges:
            for gamma in gamma_values:
                bounds = tuple(power_spaced_bounds(num_bins, finite_range, gamma=gamma).tolist())
                key = (num_bins, bounds)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    AxisBoundsCandidate(
                        num_bins=num_bins,
                        finite_range=float(finite_range),
                        gamma=float(gamma),
                        bounds=bounds,
                    )
                )
    return candidates


def _target_metadata(
    labels: list[tuple[int, int]],
    local_alphas: list[complex],
    target_index: int,
) -> dict:
    x_index, y_index = labels[target_index]
    return {
        "target_index": int(target_index),
        "target_input": (int(x_index), int(y_index)),
        "target_alphas": [
            _serialize_complex(local_alphas[x_index]),
            _serialize_complex(local_alphas[y_index]),
        ],
    }


def certify_target_inputs(
    input_states: list[np.ndarray],
    probabilities: np.ndarray,
    labels: list[tuple[int, int]],
    local_alphas: list[complex],
    target_indices: list[int] | None = None,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> tuple[dict, list[dict]]:
    """Certify one or more target inputs and return the best certified one."""
    raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
    indices = list(range(len(input_states))) if target_indices is None else list(target_indices)
    reusable_problem = SingleDeviceGuessingProblem(input_states, probabilities)

    best: dict | None = None
    scan: list[dict] = []
    for target_input in indices:
        solver_attempts = [preferred_solver] if preferred_solver is not None else [None, "CLARABEL"]
        solver_errors: list[str] = []
        current: dict | None = None
        for solver_name in solver_attempts:
            try:
                current = reusable_problem.solve(
                    target_input=target_input,
                    preferred_solver=solver_name,
                    solver_options=solver_options,
                    verbose=verbose,
                )
                break
            except RuntimeError as exc:
                label = "default" if solver_name is None else str(solver_name)
                solver_errors.append(f"{label}: {exc}")

        if current is None:
            current = {
                "solver": preferred_solver if preferred_solver is not None else "default+CLARABEL",
                "status": "solver_failed",
                "p_guess": None,
                "H_min": None,
                "solver_errors": solver_errors,
            }

        current.update(_target_metadata(labels, local_alphas, target_input))
        current.update(
            {
                "raw_H_min": float(raw_h[target_input]),
                "raw_p_guess": float(np.max(probabilities[target_input])),
            }
        )
        entry = dict(current)
        scan.append(entry)
        if best is None or (current["H_min"] or -np.inf) > (best["H_min"] or -np.inf):
            best = dict(entry)

    assert best is not None
    return best, scan


def _candidate_summary(
    candidate_index: int,
    x_candidate: AxisBoundsCandidate,
    p_candidate: AxisBoundsCandidate,
    probabilities: np.ndarray,
    labels: list[tuple[int, int]],
    local_alphas: list[complex],
) -> dict:
    raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
    raw_best_index = int(np.argmax(raw_h))
    summary = {
        "candidate_index": int(candidate_index),
        "num_x_bins": int(x_candidate.num_bins),
        "num_p_bins": int(p_candidate.num_bins),
        "num_outputs": int(probabilities.shape[1]),
        "x_bounds": list(x_candidate.bounds),
        "p_bounds": list(p_candidate.bounds),
        "x_range": float(x_candidate.finite_range),
        "p_range": float(p_candidate.finite_range),
        "x_gamma": float(x_candidate.gamma),
        "p_gamma": float(p_candidate.gamma),
        "raw_best_H_min": float(raw_h[raw_best_index]),
        "raw_best_target_index": raw_best_index,
    }
    summary.update(
        {
            "raw_best_target": labels[raw_best_index],
            "raw_best_target_alphas": [
                _serialize_complex(local_alphas[labels[raw_best_index][0]]),
                _serialize_complex(local_alphas[labels[raw_best_index][1]]),
            ],
        }
    )
    return summary


def _alphabet_summary(
    candidate_index: int,
    candidate: AlphabetCandidateSpec,
    local_rank: int,
    local_operator_span: int,
    joint_dim: int,
    joint_operator_span: int,
) -> dict:
    """Compact structural summary for one alphabet candidate."""
    local_space_dim = int(local_rank**2)
    local_span_ratio = 0.0 if local_space_dim == 0 else float(local_operator_span / local_space_dim)
    return {
        "alphabet_candidate_index": int(candidate_index),
        "radius_values": list(candidate.radius_values),
        "phase_values": list(candidate.phase_values),
        "num_local_states": int(len(candidate.alpha_values)),
        "alpha_values": serialize_complex_list(list(candidate.alpha_values)),
        "local_rank": int(local_rank),
        "local_operator_span_rank": int(local_operator_span),
        "local_operator_space_dim": int(local_space_dim),
        "local_span_ratio": float(local_span_ratio),
        "joint_dim": int(joint_dim),
        "operator_span_rank": int(joint_operator_span),
        "operator_space_dim": int(joint_dim**2),
    }


def _raw_partition_candidates(
    joint_states: list[np.ndarray],
    labels: list[tuple[int, int]],
    joint_basis: np.ndarray,
    local_alphas: list[complex],
    cutoff: int,
    num_x_bins_values: list[int] | None,
    num_p_bins_values: list[int] | None,
    quadrature_ranges: list[float] | None,
    gamma_values: list[float] | None,
    num_quadrature_nodes: int | None,
    store_probabilities: bool = True,
) -> tuple[list[dict], list[AxisBoundsCandidate], list[AxisBoundsCandidate]]:
    """Enumerate physically constrained IQ partitions and rank them by raw entropy."""
    x_candidates = generate_axis_bound_candidates(
        [2] if num_x_bins_values is None else list(num_x_bins_values),
        DEFAULT_QUADRATURE_RANGES if quadrature_ranges is None else list(quadrature_ranges),
        DEFAULT_GAMMA_VALUES if gamma_values is None else list(gamma_values),
    )
    p_candidates = generate_axis_bound_candidates(
        [2] if num_p_bins_values is None else list(num_p_bins_values),
        DEFAULT_QUADRATURE_RANGES if quadrature_ranges is None else list(quadrature_ranges),
        DEFAULT_GAMMA_VALUES if gamma_values is None else list(gamma_values),
    )

    raw_candidates: list[dict] = []
    for candidate_index, (x_candidate, p_candidate) in enumerate(
        (pair for pair in ((x_candidate, p_candidate) for x_candidate in x_candidates for p_candidate in p_candidates))
    ):
        probabilities, _, _, _ = dual_homodyne_probabilities(
            joint_states,
            joint_basis,
            cutoff,
            num_x_bins=x_candidate.num_bins,
            num_p_bins=p_candidate.num_bins,
            x_bounds=np.array(x_candidate.bounds, dtype=float),
            p_bounds=np.array(p_candidate.bounds, dtype=float),
            quadrature_range=max(x_candidate.finite_range, p_candidate.finite_range, 1.0),
            num_nodes=num_quadrature_nodes,
        )
        summary = _candidate_summary(
            candidate_index,
            x_candidate,
            p_candidate,
            probabilities,
            labels,
            local_alphas,
        )
        if store_probabilities:
            summary["probabilities"] = probabilities
        raw_candidates.append(summary)

    if len(raw_candidates) == 0:
        raise RuntimeError("No IQ partition candidates were generated.")

    ranked_candidates = sorted(raw_candidates, key=lambda item: item["raw_best_H_min"], reverse=True)
    return ranked_candidates, x_candidates, p_candidates


def run_route5(
    cutoff: int = 6,
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
    num_x_bins: int = 2,
    num_p_bins: int = 2,
    x_bounds: np.ndarray | None = None,
    p_bounds: np.ndarray | None = None,
    quadrature_range: float = 3.0,
    boundary_gamma: float = 1.0,
    num_quadrature_nodes: int | None = None,
    max_inputs_to_certify: int | None = 1,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """Run one generalized coherent-alphabet route with a physically constrained IQ partition."""
    (
        joint_states,
        labels,
        joint_basis,
        local_alphas,
        local_rank,
        joint_dim,
        local_operator_span,
    ) = reduced_joint_inputs_from_alphas(
        cutoff,
        alpha_values=alpha_values,
        radius_values=radius_values,
        phase_values=phase_values,
    )
    resolved_x_bounds = (
        power_spaced_bounds(num_x_bins, quadrature_range, gamma=boundary_gamma)
        if x_bounds is None
        else np.asarray(x_bounds, dtype=float)
    )
    resolved_p_bounds = (
        power_spaced_bounds(num_p_bins, quadrature_range, gamma=boundary_gamma)
        if p_bounds is None
        else np.asarray(p_bounds, dtype=float)
    )

    probabilities, output_labels, x_bounds_out, p_bounds_out = dual_homodyne_probabilities(
        joint_states,
        joint_basis,
        cutoff,
        num_x_bins=num_x_bins,
        num_p_bins=num_p_bins,
        x_bounds=resolved_x_bounds,
        p_bounds=resolved_p_bounds,
        quadrature_range=quadrature_range,
        num_nodes=num_quadrature_nodes,
    )

    raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
    candidate_order = list(np.argsort(-raw_h))
    if max_inputs_to_certify is not None:
        candidate_order = candidate_order[:max_inputs_to_certify]

    best, target_scan = certify_target_inputs(
        joint_states,
        probabilities,
        labels,
        local_alphas,
        target_indices=candidate_order,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
    )
    raw_best_index = int(np.argmax(raw_h))
    best.update(
        {
            "route": "route5_cv_generalized_iq",
            "cutoff": int(cutoff),
            "num_local_states": len(local_alphas),
            "num_inputs": len(joint_states),
            "num_outputs": int(probabilities.shape[1]),
            "num_x_bins": int(num_x_bins),
            "num_p_bins": int(num_p_bins),
            "output_labels": output_labels,
            "local_alphas": serialize_complex_list(local_alphas),
            "local_rank": int(local_rank),
            "local_operator_span_rank": int(local_operator_span),
            "local_operator_space_dim": int(local_rank**2),
            "joint_dim": int(joint_dim),
            "operator_span_rank": int(operator_span_rank(joint_states)),
            "operator_space_dim": int(joint_dim**2),
            "x_bounds": x_bounds_out.tolist(),
            "p_bounds": p_bounds_out.tolist(),
            "boundary_gamma": float(boundary_gamma),
            "quadrature_range": float(quadrature_range),
            "num_quadrature_nodes": None if num_quadrature_nodes is None else int(num_quadrature_nodes),
            "raw_best_target_index": raw_best_index,
            "raw_best_target": labels[raw_best_index],
            "raw_best_target_alphas": [
                _serialize_complex(local_alphas[labels[raw_best_index][0]]),
                _serialize_complex(local_alphas[labels[raw_best_index][1]]),
            ],
            "raw_best_H_min": float(raw_h[raw_best_index]),
            "certified_best_target_index": int(best["target_index"]),
            "certified_best_target": best["target_input"],
            "certified_best_target_alphas": best["target_alphas"],
            "num_inputs_certified": len(target_scan),
            "target_scan": target_scan,
        }
    )
    return best


def search_route5_iq_partitions(
    cutoff: int = 6,
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
    num_x_bins_values: list[int] | None = None,
    num_p_bins_values: list[int] | None = None,
    quadrature_ranges: list[float] | None = None,
    gamma_values: list[float] | None = None,
    num_quadrature_nodes: int | None = None,
    certify_top_k: int = 3,
    max_inputs_to_certify: int | None = 1,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """Search physically constrained axis-aligned IQ partitions for route 5."""
    (
        joint_states,
        labels,
        joint_basis,
        local_alphas,
        local_rank,
        joint_dim,
        local_operator_span,
    ) = reduced_joint_inputs_from_alphas(
        cutoff,
        alpha_values=alpha_values,
        radius_values=radius_values,
        phase_values=phase_values,
    )

    ranked_candidates, x_candidates, p_candidates = _raw_partition_candidates(
        joint_states,
        labels,
        joint_basis,
        local_alphas,
        cutoff,
        num_x_bins_values=num_x_bins_values,
        num_p_bins_values=num_p_bins_values,
        quadrature_ranges=quadrature_ranges,
        gamma_values=gamma_values,
        num_quadrature_nodes=num_quadrature_nodes,
        store_probabilities=True,
    )
    certify_count = min(max(0, certify_top_k), len(ranked_candidates))
    certified_candidates: list[dict] = []
    for candidate in ranked_candidates[:certify_count]:
        probabilities = candidate["probabilities"]
        raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
        target_indices = list(np.argsort(-raw_h))
        if max_inputs_to_certify is not None:
            target_indices = target_indices[:max_inputs_to_certify]

        certified, target_scan = certify_target_inputs(
            joint_states,
            probabilities,
            labels,
            local_alphas,
            target_indices=target_indices,
            preferred_solver=preferred_solver,
            solver_options=solver_options,
            verbose=verbose,
        )
        certified_entry = {
            key: value
            for key, value in candidate.items()
            if key != "probabilities"
        }
        certified_entry.update(certified)
        certified_entry["num_inputs_certified"] = len(target_scan)
        certified_entry["target_scan"] = target_scan
        certified_candidates.append(certified_entry)

    if len(certified_candidates) > 0:
        best = dict(
            max(
                certified_candidates,
                key=lambda item: item["H_min"] if item["H_min"] is not None else -np.inf,
            )
        )
    else:
        best = {
            key: value
            for key, value in ranked_candidates[0].items()
            if key != "probabilities"
        }
        best.update(
            {
                "solver": None,
                "status": "not_certified",
                "p_guess": None,
                "H_min": None,
                "target_index": int(best["raw_best_target_index"]),
                "target_input": best["raw_best_target"],
                "target_alphas": best["raw_best_target_alphas"],
                "num_inputs_certified": 0,
                "target_scan": [],
            }
        )

    best.update(
        {
            "route": "route5_cv_generalized_iq_search",
            "cutoff": int(cutoff),
            "num_local_states": len(local_alphas),
            "num_inputs": len(joint_states),
            "local_alphas": serialize_complex_list(local_alphas),
            "local_rank": int(local_rank),
            "local_operator_span_rank": int(local_operator_span),
            "local_operator_space_dim": int(local_rank**2),
            "joint_dim": int(joint_dim),
            "operator_span_rank": int(operator_span_rank(joint_states)),
            "operator_space_dim": int(joint_dim**2),
            "num_x_candidates": len(x_candidates),
            "num_p_candidates": len(p_candidates),
            "num_partition_candidates": len(ranked_candidates),
            "certify_top_k": int(certify_count),
            "selection_strategy": "rank partitions by raw-best target entropy, then certify top candidates",
            "raw_partition_ranking": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "probabilities"
                }
                for candidate in ranked_candidates[: min(10, len(ranked_candidates))]
            ],
            "certified_partition_results": certified_candidates,
            "selected_candidate_index": int(best["candidate_index"]),
            "certified_best_target_index": int(best["target_index"]),
            "certified_best_target": best["target_input"],
            "certified_best_target_alphas": best["target_alphas"],
        }
    )
    return best


def search_route5_alphabets(
    cutoff: int = 4,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
    num_radii_values: list[int] | None = None,
    num_phase_values: list[int] | None = None,
    require_vacuum: bool = True,
    max_local_states: int | None = None,
    num_x_bins_values: list[int] | None = None,
    num_p_bins_values: list[int] | None = None,
    quadrature_ranges: list[float] | None = None,
    gamma_values: list[float] | None = None,
    num_quadrature_nodes: int | None = None,
    alphabet_top_k: int = 3,
    certify_top_k: int = 1,
    max_inputs_to_certify: int | None = 1,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """Search trusted alphabets first, then certify the best IQ-partition candidates."""
    radius_pool = DEFAULT_RADIUS_VALUES if radius_values is None else list(radius_values)
    phase_pool = DEFAULT_PHASE_VALUES if phase_values is None else list(phase_values)
    radius_count_values = [2, 3] if num_radii_values is None else list(num_radii_values)
    phase_count_values = [4] if num_phase_values is None else list(num_phase_values)

    candidate_specs = generate_alphabet_candidates_from_grid(
        radius_pool,
        phase_pool,
        radius_count_values,
        phase_count_values,
        require_vacuum=require_vacuum,
        max_local_states=max_local_states,
    )
    if len(candidate_specs) == 0:
        raise RuntimeError("No trusted-alphabet candidates were generated.")

    raw_alphabet_results: list[dict] = []
    for alphabet_index, candidate in enumerate(candidate_specs):
        (
            joint_states,
            labels,
            joint_basis,
            local_alphas,
            local_rank,
            joint_dim,
            local_operator_span,
        ) = reduced_joint_inputs_from_alphas(
            cutoff,
            alpha_values=list(candidate.alpha_values),
        )
        ranked_partitions, _, _ = _raw_partition_candidates(
            joint_states,
            labels,
            joint_basis,
            local_alphas,
            cutoff,
            num_x_bins_values=num_x_bins_values,
            num_p_bins_values=num_p_bins_values,
            quadrature_ranges=quadrature_ranges,
            gamma_values=gamma_values,
            num_quadrature_nodes=num_quadrature_nodes,
            store_probabilities=False,
        )
        summary = _alphabet_summary(
            alphabet_index,
            candidate,
            local_rank=local_rank,
            local_operator_span=local_operator_span,
            joint_dim=joint_dim,
            joint_operator_span=operator_span_rank(joint_states),
        )
        summary["raw_best_partition"] = ranked_partitions[0]
        summary["raw_partition_ranking"] = ranked_partitions[: min(5, len(ranked_partitions))]
        raw_alphabet_results.append(summary)

    ranked_alphabets = sorted(
        raw_alphabet_results,
        key=lambda item: (
            item["local_span_ratio"],
            item["raw_best_partition"]["raw_best_H_min"],
            item["local_operator_span_rank"],
            item["operator_span_rank"],
            -item["num_local_states"],
        ),
        reverse=True,
    )

    alphabet_certify_count = min(max(0, alphabet_top_k), len(ranked_alphabets))
    certified_alphabet_results: list[dict] = []
    for alphabet_result in ranked_alphabets[:alphabet_certify_count]:
        partition_result = search_route5_iq_partitions(
            cutoff=cutoff,
            alpha_values=[
                complex(entry["real"], entry["imag"])
                for entry in alphabet_result["alpha_values"]
            ],
            num_x_bins_values=num_x_bins_values,
            num_p_bins_values=num_p_bins_values,
            quadrature_ranges=quadrature_ranges,
            gamma_values=gamma_values,
            num_quadrature_nodes=num_quadrature_nodes,
            certify_top_k=certify_top_k,
            max_inputs_to_certify=max_inputs_to_certify,
            preferred_solver=preferred_solver,
            solver_options=solver_options,
            verbose=verbose,
        )
        certified_entry = dict(alphabet_result)
        certified_entry["partition_search_result"] = partition_result
        certified_entry["best_certified_H_min"] = partition_result["H_min"]
        certified_entry["best_certified_partition"] = {
            "candidate_index": partition_result["selected_candidate_index"],
            "num_outputs": partition_result["num_outputs"],
            "num_x_bins": partition_result["num_x_bins"],
            "num_p_bins": partition_result["num_p_bins"],
            "x_bounds": partition_result["x_bounds"],
            "p_bounds": partition_result["p_bounds"],
            "raw_best_H_min": partition_result["raw_best_H_min"],
            "certified_H_min": partition_result["H_min"],
            "certified_best_target": partition_result["certified_best_target"],
            "certified_best_target_alphas": partition_result["certified_best_target_alphas"],
        }
        certified_alphabet_results.append(certified_entry)

    if len(certified_alphabet_results) > 0:
        best = dict(
            max(
                certified_alphabet_results,
                key=lambda item: item["best_certified_H_min"] if item["best_certified_H_min"] is not None else -np.inf,
            )
        )
        partition_result = best["partition_search_result"]
    else:
        best = dict(ranked_alphabets[0])
        partition_result = None

    best.update(
        {
            "route": "route5_cv_alphabet_search",
            "cutoff": int(cutoff),
            "radius_pool": [float(value) for value in radius_pool],
            "phase_pool": [float(value) for value in _unique_sorted_phases(phase_pool)],
            "num_radii_values": [int(value) for value in radius_count_values],
            "num_phase_values": [int(value) for value in phase_count_values],
            "require_vacuum": bool(require_vacuum),
            "max_local_states": None if max_local_states is None else int(max_local_states),
            "num_alphabet_candidates": len(candidate_specs),
            "alphabet_top_k": int(alphabet_certify_count),
            "partition_certify_top_k": int(certify_top_k),
            "alphabet_selection_strategy": "rank alphabets by local span ratio, then raw-best IQ entropy",
            "raw_alphabet_ranking": ranked_alphabets[: min(10, len(ranked_alphabets))],
            "certified_alphabet_results": certified_alphabet_results,
            "selected_alphabet_candidate_index": int(best["alphabet_candidate_index"]),
            "selected_partition_candidate_index": (
                int(partition_result["selected_candidate_index"])
                if partition_result is not None
                else int(best["raw_best_partition"]["candidate_index"])
            ),
            "best_certified_H_min": None if partition_result is None else partition_result["H_min"],
            "best_partition_search_result": partition_result,
        }
    )
    return best
