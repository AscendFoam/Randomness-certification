from __future__ import annotations

import math

import numpy as np

from ..common import (
    balanced_beamsplitter_unitary,
    coherent_state,
    density_from_ket,
    guessing_prob_single_device,
    kron,
    operator_span_rank,
    project_density_to_basis,
    quadrature_hermite_data,
    quadrature_povms_from_node_masks,
    support_basis,
)


def phase_alphabet(
    mu: float,
    cutoff: int,
    num_phases: int = 4,
) -> tuple[list[np.ndarray], list[float]]:
    """Local coherent-state alphabet with uniformly spaced phases."""
    phases = list(np.linspace(0.0, 2.0 * np.pi, num_phases, endpoint=False))
    amplitude = math.sqrt(mu)
    states = []
    for phase in phases:
        ket = coherent_state(cutoff, amplitude * np.exp(1j * phase))
        states.append(density_from_ket(ket))
    return states, phases


def four_phase_alphabet(mu: float, cutoff: int) -> tuple[list[np.ndarray], list[float]]:
    """Backward-compatible four-phase alphabet helper."""
    return phase_alphabet(mu, cutoff, num_phases=4)


def reduced_joint_inputs(
    mu: float,
    cutoff: int,
    num_phases: int = 4,
) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray, int, int]:
    """Product inputs projected to the exact local support of the alphabet."""
    local_states, _ = phase_alphabet(mu, cutoff, num_phases=num_phases)
    local_kets = []
    for rho in local_states:
        values, vectors = np.linalg.eigh(rho)
        local_kets.append(vectors[:, -1])
    local_basis = support_basis(local_kets)
    reduced_local_states = [project_density_to_basis(rho, local_basis) for rho in local_states]

    joint_states: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for x, rho_a in enumerate(reduced_local_states):
        for y, rho_b in enumerate(reduced_local_states):
            joint_states.append(kron(rho_a, rho_b))
            labels.append((x, y))
    joint_basis = kron(local_basis, local_basis)
    return joint_states, labels, joint_basis, local_basis.shape[1], joint_states[0].shape[0]


def default_quadrature_nodes(cutoff: int) -> int:
    """Stable quadrature integration grid for CV Bell statistics."""
    return max(400, 60 * cutoff)


def default_quadrature_bounds(num_bins: int, finite_range: float = 3.0) -> np.ndarray:
    """Default quadrature bins used after the central CV Bell measurement."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if num_bins == 2:
        return np.array([-np.inf, 0.0, np.inf], dtype=float)
    bounds = np.linspace(-finite_range, finite_range, num_bins + 1, dtype=float)
    bounds[0] = -np.inf
    bounds[-1] = np.inf
    return bounds


def quadrature_povms_from_bounds(
    cutoff: int,
    theta: float,
    bounds: np.ndarray,
    num_nodes: int | None = None,
) -> list[np.ndarray]:
    """Coarse-grained quadrature POVM for arbitrary interval boundaries."""
    edges = np.asarray(bounds, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bounds must be a one-dimensional array with at least two entries.")
    if not np.all(np.diff(edges) >= 0):
        raise ValueError("bounds must be nondecreasing.")

    nodes_count = default_quadrature_nodes(cutoff) if num_nodes is None else num_nodes
    nodes, _, _ = quadrature_hermite_data(cutoff, nodes_count)
    num_bins = edges.size - 1
    masks = np.zeros((num_bins, nodes.size), dtype=float)
    for index in range(num_bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = np.ones(nodes.size, dtype=bool)
        if np.isfinite(lower):
            mask &= nodes >= lower
        if np.isfinite(upper):
            if index == num_bins - 1:
                mask &= nodes <= upper
            else:
                mask &= nodes < upper
        masks[index, mask] = 1.0
    return quadrature_povms_from_node_masks(
        cutoff,
        theta,
        masks,
        num_nodes=nodes_count,
    )


def dual_homodyne_povm(
    cutoff: int,
    num_x_bins: int = 2,
    num_p_bins: int = 2,
    x_bounds: np.ndarray | None = None,
    p_bounds: np.ndarray | None = None,
    quadrature_range: float = 3.0,
    num_nodes: int | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Central POVM from a balanced beamsplitter followed by q/p coarse graining."""
    x_edges = (
        default_quadrature_bounds(num_x_bins, finite_range=quadrature_range)
        if x_bounds is None
        else np.asarray(x_bounds, dtype=float)
    )
    p_edges = (
        default_quadrature_bounds(num_p_bins, finite_range=quadrature_range)
        if p_bounds is None
        else np.asarray(p_bounds, dtype=float)
    )

    x_povms = quadrature_povms_from_bounds(cutoff, 0.0, x_edges, num_nodes=num_nodes)
    p_povms = quadrature_povms_from_bounds(cutoff, np.pi / 2.0, p_edges, num_nodes=num_nodes)
    beamsplitter = balanced_beamsplitter_unitary(cutoff)

    povm: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for x_index, q_effect in enumerate(x_povms):
        for p_index, p_effect in enumerate(p_povms):
            output_effect = kron(q_effect, p_effect)
            povm.append(beamsplitter.conj().T @ output_effect @ beamsplitter)
            labels.append((x_index, p_index))
    return povm, labels, x_edges, p_edges


def project_povm_to_basis(povm: list[np.ndarray], basis: np.ndarray) -> list[np.ndarray]:
    """Project POVM elements to the exact support of the trusted input alphabet."""
    projected: list[np.ndarray] = []
    for effect in povm:
        reduced = basis.conj().T @ effect @ basis
        projected.append(0.5 * (reduced + reduced.conj().T))
    return projected


def measurement_probabilities(
    input_states: list[np.ndarray],
    measurement_povm: list[np.ndarray],
) -> np.ndarray:
    """P(c|input) from a common state and POVM representation."""
    out = np.zeros((len(input_states), len(measurement_povm)))
    for s, rho in enumerate(input_states):
        for c, povm in enumerate(measurement_povm):
            out[s, c] = float(np.real(np.trace(povm @ rho)))
    return out


def dual_homodyne_probabilities(
    joint_states: list[np.ndarray],
    joint_basis: np.ndarray,
    cutoff: int,
    num_x_bins: int = 2,
    num_p_bins: int = 2,
    x_bounds: np.ndarray | None = None,
    p_bounds: np.ndarray | None = None,
    quadrature_range: float = 3.0,
    num_nodes: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Joint CV Bell statistics from the same truncated states used by the SDP."""
    full_povm, output_labels, x_edges, p_edges = dual_homodyne_povm(
        cutoff,
        num_x_bins=num_x_bins,
        num_p_bins=num_p_bins,
        x_bounds=x_bounds,
        p_bounds=p_bounds,
        quadrature_range=quadrature_range,
        num_nodes=num_nodes,
    )
    reduced_povm = project_povm_to_basis(full_povm, joint_basis)
    probabilities = measurement_probabilities(joint_states, reduced_povm)
    return probabilities, output_labels, x_edges, p_edges


def run_route3(
    mu: float = 0.5,
    cutoff: int = 12,
    num_phases: int = 4,
    num_x_bins: int = 2,
    num_p_bins: int = 2,
    x_bounds: np.ndarray | None = None,
    p_bounds: np.ndarray | None = None,
    quadrature_range: float = 3.0,
    num_quadrature_nodes: int | None = None,
    max_inputs_to_certify: int | None = None,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> dict:
    """CV hardware route with four phase states and a single-device SDP."""
    joint_states, labels, joint_basis, local_rank, joint_dim = reduced_joint_inputs(
        mu,
        cutoff,
        num_phases=num_phases,
    )
    probabilities, output_labels, x_bounds, p_bounds = dual_homodyne_probabilities(
        joint_states,
        joint_basis,
        cutoff,
        num_x_bins=num_x_bins,
        num_p_bins=num_p_bins,
        x_bounds=x_bounds,
        p_bounds=p_bounds,
        quadrature_range=quadrature_range,
        num_nodes=num_quadrature_nodes,
    )

    raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
    candidate_order = list(np.argsort(-raw_h))
    if max_inputs_to_certify is not None:
        candidate_order = candidate_order[:max_inputs_to_certify]

    best: dict | None = None
    for target_input in candidate_order:
        current = guessing_prob_single_device(
            joint_states,
            probabilities,
            target_input=target_input,
            preferred_solver=preferred_solver,
            verbose=verbose,
        )
        current["target_input"] = labels[target_input]
        current["raw_H_min"] = float(raw_h[target_input])
        if best is None or (current["H_min"] or -np.inf) > (best["H_min"] or -np.inf):
            best = current

    assert best is not None
    best.update(
        {
            "route": "route3_cv_four_phase",
            "mu": mu,
            "cutoff": cutoff,
            "num_phases": num_phases,
            "num_inputs": len(joint_states),
            "num_outputs": probabilities.shape[1],
            "output_labels": output_labels,
            "local_rank": local_rank,
            "joint_dim": joint_dim,
            "operator_span_rank": operator_span_rank(joint_states),
            "operator_space_dim": joint_dim**2,
            "x_bounds": x_bounds.tolist(),
            "p_bounds": p_bounds.tolist(),
            "num_x_bins": num_x_bins,
            "num_p_bins": num_p_bins,
            "quadrature_range": quadrature_range,
            "num_quadrature_nodes": default_quadrature_nodes(cutoff)
            if num_quadrature_nodes is None
            else int(num_quadrature_nodes),
            "num_inputs_certified": len(candidate_order),
        }
    )
    return best
