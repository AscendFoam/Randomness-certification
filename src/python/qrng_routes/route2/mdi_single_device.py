from __future__ import annotations

import numpy as np

from ..common import density_from_ket, guessing_prob_single_device, kron, operator_span_rank


def tetrahedral_qubit_states() -> list[np.ndarray]:
    """Four informationally complete qubit states."""
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    bloch_vectors = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    ) / np.sqrt(3.0)
    states = []
    for vx, vy, vz in bloch_vectors:
        states.append(0.5 * (identity + vx * sigma_x + vy * sigma_y + vz * sigma_z))
    return states


def local_ic_qubit_states() -> list[np.ndarray]:
    """A simple linearly independent qubit input set."""
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    plus = (zero + one) / np.sqrt(2.0)
    plus_i = (zero + 1j * one) / np.sqrt(2.0)
    return [density_from_ket(vec) for vec in (zero, one, plus, plus_i)]


def bell_povm() -> list[np.ndarray]:
    """Projective Bell POVM on two qubits."""
    basis = np.eye(2, dtype=complex)
    zero = basis[:, 0]
    one = basis[:, 1]
    bell_kets = [
        (np.kron(zero, zero) + np.kron(one, one)) / np.sqrt(2.0),
        (np.kron(zero, zero) - np.kron(one, one)) / np.sqrt(2.0),
        (np.kron(zero, one) + np.kron(one, zero)) / np.sqrt(2.0),
        (np.kron(zero, one) - np.kron(one, zero)) / np.sqrt(2.0),
    ]
    return [density_from_ket(ket) for ket in bell_kets]


def fourier_povm_4d() -> list[np.ndarray]:
    """Extremal four-outcome POVM on the four-dimensional joint input space."""
    omega = 1j
    basis = np.eye(4, dtype=complex)
    povm = []
    for k in range(4):
        vec = sum((omega ** (j * k)) * basis[:, j] for j in range(4)) / 2.0
        povm.append(density_from_ket(vec))
    return povm


def random_frame_povm(
    dimension: int,
    num_outputs: int,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Random rank-1 POVM built from a normalized frame."""
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(num_outputs):
        vec = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        vec /= np.linalg.norm(vec)
        vectors.append(vec)

    frame = sum(np.outer(vec, vec.conj()) for vec in vectors)
    values, basis = np.linalg.eigh(frame)
    frame_inv_half = basis @ np.diag(1.0 / np.sqrt(np.maximum(values, 1e-12))) @ basis.conj().T
    return [frame_inv_half @ np.outer(vec, vec.conj()) @ frame_inv_half.conj().T for vec in vectors]


def product_input_states(local_a: list[np.ndarray], local_b: list[np.ndarray]) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Product inputs and their labels."""
    states: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for x, rho_a in enumerate(local_a):
        for y, rho_b in enumerate(local_b):
            states.append(kron(rho_a, rho_b))
            labels.append((x, y))
    return states, labels


def measurement_probabilities(
    input_states: list[np.ndarray],
    measurement_povm: list[np.ndarray],
) -> np.ndarray:
    """P[c|input]."""
    out = np.zeros((len(input_states), len(measurement_povm)))
    for s, rho in enumerate(input_states):
        for c, povm in enumerate(measurement_povm):
            out[s, c] = float(np.real(np.trace(povm @ rho)))
    return out


def certify_target_inputs(
    input_states: list[np.ndarray],
    probabilities: np.ndarray,
    labels: list[tuple[int, int]],
    target_indices: list[int] | None = None,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> tuple[dict, list[dict]]:
    """Certify one or more target inputs and return the best certified one."""
    raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
    indices = list(range(len(input_states))) if target_indices is None else list(target_indices)

    best: dict | None = None
    scan: list[dict] = []
    for target_input in indices:
        current = guessing_prob_single_device(
            input_states,
            probabilities,
            target_input=target_input,
            preferred_solver=preferred_solver,
            verbose=verbose,
        )
        current.update(
            {
                "target_index": int(target_input),
                "target_input": labels[target_input],
                "raw_H_min": float(raw_h[target_input]),
                "raw_p_guess": float(np.max(probabilities[target_input])),
            }
        )
        current_entry = dict(current)
        scan.append(current_entry)
        if best is None or (current["H_min"] or -np.inf) > (best["H_min"] or -np.inf):
            best = dict(current_entry)

    assert best is not None
    return best, scan


def run_route2(
    max_inputs_to_certify: int | None = None,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> dict:
    """Correct single-device MDI prototype with informationally complete inputs."""
    local_states = local_ic_qubit_states()
    joint_states, labels = product_input_states(local_states, local_states)
    povm = fourier_povm_4d()
    probabilities = measurement_probabilities(joint_states, povm)

    raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
    raw_best_index = int(np.argmax(raw_h))
    target_indices = list(np.argsort(-raw_h))
    if max_inputs_to_certify is not None:
        target_indices = target_indices[:max_inputs_to_certify]

    best, target_scan = certify_target_inputs(
        joint_states,
        probabilities,
        labels,
        target_indices=target_indices,
        preferred_solver=preferred_solver,
        verbose=verbose,
    )
    best.update(
        {
            "route": "route2_mdi_single_device",
            "num_inputs": len(joint_states),
            "num_outputs": len(povm),
            "joint_dim": joint_states[0].shape[0],
            "operator_span_rank": operator_span_rank(joint_states),
            "operator_space_dim": joint_states[0].shape[0] ** 2,
            "raw_best_target_index": raw_best_index,
            "raw_best_target": labels[raw_best_index],
            "raw_best_H_min": float(raw_h[raw_best_index]),
            "certified_best_target_index": best["target_index"],
            "certified_best_target": best["target_input"],
            "num_inputs_certified": len(target_scan),
            "target_scan": target_scan,
        }
    )
    return best


def search_route2_high_entropy(
    num_outputs: int,
    num_trials: int = 20,
    preferred_solver: str | None = None,
    seed: int = 7,
    verbose: bool = False,
) -> dict:
    """Search random higher-output POVMs to test whether H_min can exceed 2 bits."""
    rng = np.random.default_rng(seed)
    local_states = local_ic_qubit_states()
    joint_states, labels = product_input_states(local_states, local_states)

    best_raw = -np.inf
    best_trial: dict | None = None
    for trial_index in range(num_trials):
        povm = random_frame_povm(4, num_outputs, seed=int(rng.integers(0, 2**31 - 1)))
        probabilities = measurement_probabilities(joint_states, povm)
        raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
        raw_best_index = int(np.argmax(raw_h))
        if float(raw_h[raw_best_index]) > best_raw:
            best_raw = float(raw_h[raw_best_index])
            best_trial = {
                "trial_index": int(trial_index),
                "probabilities": probabilities,
                "raw_best_target_index": raw_best_index,
                "raw_best_target": labels[raw_best_index],
                "raw_best_H_min": float(raw_h[raw_best_index]),
            }

    assert best_trial is not None
    certified, target_scan = certify_target_inputs(
        joint_states,
        best_trial["probabilities"],
        labels,
        target_indices=list(range(len(joint_states))),
        preferred_solver=preferred_solver,
        verbose=verbose,
    )
    certified.update(
        {
            "route": "route2_random_high_output_search",
            "num_outputs": num_outputs,
            "num_trials": num_trials,
            "selected_trial_index": best_trial["trial_index"],
            "selection_strategy": "raw-best trial, then full target certification",
            "raw_best_target_index": best_trial["raw_best_target_index"],
            "raw_best_target": best_trial["raw_best_target"],
            "raw_best_H_min": best_trial["raw_best_H_min"],
            "certified_best_target_index": certified["target_index"],
            "certified_best_target": certified["target_input"],
            "num_inputs_certified": len(target_scan),
            "target_scan": target_scan,
        }
    )
    return certified
