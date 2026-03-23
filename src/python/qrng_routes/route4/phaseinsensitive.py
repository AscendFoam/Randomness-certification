from __future__ import annotations

import json
import warnings
from itertools import combinations, product as iterproduct
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.io import loadmat
from scipy.special import gammaln

from ..common import solve_cvxpy_problem

FULL_MU = [0, 20, 40, 60, 80, 100, 120, 140, 160]
DEFAULT_SELECTED_MU = [100, 120, 140]
DEFAULT_Q = [0.25, 0.25, 0.5]
DEFAULT_CUTOFF = 280
DEFAULT_NUM_OUTPUTS = 6
DEFAULT_PROB_FLOOR = 1e-12
DEFAULT_SHIFT = 0


def _default_probability_path() -> Path:
    return Path(__file__).resolve().parents[3] / "matlab" / "Probability.mat"


def _clean_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def result_to_json(result: Any) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False, default=_clean_value)


def build_equal_cover_edges(num_raw_bins: int, num_outputs: int) -> np.ndarray:
    """Partition raw bins into contiguous coarse bins that cover every entry exactly once."""
    if num_outputs <= 0:
        raise ValueError("num_outputs must be positive.")
    if num_outputs > num_raw_bins:
        raise ValueError(
            f"num_outputs={num_outputs} exceeds the available raw bins ({num_raw_bins})."
        )
    edges = np.array([(k * num_raw_bins) // num_outputs for k in range(num_outputs + 1)], dtype=int)
    if edges[0] != 0 or edges[-1] != num_raw_bins:
        raise RuntimeError("Internal error while constructing coarse-graining edges.")
    if np.any(np.diff(edges) <= 0):
        raise RuntimeError("Coarse-graining edges must be strictly increasing.")
    return edges


def coarse_grain_row(probabilities_256: np.ndarray, num_outputs: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate a raw detector row into a smaller number of bins."""
    raw = np.asarray(probabilities_256, dtype=float).reshape(-1)
    edges = build_equal_cover_edges(raw.size, num_outputs)
    coarse = np.array([raw[edges[k] : edges[k + 1]].sum() for k in range(num_outputs)], dtype=float)
    return coarse, edges


def load_probability_data(probability_path: str | Path | None = None) -> np.ndarray:
    """Load the detector statistics used by the phase-insensitive model."""
    path = Path(probability_path) if probability_path is not None else _default_probability_path()
    mat_data = loadmat(path)
    variable_names = [name for name in mat_data.keys() if not name.startswith("__")]
    if not variable_names:
        raise ValueError(f"No data arrays found in {path}.")
    table = np.asarray(mat_data[variable_names[0]], dtype=float)
    if table.ndim != 2:
        raise ValueError(f"Expected a 2-D probability table in {path}, got shape {table.shape}.")
    return table


def build_coherent_diagonals(selected_mu_list: list[int] | tuple[int, ...], cutoff: int) -> np.ndarray:
    """Fock-diagonal coherent-state distributions truncated at cutoff."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")
    diagonals = np.zeros((len(selected_mu_list), cutoff), dtype=float)
    photon_numbers = np.arange(cutoff, dtype=float)
    for idx, mu in enumerate(selected_mu_list):
        if mu < 0:
            raise ValueError("Mean photon numbers must be non-negative.")
        if mu == 0:
            diagonals[idx, 0] = 1.0
            continue
        log_probs = -mu + photon_numbers * np.log(mu) - gammaln(photon_numbers + 1.0)
        diagonals[idx, :] = np.exp(log_probs)
    return diagonals


def distribution_only_guessing_probability(probabilities: np.ndarray, q_selected: np.ndarray) -> float:
    """Best guessing probability implied only by the observed output distribution."""
    return float(np.dot(q_selected, probabilities.max(axis=1)))


def prepare_phaseinsensitive_instance(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    probability_path: str | Path | None = None,
    full_mu: list[int] | tuple[int, ...] = FULL_MU,
) -> dict[str, Any]:
    """Prepare the statistics and coherent-state diagonals shared by the primal and dual solvers."""
    selected_mu = list(selected_mu_list)
    full_mu_list = list(full_mu)
    if len(selected_mu) == 0:
        raise ValueError("At least one input state is required.")
    if any(mu not in full_mu_list for mu in selected_mu):
        raise ValueError(f"selected_mu_list must be a subset of {full_mu_list}.")

    q = np.asarray(q_selected, dtype=float).reshape(-1)
    if q.size != len(selected_mu):
        raise ValueError("q_selected must have the same length as selected_mu_list.")
    if np.any(q < 0):
        raise ValueError("q_selected must be non-negative.")
    if float(q.sum()) <= 0.0:
        raise ValueError("q_selected must sum to a positive value.")
    q = q / q.sum()

    probability_table = load_probability_data(probability_path)
    if probability_table.shape[0] <= max(full_mu_list.index(mu) + shift for mu in selected_mu):
        raise ValueError("Probability table does not contain the requested shifted rows.")

    raw_probabilities = np.zeros((len(selected_mu), num_outputs), dtype=float)
    edges: np.ndarray | None = None
    selected_indices = [full_mu_list.index(mu) for mu in selected_mu]
    for row_idx, full_idx in enumerate(selected_indices):
        coarse, row_edges = coarse_grain_row(probability_table[full_idx + shift, :], num_outputs)
        raw_probabilities[row_idx, :] = coarse
        if edges is None:
            edges = row_edges

    assert edges is not None
    regularized_probabilities = raw_probabilities.copy()
    regularized_entries = 0
    if prob_floor is not None and prob_floor > 0:
        regularized_entries = int((regularized_probabilities == 0.0).sum())
        regularized_probabilities = np.maximum(regularized_probabilities, prob_floor)
        regularized_probabilities = regularized_probabilities / regularized_probabilities.sum(
            axis=1, keepdims=True
        )

    rho_diag = build_coherent_diagonals(selected_mu, cutoff)
    mixed_zero_columns = [
        int(column)
        for column in range(num_outputs)
        if np.any(raw_probabilities[:, column] == 0.0) and np.any(raw_probabilities[:, column] > 0.0)
    ]
    all_zero_columns = [
        int(column) for column in range(num_outputs) if np.all(raw_probabilities[:, column] == 0.0)
    ]

    return {
        "selected_mu_list": selected_mu,
        "q_selected": q,
        "cutoff": cutoff,
        "num_inputs": len(selected_mu),
        "num_outputs": num_outputs,
        "shift": shift,
        "probability_path": str(
            Path(probability_path) if probability_path is not None else _default_probability_path()
        ),
        "rho_diag": rho_diag,
        "probabilities_raw": raw_probabilities,
        "probabilities": regularized_probabilities,
        "edges": edges,
        "block_widths": np.diff(edges),
        "row_sums_raw": raw_probabilities.sum(axis=1),
        "mixed_zero_columns_raw": mixed_zero_columns,
        "all_zero_columns_raw": all_zero_columns,
        "regularized_entries": regularized_entries,
        "prob_floor": prob_floor,
        "distribution_only_p_guess_raw": distribution_only_guessing_probability(raw_probabilities, q),
        "distribution_only_p_guess": distribution_only_guessing_probability(regularized_probabilities, q),
    }


def estimate_primal_problem_size(
    num_inputs: int,
    num_outputs: int,
    cutoff: int,
) -> dict[str, int]:
    num_strategies = num_outputs ** (num_inputs + 1)
    variable_count = cutoff * num_outputs * num_strategies
    normalization_constraints = cutoff * num_strategies - num_strategies
    statistics_constraints = num_inputs * num_outputs
    return {
        "num_strategies": int(num_strategies),
        "variable_count": int(variable_count),
        "normalization_constraints": int(normalization_constraints),
        "statistics_constraints": int(statistics_constraints),
    }


def _instance_summary(instance: dict[str, Any]) -> dict[str, Any]:
    raw_p_guess = float(instance["distribution_only_p_guess_raw"])
    reg_p_guess = float(instance["distribution_only_p_guess"])
    return {
        "selected_mu_list": list(instance["selected_mu_list"]),
        "q_selected": np.asarray(instance["q_selected"], dtype=float).tolist(),
        "num_inputs": int(instance["num_inputs"]),
        "num_outputs": int(instance["num_outputs"]),
        "cutoff": int(instance["cutoff"]),
        "shift": int(instance["shift"]),
        "prob_floor": None if instance["prob_floor"] is None else float(instance["prob_floor"]),
        "regularized_entries": int(instance["regularized_entries"]),
        "probability_path": str(instance["probability_path"]),
        "edges": np.asarray(instance["edges"], dtype=int).tolist(),
        "block_widths": np.asarray(instance["block_widths"], dtype=int).tolist(),
        "row_sums_raw": np.asarray(instance["row_sums_raw"], dtype=float).tolist(),
        "mixed_zero_columns_raw": list(instance["mixed_zero_columns_raw"]),
        "all_zero_columns_raw": list(instance["all_zero_columns_raw"]),
        "has_mixed_zero_column_pathology": bool(instance["mixed_zero_columns_raw"]),
        "distribution_only_p_guess_raw": raw_p_guess,
        "distribution_only_H_min_raw": float(-np.log2(raw_p_guess)) if raw_p_guess > 0 else None,
        "distribution_only_p_guess": reg_p_guess,
        "distribution_only_H_min": float(-np.log2(reg_p_guess)) if reg_p_guess > 0 else None,
    }


def solve_phaseinsensitive_dual(
    instance: dict[str, Any],
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Dual LP for the diagonal, phase-insensitive model."""
    probabilities = np.asarray(instance["probabilities"], dtype=float)
    rho_diag = np.asarray(instance["rho_diag"], dtype=float)
    q_selected = np.asarray(instance["q_selected"], dtype=float)
    num_inputs, num_outputs = probabilities.shape
    cutoff = rho_diag.shape[1]

    num_guess_funcs = num_outputs**num_inputs
    guess_funcs = np.array(list(iterproduct(range(num_outputs), repeat=num_inputs)), dtype=int)

    q_rho = q_selected[:, None] * rho_diag
    coeffs = np.zeros((num_guess_funcs, num_outputs, cutoff), dtype=float)
    for guess_index in range(num_guess_funcs):
        for output in range(num_outputs):
            mask = guess_funcs[guess_index, :] == output
            if np.any(mask):
                coeffs[guess_index, output, :] = q_rho[mask, :].sum(axis=0)

    dual_vars = cp.Variable((num_inputs, num_outputs))
    sigma_matrix = dual_vars.T @ rho_diag
    constraints: list[cp.Constraint] = []
    for guess_index in range(num_guess_funcs):
        expr_matrix = coeffs[guess_index] - sigma_matrix
        constraints.append(cp.sum(cp.max(expr_matrix, axis=0)) <= 0)

    objective = cp.Minimize(cp.sum(cp.multiply(probabilities, dual_vars)))
    problem = cp.Problem(objective, constraints)
    solver_name, status = solve_cvxpy_problem(
        problem,
        preferred_solver=preferred_solver,
        verbose=verbose,
    )

    value = None if problem.value is None else float(np.real_if_close(problem.value))
    h_min = None
    if value is not None and value > 0 and status in ("optimal", "optimal_inaccurate"):
        h_min = float(-np.log2(value))

    result = {
        "route": "route4_phaseinsensitive_dual",
        "solver": solver_name,
        "status": status,
        "p_guess": value,
        "H_min": h_min,
        "num_guess_functions": int(num_guess_funcs),
    }
    result.update(_instance_summary(instance))
    return result


def solve_phaseinsensitive_primal(
    instance: dict[str, Any],
    preferred_solver: str | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = None,
) -> dict[str, Any]:
    """Primal SDP for the diagonal, phase-insensitive model."""
    probabilities = np.asarray(instance["probabilities"], dtype=float)
    rho_diag = np.asarray(instance["rho_diag"], dtype=float)
    q_selected = np.asarray(instance["q_selected"], dtype=float)
    num_inputs, num_outputs = probabilities.shape
    cutoff = rho_diag.shape[1]

    size_info = estimate_primal_problem_size(num_inputs, num_outputs, cutoff)
    if max_primal_variables is not None and size_info["variable_count"] > max_primal_variables:
        raise ValueError(
            "The requested primal instance is too large for the configured safeguard: "
            f"{size_info['variable_count']} > {max_primal_variables}."
        )
    if size_info["variable_count"] > 3_000_000:
        warnings.warn(
            "The primal route4 SDP is very large and may take a long time to canonicalize or solve. "
            f"Estimated variables: {size_info['variable_count']}.",
            stacklevel=2,
        )

    lambda_indices = np.array(
        list(iterproduct(range(num_outputs), repeat=num_inputs + 1)),
        dtype=int,
    )
    num_strategies = lambda_indices.shape[0]
    primal_elements = cp.Variable((cutoff, num_outputs, num_strategies), nonneg=True)

    objective_expr = 0
    for input_index in range(num_inputs):
        target_outputs = lambda_indices[:, input_index + 1]
        for output in range(num_outputs):
            strategy_ids = np.where(target_outputs == output)[0]
            if strategy_ids.size == 0:
                continue
            primal_sum = cp.sum(primal_elements[:, output, strategy_ids], axis=1)
            objective_expr += q_selected[input_index] * (rho_diag[input_index, :] @ primal_sum)

    constraints: list[cp.Constraint] = []
    sum_over_outputs = cp.sum(primal_elements, axis=1)
    for strategy_id in range(num_strategies):
        vec = sum_over_outputs[:, strategy_id]
        constraints.append(vec[1:] == vec[:-1])

    total_elements = cp.sum(primal_elements, axis=2)
    for input_index in range(num_inputs):
        for output in range(num_outputs):
            constraints.append(
                rho_diag[input_index, :] @ total_elements[:, output] == probabilities[input_index, output]
            )

    problem = cp.Problem(cp.Maximize(objective_expr), constraints)
    solver_name, status = solve_cvxpy_problem(
        problem,
        preferred_solver=preferred_solver,
        verbose=verbose,
    )

    value = None if problem.value is None else float(np.real_if_close(problem.value))
    h_min = None
    if value is not None and value > 0 and status in ("optimal", "optimal_inaccurate"):
        h_min = float(-np.log2(value))

    result = {
        "route": "route4_phaseinsensitive_primal",
        "solver": solver_name,
        "status": status,
        "p_guess": value,
        "H_min": h_min,
    }
    result.update(size_info)
    result.update(_instance_summary(instance))
    return result


def run_route4_dual(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
) -> dict[str, Any]:
    instance = prepare_phaseinsensitive_instance(
        num_outputs=num_outputs,
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        cutoff=cutoff,
        prob_floor=prob_floor,
        shift=shift,
        probability_path=probability_path,
    )
    return solve_phaseinsensitive_dual(
        instance,
        preferred_solver=preferred_solver,
        verbose=verbose,
    )


def run_route4_primal(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    max_primal_variables: int | None = 3_000_000,
) -> dict[str, Any]:
    instance = prepare_phaseinsensitive_instance(
        num_outputs=num_outputs,
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        cutoff=cutoff,
        prob_floor=prob_floor,
        shift=shift,
        probability_path=probability_path,
    )
    return solve_phaseinsensitive_primal(
        instance,
        preferred_solver=preferred_solver,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )


def compare_route4_primal_dual(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    max_primal_variables: int | None = 3_000_000,
) -> dict[str, Any]:
    instance = prepare_phaseinsensitive_instance(
        num_outputs=num_outputs,
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        cutoff=cutoff,
        prob_floor=prob_floor,
        shift=shift,
        probability_path=probability_path,
    )
    return {
        "route": "route4_phaseinsensitive_compare",
        "instance": _instance_summary(instance),
        "dual": solve_phaseinsensitive_dual(
            instance,
            preferred_solver=preferred_solver,
            verbose=verbose,
        ),
        "primal": solve_phaseinsensitive_primal(
            instance,
            preferred_solver=preferred_solver,
            verbose=verbose,
            max_primal_variables=max_primal_variables,
        ),
    }


def sweep_route4_outputs(
    output_values: list[int],
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    return [
        run_route4_dual(
            num_outputs=num_outputs,
            selected_mu_list=selected_mu_list,
            q_selected=q_selected,
            cutoff=cutoff,
            prob_floor=prob_floor,
            shift=shift,
            preferred_solver=preferred_solver,
            verbose=verbose,
            probability_path=probability_path,
        )
        for num_outputs in output_values
    ]


def search_route4_triplets(
    num_outputs: int,
    subset_size: int = 3,
    certify_top_k: int = 3,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    full_mu: list[int] | tuple[int, ...] = FULL_MU,
) -> dict[str, Any]:
    """Search subsets of input intensities, then certify only the most promising ones."""
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    candidates: list[dict[str, Any]] = []
    for subset in combinations(list(full_mu), subset_size):
        q_selected = np.full(subset_size, 1.0 / subset_size)
        instance = prepare_phaseinsensitive_instance(
            num_outputs=num_outputs,
            selected_mu_list=list(subset),
            q_selected=q_selected.tolist(),
            cutoff=cutoff,
            prob_floor=prob_floor,
            shift=shift,
            probability_path=probability_path,
            full_mu=full_mu,
        )
        candidates.append(
            {
                "selected_mu_list": list(subset),
                "q_selected": q_selected.tolist(),
                "distribution_only_p_guess": float(instance["distribution_only_p_guess"]),
                "distribution_only_H_min": float(-np.log2(instance["distribution_only_p_guess"])),
                "distribution_only_p_guess_raw": float(instance["distribution_only_p_guess_raw"]),
                "distribution_only_H_min_raw": float(-np.log2(instance["distribution_only_p_guess_raw"])),
                "mixed_zero_columns_raw": list(instance["mixed_zero_columns_raw"]),
            }
        )

    candidates.sort(key=lambda item: item["distribution_only_p_guess"])
    certified: list[dict[str, Any]] = []
    for item in candidates[: max(certify_top_k, 0)]:
        certified.append(
            run_route4_dual(
                num_outputs=num_outputs,
                selected_mu_list=item["selected_mu_list"],
                q_selected=item["q_selected"],
                cutoff=cutoff,
                prob_floor=prob_floor,
                shift=shift,
                preferred_solver=preferred_solver,
                verbose=verbose,
                probability_path=probability_path,
            )
        )

    return {
        "route": "route4_phaseinsensitive_subset_search",
        "num_outputs": num_outputs,
        "subset_size": subset_size,
        "certify_top_k": certify_top_k,
        "num_candidates": len(candidates),
        "best_distribution_only": candidates[0] if candidates else None,
        "top_distribution_only": candidates[: min(len(candidates), max(certify_top_k, 5))],
        "certified": certified,
    }
