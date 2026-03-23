from __future__ import annotations

import cvxpy as cp
import numpy as np

from ..common import (
    apply_symmetric_loss,
    kron,
    partial_trace,
    quadrature_hermite_data,
    quadrature_op,
    quadrature_povms_from_node_masks,
    solve_cvxpy_problem,
    split_sms_density,
    tmsv_density,
)


def default_quadrature_nodes(dimension: int) -> int:
    """Stable quadrature integration grid for coarse-grained POVMs."""
    return max(400, 60 * dimension)


def periodic_binning_povms(
    dimension: int,
    theta: float,
    period: float,
    num_bins: int,
    num_nodes: int | None = None,
) -> list[np.ndarray]:
    """Periodic coarse-grained quadrature POVMs built from the continuous mask."""
    if period <= 0:
        raise ValueError("period must be positive.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    nodes_count = default_quadrature_nodes(dimension) if num_nodes is None else num_nodes
    nodes, _, _ = quadrature_hermite_data(dimension, nodes_count)
    width = period / num_bins
    reduced = np.mod(nodes, period)
    bin_ids = np.floor(reduced / width).astype(int)
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    masks = np.zeros((num_bins, nodes.size), dtype=float)
    masks[bin_ids, np.arange(nodes.size)] = 1.0
    return quadrature_povms_from_node_masks(
        dimension,
        theta,
        masks,
        num_nodes=nodes_count,
    )


def nonperiodic_binning_povms(
    dimension: int,
    theta: float,
    num_bins: int,
    cutoff_range: float,
    num_nodes: int | None = None,
) -> list[np.ndarray]:
    """Bob's finite-range homodyne bins with an explicit outside bin."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    if cutoff_range <= 0:
        raise ValueError("cutoff_range must be positive.")

    nodes_count = default_quadrature_nodes(dimension) if num_nodes is None else num_nodes
    nodes, _, _ = quadrature_hermite_data(dimension, nodes_count)
    central_bins = num_bins - 1
    edges = np.linspace(-cutoff_range, cutoff_range, central_bins + 1)
    inside = (nodes >= -cutoff_range) & (nodes <= cutoff_range)
    central_ids = np.searchsorted(edges, nodes, side="right") - 1
    central_ids = np.clip(central_ids, 0, central_bins - 1)

    masks = np.zeros((num_bins, nodes.size), dtype=float)
    masks[central_ids[inside], np.arange(nodes.size)[inside]] = 1.0
    masks[-1, ~inside] = 1.0
    return quadrature_povms_from_node_masks(
        dimension,
        theta,
        masks,
        num_nodes=nodes_count,
    )


def estimate_bob_range(rho_ab: np.ndarray, dimension: int) -> float:
    """Finite range r = 5 sigma using the largest local quadrature variance."""
    identity = np.eye(dimension, dtype=complex)
    quadratures = (
        kron(quadrature_op(dimension, 0.0), identity),
        kron(quadrature_op(dimension, np.pi / 2.0), identity),
        kron(identity, quadrature_op(dimension, 0.0)),
        kron(identity, quadrature_op(dimension, np.pi / 2.0)),
    )
    variances: list[float] = []
    for observable in quadratures:
        mean = float(np.real(np.trace(observable @ rho_ab)))
        second_moment = float(np.real(np.trace(observable @ observable @ rho_ab)))
        variances.append(max(second_moment - mean**2, 1e-12))
    return 5.0 * np.sqrt(max(variances))


def bob_homodyne_angles(num_bob_settings: int) -> list[float]:
    """Angles equally spaced between q and p, inclusive."""
    if num_bob_settings <= 1:
        return [0.0]
    return list(np.linspace(0.0, np.pi / 2.0, num_bob_settings))


def assemblage_tomography(
    rho_ab: np.ndarray,
    alice_povms: list[list[np.ndarray]],
) -> list[list[np.ndarray]]:
    """Observed assemblage sigma[a|x] = Tr_A[(M_a|x \\otimes I) rho]."""
    dimension = alice_povms[0][0].shape[0]
    identity_b = np.eye(dimension, dtype=complex)
    out: list[list[np.ndarray]] = []
    for row_povms in alice_povms:
        row: list[np.ndarray] = []
        for meas in row_povms:
            row.append(
                partial_trace(
                    kron(meas, identity_b) @ rho_ab,
                    [dimension, dimension],
                    keep=[1],
                )
            )
        out.append(row)
    return out


def alice_periodic_povms(
    dimension: int,
    period_q: float,
    num_alice_bins: int,
    num_nodes: int | None = None,
) -> tuple[list[list[np.ndarray]], float]:
    """Alice's q/p periodic bins with T_p fixed by the mutual-unbiasedness rule."""
    width_q = period_q / num_alice_bins
    period_p = 2.0 * np.pi / width_q
    alice_q = periodic_binning_povms(
        dimension,
        0.0,
        period_q,
        num_alice_bins,
        num_nodes=num_nodes,
    )
    alice_p = periodic_binning_povms(
        dimension,
        np.pi / 2.0,
        period_p,
        num_alice_bins,
        num_nodes=num_nodes,
    )
    return [alice_q, alice_p], float(period_p)


def joint_probabilities(
    rho_ab: np.ndarray,
    alice_povms: list[list[np.ndarray]],
    bob_povms: list[list[np.ndarray]],
) -> np.ndarray:
    """Joint probabilities P[x, a, y, b]."""
    m_x = len(alice_povms)
    o_a = len(alice_povms[0])
    m_y = len(bob_povms)
    o_b = len(bob_povms[0])
    out = np.zeros((m_x, o_a, m_y, o_b))
    for x in range(m_x):
        for a in range(o_a):
            for y in range(m_y):
                for b in range(o_b):
                    out[x, a, y, b] = float(
                        np.real(np.trace(kron(alice_povms[x][a], bob_povms[y][b]) @ rho_ab))
                    )
    return out


def guessing_prob_sdp_tomography(
    sigma_obs: list[list[np.ndarray]],
    x_star: int = 0,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> dict:
    """Ioannou et al. tomography SDP with the no-signalling bug fixed."""
    num_inputs = len(sigma_obs)
    num_outputs = len(sigma_obs[0])
    dimension = sigma_obs[0][0].shape[0]

    sigma_e = {
        (e, a, x): cp.Variable((dimension, dimension), hermitian=True)
        for e in range(num_outputs)
        for a in range(num_outputs)
        for x in range(num_inputs)
    }
    constraints: list[cp.Constraint] = []

    for x in range(num_inputs):
        for a in range(num_outputs):
            constraints.append(
                sum(sigma_e[(e, a, x)] for e in range(num_outputs)) == sigma_obs[x][a]
            )

    for e in range(num_outputs):
        reference = sum(sigma_e[(e, a, 0)] for a in range(num_outputs))
        for x in range(1, num_inputs):
            constraints.append(sum(sigma_e[(e, a, x)] for a in range(num_outputs)) == reference)

    for e in range(num_outputs):
        for a in range(num_outputs):
            for x in range(num_inputs):
                constraints.append(sigma_e[(e, a, x)] >> 0)

    objective = cp.Maximize(
        cp.real(sum(cp.trace(sigma_e[(e, e, x_star)]) for e in range(num_outputs)))
    )
    problem = cp.Problem(objective, constraints)
    solver_name, status = solve_cvxpy_problem(problem, preferred_solver, verbose)

    value = problem.value
    return {
        "solver": solver_name,
        "status": status,
        "p_guess": None if value is None else float(np.real_if_close(value)),
        "H_min": None
        if value is None or value <= 0 or status not in ("optimal", "optimal_inaccurate")
        else float(-np.log2(value)),
    }


def guessing_prob_sdp_homodyne(
    probabilities: np.ndarray,
    bob_povms: list[list[np.ndarray]],
    x_star: int = 0,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> dict:
    """Steering SDP when Bob uses trusted binned homodyne measurements."""
    num_inputs, num_outputs, num_bob_settings, num_bob_bins = probabilities.shape
    dimension = bob_povms[0][0].shape[0]

    sigma_e = {
        (e, a, x): cp.Variable((dimension, dimension), hermitian=True)
        for e in range(num_outputs)
        for a in range(num_outputs)
        for x in range(num_inputs)
    }
    constraints: list[cp.Constraint] = []

    for x in range(num_inputs):
        for a in range(num_outputs):
            sigma_ax = sum(sigma_e[(e, a, x)] for e in range(num_outputs))
            for y in range(num_bob_settings):
                for b in range(num_bob_bins):
                    constraints.append(
                        cp.real(cp.trace(bob_povms[y][b] @ sigma_ax)) == probabilities[x, a, y, b]
                    )

    for e in range(num_outputs):
        reference = sum(sigma_e[(e, a, 0)] for a in range(num_outputs))
        for x in range(1, num_inputs):
            constraints.append(sum(sigma_e[(e, a, x)] for a in range(num_outputs)) == reference)

    for e in range(num_outputs):
        for a in range(num_outputs):
            for x in range(num_inputs):
                constraints.append(sigma_e[(e, a, x)] >> 0)

    objective = cp.Maximize(
        cp.real(sum(cp.trace(sigma_e[(e, e, x_star)]) for e in range(num_outputs)))
    )
    problem = cp.Problem(objective, constraints)
    solver_name, status = solve_cvxpy_problem(problem, preferred_solver, verbose)

    value = problem.value
    return {
        "solver": solver_name,
        "status": status,
        "p_guess": None if value is None else float(np.real_if_close(value)),
        "H_min": None
        if value is None or value <= 0 or status not in ("optimal", "optimal_inaccurate")
        else float(-np.log2(value)),
    }


def build_source_state(source: str, dimension: int, squeezing_db: float, eta: float) -> np.ndarray:
    """Supported Gaussian sources for route 1."""
    if source == "tmsv":
        rho = tmsv_density(dimension, squeezing_db)
    elif source == "split_sms":
        rho = split_sms_density(dimension, squeezing_db)
    else:
        raise ValueError(f"Unsupported source: {source}")
    return apply_symmetric_loss(rho, dimension, eta)


def run_route1(
    source: str = "tmsv",
    bob_mode: str = "homodyne",
    dimension: int = 8,
    squeezing_db: float = -4.0,
    eta: float = 0.9,
    num_alice_bins: int = 8,
    num_bob_bins: int = 12,
    num_bob_settings: int = 2,
    tq_grid: np.ndarray | None = None,
    num_quadrature_nodes: int | None = None,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> dict:
    """Prototype scan for the 2022 steering route."""
    if tq_grid is None:
        tq_grid = np.linspace(2.0, 10.0, 5)

    rho = build_source_state(source, dimension, squeezing_db, eta)
    bob_angles = bob_homodyne_angles(num_bob_settings)
    bob_range = estimate_bob_range(rho, dimension)
    bob_povms = [
        nonperiodic_binning_povms(
            dimension,
            angle,
            num_bob_bins,
            bob_range,
            num_nodes=num_quadrature_nodes,
        )
        for angle in bob_angles
    ]

    best: dict | None = None
    for tq in tq_grid:
        alice_povms, tp = alice_periodic_povms(
            dimension,
            float(tq),
            num_alice_bins,
            num_nodes=num_quadrature_nodes,
        )
        if bob_mode == "tomography":
            assemblage = assemblage_tomography(rho, alice_povms)
            current = guessing_prob_sdp_tomography(
                assemblage,
                x_star=0,
                preferred_solver=preferred_solver,
                verbose=verbose,
            )
        elif bob_mode == "homodyne":
            probabilities = joint_probabilities(rho, alice_povms, bob_povms)
            current = guessing_prob_sdp_homodyne(
                probabilities,
                bob_povms,
                x_star=0,
                preferred_solver=preferred_solver,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unsupported bob_mode: {bob_mode}")

        current["T_q"] = float(tq)
        current["T_p"] = float(tp)
        if best is None or (current["H_min"] or -np.inf) > (best["H_min"] or -np.inf):
            best = current

    assert best is not None
    best.update(
        {
            "route": "route1_steering",
            "source": source,
            "bob_mode": bob_mode,
            "dimension": dimension,
            "squeezing_db": squeezing_db,
            "eta": eta,
            "num_alice_bins": num_alice_bins,
            "num_bob_bins": num_bob_bins,
            "num_bob_settings": num_bob_settings,
            "bob_range": float(bob_range),
            "num_quadrature_nodes": default_quadrature_nodes(dimension)
            if num_quadrature_nodes is None
            else int(num_quadrature_nodes),
        }
    )
    return best


def sweep_route1_eta(
    source: str,
    bob_mode: str,
    eta_values: np.ndarray,
    dimension: int,
    squeezing_db: float,
    num_alice_bins: int,
    num_bob_bins: int,
    tq_grid: np.ndarray,
    num_bob_settings: int = 2,
    num_quadrature_nodes: int | None = None,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Sweep route 1 over eta values."""
    results = []
    for eta in eta_values:
        results.append(
            run_route1(
                source=source,
                bob_mode=bob_mode,
                dimension=dimension,
                squeezing_db=squeezing_db,
                eta=float(eta),
                num_alice_bins=num_alice_bins,
                num_bob_bins=num_bob_bins,
                num_bob_settings=num_bob_settings,
                tq_grid=tq_grid,
                num_quadrature_nodes=num_quadrature_nodes,
                preferred_solver=preferred_solver,
                verbose=verbose,
            )
        )
    return results
