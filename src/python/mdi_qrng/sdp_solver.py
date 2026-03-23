"""
MDI-QRNG SDP solver.

This module builds the primal SDP used to upper-bound the guessing
probability for the discretized CV Bell-measurement model.

The current formulation follows docs/SDP.md and assumes local
no-signalling constraints C3/C4 on the effective operators. Before
solving, we explicitly check whether the observed marginals are even
compatible with those assumptions. If not, we return a diagnostic
status instead of reporting a misleading solver failure.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from .states import sigma_index_to_signs


def encode_eve_guess(a: int, b: int, n_b: int) -> int:
    """Encode Eve's guess (a, b) into a single index."""
    return a * n_b + b


def decode_eve_guess(e: int, n_b: int) -> tuple[int, int]:
    """Decode Eve's flat index back into (a, b)."""
    return divmod(e, n_b)


def _build_kron_IA_MB(M_B: cp.Expression, zero_2x2: np.ndarray) -> cp.Expression:
    """Build I_A kron M_B as a 4x4 block matrix."""
    return cp.bmat([[M_B, zero_2x2], [zero_2x2, M_B]])


def _build_kron_MA_IB(M_A: cp.Expression, I2: np.ndarray) -> cp.Expression:
    """Build M_A kron I_B as a 4x4 block matrix."""
    return cp.bmat(
        [
            [M_A[0, 0] * I2, M_A[0, 1] * I2],
            [M_A[1, 0] * I2, M_A[1, 1] * I2],
        ]
    )


def _local_marginal_consistency_report(
    prob: np.ndarray,
    tol: float = 1e-9,
) -> list[str]:
    """Check the observable implications of C3/C4.

    C3 implies P(b|x,y) is independent of x after summing over a.
    C4 implies P(a|x,y) is independent of y after summing over b.
    """
    messages: list[str] = []

    marg_b = prob.sum(axis=2)  # (x, y, b)
    gap_c3 = np.abs(marg_b[0] - marg_b[1])
    max_gap_c3 = float(np.max(gap_c3))
    if max_gap_c3 > tol:
        worst_y, worst_b = np.unravel_index(np.argmax(gap_c3), gap_c3.shape)
        messages.append(
            "C3 would require P(b|x,y) to be independent of x, "
            f"but the largest violation is {max_gap_c3:.6e} at "
            f"y={worst_y}, b={worst_b}."
        )

    marg_a = prob.sum(axis=3)  # (x, y, a)
    gap_c4 = np.abs(marg_a[:, 0] - marg_a[:, 1])
    max_gap_c4 = float(np.max(gap_c4))
    if max_gap_c4 > tol:
        worst_x, worst_a = np.unravel_index(np.argmax(gap_c4), gap_c4.shape)
        messages.append(
            "C4 would require P(a|x,y) to be independent of y, "
            f"but the largest violation is {max_gap_c4:.6e} at "
            f"x={worst_x}, a={worst_a}."
        )

    return messages


def solve_mdi_qrng_sdp(
    prob: np.ndarray,
    rho: dict[tuple[int, int], np.ndarray],
    n_a: int,
    n_b: int,
    x_star: int = 0,
    y_star: int = 0,
    solver: str | None = None,
    verbose: bool = True,
) -> dict:
    """Build and solve the MDI-QRNG SDP."""
    n_e = n_a * n_b

    consistency_issues = _local_marginal_consistency_report(prob)
    if consistency_issues:
        diagnostic = (
            "Observed probabilities are inconsistent with the local "
            "no-signalling assumptions encoded by C3/C4.\n"
            + "\n".join(f"- {issue}" for issue in consistency_issues)
            + "\nThis is a modelling issue, not a solver issue."
        )
        print("\nModel diagnostic:")
        print(diagnostic)
        return {
            "p_guess": None,
            "H_min": None,
            "status": "model_inconsistent",
            "solve_time": 0.0,
            "compile_time": 0.0,
            "diagnostic": diagnostic,
            "p_e_values": None,
        }

    print(f"SDP size: n_a={n_a}, n_b={n_b}, n_e={n_e}")
    print(f"  Main 4x4 PSD blocks: {n_a * n_b * n_e}")
    print(f"  Aux 2x2 PSD blocks: {n_b * n_e + n_a * n_e}")
    print(f"  Scalar vars: {n_e}")

    M_tilde = {
        (a, b, e): cp.Variable((4, 4), symmetric=True)
        for a in range(n_a)
        for b in range(n_b)
        for e in range(n_e)
    }
    M_B = {
        (b, e): cp.Variable((2, 2), symmetric=True)
        for b in range(n_b)
        for e in range(n_e)
    }
    M_A = {
        (a, e): cp.Variable((2, 2), symmetric=True)
        for a in range(n_a)
        for e in range(n_e)
    }
    p_e = cp.Variable(n_e, nonneg=True)

    s1_star, s2_star = sigma_index_to_signs(x_star, y_star)
    rho_star = rho[(s1_star, s2_star)]
    objective = cp.Maximize(
        cp.sum(
            [
                cp.trace(M_tilde[(a, b, encode_eve_guess(a, b, n_b))] @ rho_star)
                for a in range(n_a)
                for b in range(n_b)
            ]
        )
    )

    constraints: list[cp.Constraint] = []
    I2 = np.eye(2)
    Z2 = np.zeros((2, 2))

    for a in range(n_a):
        for b in range(n_b):
            for e in range(n_e):
                constraints.append(M_tilde[(a, b, e)] >> 0)

    for b in range(n_b):
        for e in range(n_e):
            constraints.append(M_B[(b, e)] >> 0)

    for a in range(n_a):
        for e in range(n_e):
            constraints.append(M_A[(a, e)] >> 0)

    for x in range(2):
        for y in range(2):
            s1, s2 = sigma_index_to_signs(x, y)
            rho_xy = rho[(s1, s2)]
            for a in range(n_a):
                for b in range(n_b):
                    constraints.append(
                        cp.sum(
                            [
                                cp.trace(M_tilde[(a, b, e)] @ rho_xy)
                                for e in range(n_e)
                            ]
                        )
                        == prob[x, y, a, b]
                    )

    for b in range(n_b):
        for e in range(n_e):
            constraints.append(
                cp.sum([M_tilde[(a, b, e)] for a in range(n_a)])
                == _build_kron_IA_MB(M_B[(b, e)], Z2)
            )

    for a in range(n_a):
        for e in range(n_e):
            constraints.append(
                cp.sum([M_tilde[(a, b, e)] for b in range(n_b)])
                == _build_kron_MA_IB(M_A[(a, e)], I2)
            )

    for e in range(n_e):
        constraints.append(cp.sum([M_B[(b, e)] for b in range(n_b)]) == p_e[e] * I2)
        constraints.append(cp.sum([M_A[(a, e)] for a in range(n_a)]) == p_e[e] * I2)

    constraints.append(cp.sum(p_e) == 1)

    problem = cp.Problem(objective, constraints)

    print("\nCompiling SDP problem...")
    t_start = time.time()

    if solver is not None:
        solvers_to_try = [(solver, {})]
    else:
        solvers_to_try = [
            (cp.MOSEK, {}),
            (
                cp.SCS,
                {
                    "max_iters": 20000,
                    "eps_abs": 1e-5,
                    "eps_rel": 1e-5,
                    "eps_infeas": 1e-7,
                },
            ),
            (cp.CVXOPT, {}),
        ]

    result: dict | None = None
    for slv, opts in solvers_to_try:
        try:
            print(f"Trying solver: {slv}")
            problem.solve(solver=slv, verbose=verbose, **opts)
            result = {"solver": str(slv), "status": problem.status}
            break
        except (cp.error.SolverError, Exception) as exc:
            print(f"Solver {slv} failed: {exc}")

    t_total = time.time() - t_start

    if result is None:
        return {
            "p_guess": None,
            "H_min": None,
            "status": "all_solvers_failed",
            "solve_time": t_total,
            "compile_time": 0.0,
            "diagnostic": None,
            "p_e_values": None,
        }

    if problem.status in ("optimal", "optimal_inaccurate"):
        p_guess = problem.value
        H_min = -np.log2(p_guess) if p_guess > 0 else float("inf")
    else:
        p_guess = problem.value
        H_min = None

    return {
        "p_guess": p_guess,
        "H_min": H_min,
        "status": problem.status,
        "solve_time": t_total,
        "compile_time": 0.0,
        "diagnostic": None,
        "p_e_values": p_e.value if p_e.value is not None else None,
    }
