from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable

import cvxpy as cp
import numpy as np
from scipy.linalg import expm
from scipy.special import eval_hermite, roots_hermite


def destroy(dimension: int) -> np.ndarray:
    """Annihilation operator in the Fock basis."""
    op = np.zeros((dimension, dimension), dtype=complex)
    for n in range(1, dimension):
        op[n - 1, n] = math.sqrt(n)
    return op


def create(dimension: int) -> np.ndarray:
    """Creation operator in the Fock basis."""
    return destroy(dimension).conj().T


def kron(*operators: np.ndarray) -> np.ndarray:
    """Kronecker product of many operators."""
    out = np.array([[1.0 + 0.0j]])
    for op in operators:
        out = np.kron(out, op)
    return out


def partial_trace(rho: np.ndarray, dims: Iterable[int], keep: Iterable[int]) -> np.ndarray:
    """Partial trace over subsystems not in keep."""
    dims = list(dims)
    keep = list(keep)
    trace = [i for i in range(len(dims)) if i not in keep]
    rho_t = rho.reshape(*(dims + dims))
    order = keep + trace + [len(dims) + i for i in keep] + [len(dims) + i for i in trace]
    rho_perm = rho_t.transpose(order)
    dim_keep = int(np.prod([dims[i] for i in keep]))
    dim_trace = int(np.prod([dims[i] for i in trace])) if trace else 1
    rho_perm = rho_perm.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
    return np.einsum("ikjk->ij", rho_perm)


def coherent_state(dimension: int, alpha: complex) -> np.ndarray:
    """Truncated coherent-state ket in the Fock basis."""
    coeffs = np.zeros(dimension, dtype=complex)
    prefactor = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(dimension):
        coeffs[n] = prefactor * alpha**n / math.sqrt(math.factorial(n))
    norm = np.linalg.norm(coeffs)
    if norm > 0:
        coeffs /= norm
    return coeffs


def density_from_ket(ket: np.ndarray) -> np.ndarray:
    """Density matrix of a pure state."""
    return np.outer(ket, ket.conj())


def single_mode_squeezed_vacuum(dimension: int, squeezing_db: float) -> np.ndarray:
    """Single-mode squeezed vacuum ket in the Fock basis."""
    r = -0.5 * np.log(10 ** (squeezing_db / 10.0))
    a = destroy(dimension)
    adag = a.conj().T
    generator = 0.5 * r * (a @ a - adag @ adag)
    squeeze = expm(generator)
    vacuum = np.zeros(dimension, dtype=complex)
    vacuum[0] = 1.0
    ket = squeeze @ vacuum
    ket /= np.linalg.norm(ket)
    return ket


def balanced_beamsplitter_unitary(dimension: int) -> np.ndarray:
    """Balanced beamsplitter on two truncated modes."""
    a = destroy(dimension)
    b = destroy(dimension)
    generator = (np.pi / 4.0) * (kron(create(dimension), b) - kron(a, create(dimension)))
    return expm(generator)


def tmsv_density(dimension: int, squeezing_db: float) -> np.ndarray:
    """Two-mode squeezed vacuum in a truncated Fock space."""
    r = -0.5 * np.log(10 ** (squeezing_db / 10.0))
    lam = np.tanh(r)
    ket = np.zeros(dimension * dimension, dtype=complex)
    prefactor = math.sqrt(1.0 - lam**2)
    for n in range(dimension):
        ket[n * dimension + n] = prefactor * lam**n
    ket /= np.linalg.norm(ket)
    return density_from_ket(ket)


def split_sms_density(dimension: int, squeezing_db: float) -> np.ndarray:
    """Single-mode squeezed vacuum split on a balanced beamsplitter."""
    sms = single_mode_squeezed_vacuum(dimension, squeezing_db)
    vacuum = np.zeros(dimension, dtype=complex)
    vacuum[0] = 1.0
    ket_in = np.kron(sms, vacuum)
    unitary = balanced_beamsplitter_unitary(dimension)
    ket_out = unitary @ ket_in
    ket_out /= np.linalg.norm(ket_out)
    return density_from_ket(ket_out)


def loss_kraus_1mode(dimension: int, eta: float) -> list[np.ndarray]:
    """Pure-loss channel Kraus operators."""
    kraus = []
    for k in range(dimension):
        op = np.zeros((dimension, dimension), dtype=complex)
        for n in range(k, dimension):
            coeff = math.sqrt(math.comb(n, k) * (1.0 - eta) ** k * eta ** (n - k))
            op[n - k, n] = coeff
        kraus.append(op)
    return kraus


def apply_symmetric_loss(rho_ab: np.ndarray, dimension: int, eta: float) -> np.ndarray:
    """Apply the same pure loss to both modes."""
    out = np.zeros_like(rho_ab, dtype=complex)
    kraus_a = loss_kraus_1mode(dimension, eta)
    kraus_b = loss_kraus_1mode(dimension, eta)
    for ka in kraus_a:
        for kb in kraus_b:
            op = kron(ka, kb)
            out += op @ rho_ab @ op.conj().T
    return out


def quadrature_op(dimension: int, theta: float) -> np.ndarray:
    """Quadrature operator x_theta."""
    a = destroy(dimension)
    adag = a.conj().T
    return (np.exp(-1j * theta) * a + np.exp(1j * theta) * adag) / np.sqrt(2.0)


@lru_cache(maxsize=None)
def quadrature_hermite_data(dimension: int, num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and normalized polynomial factors for quadrature integrals."""
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    nodes, weights = roots_hermite(num_nodes)
    values = np.zeros((dimension, num_nodes), dtype=float)
    prefactor = np.pi ** (-0.25)
    for n in range(dimension):
        norm = prefactor / math.sqrt((2.0**n) * math.factorial(n))
        values[n, :] = norm * eval_hermite(n, nodes)
    return nodes, weights, values


def complete_povm_via_whitening(
    povm: list[np.ndarray],
    min_eigenvalue: float = 1e-12,
) -> list[np.ndarray]:
    """Numerically enforce POVM completeness after quadrature integration."""
    total = sum(povm)
    total = 0.5 * (total + total.conj().T)
    values, basis = np.linalg.eigh(total)
    clipped = np.maximum(values, min_eigenvalue)
    inv_sqrt = basis @ np.diag(1.0 / np.sqrt(clipped)) @ basis.conj().T
    corrected = [inv_sqrt @ element @ inv_sqrt.conj().T for element in povm]
    return [0.5 * (element + element.conj().T) for element in corrected]


def quadrature_povms_from_node_masks(
    dimension: int,
    theta: float,
    node_masks: np.ndarray,
    num_nodes: int = 400,
    enforce_completeness: bool = True,
) -> list[np.ndarray]:
    """Construct coarse-grained quadrature POVMs from node-wise mask values."""
    masks = np.asarray(node_masks, dtype=float)
    if masks.ndim != 2:
        raise ValueError("node_masks must have shape (num_bins, num_nodes).")

    _, weights, values = quadrature_hermite_data(dimension, num_nodes)
    if masks.shape[1] != weights.size:
        raise ValueError(
            f"node_masks has {masks.shape[1]} columns but num_nodes={num_nodes} provides {weights.size} nodes."
        )

    weighted_values = values * np.sqrt(weights)[None, :]
    base_elements: list[np.ndarray] = []
    for mask in masks:
        masked_values = weighted_values * np.sqrt(mask)[None, :]
        base_elements.append(masked_values @ masked_values.T)

    number_indices = np.arange(dimension, dtype=float)
    phase = np.exp(-1j * theta * number_indices)
    rotated = [
        (phase[:, None] * element) * phase.conj()[None, :]
        for element in base_elements
    ]

    if enforce_completeness:
        return complete_povm_via_whitening(rotated)
    return [0.5 * (element + element.conj().T) for element in rotated]


def operator_span_rank(states: list[np.ndarray], tol: float = 1e-9) -> int:
    """Rank of the linear span of operators."""
    matrix = np.stack([state.reshape(-1) for state in states], axis=0)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def support_basis(vectors: list[np.ndarray], tol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis for the span of the given kets."""
    stacked = np.column_stack(vectors)
    u, singular_values, _ = np.linalg.svd(stacked, full_matrices=False)
    rank = int(np.sum(singular_values > tol))
    return u[:, :rank]


def project_density_to_basis(rho: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project a density matrix to a reduced orthonormal basis."""
    return basis.conj().T @ rho @ basis


def choose_solvers(preferred: str | None = None) -> list[tuple[object, dict]]:
    """Ordered solver list with stable SCS defaults."""
    if preferred is not None:
        return [(preferred, {})]
    solvers: list[tuple[object, dict]] = []
    installed = set(cp.installed_solvers())
    if "MOSEK" in installed:
        solvers.append((cp.MOSEK, {}))
    solvers.append(
        (
            cp.SCS,
            {
                "max_iters": 20000,
                "eps_abs": 1e-5,
                "eps_rel": 1e-5,
                "eps_infeas": 1e-7,
            },
        )
    )
    if "CVXOPT" in installed:
        solvers.append((cp.CVXOPT, {}))
    return solvers


def solve_cvxpy_problem(
    problem: cp.Problem,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> tuple[str, str]:
    """Solve with a small fallback chain."""
    errors: list[str] = []
    for solver, options in choose_solvers(preferred_solver):
        try:
            problem.solve(solver=solver, verbose=verbose, **options)
            return str(solver), problem.status
        except (cp.error.SolverError, Exception) as exc:
            errors.append(f"{solver}: {exc}")
    raise RuntimeError("All solvers failed: " + " | ".join(errors))


def guessing_prob_single_device(
    input_states: list[np.ndarray],
    probabilities: np.ndarray,
    target_input: int,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> dict:
    """Single-device prepare-and-measure MDI guessing-probability SDP."""
    num_inputs = len(input_states)
    num_outputs = probabilities.shape[1]
    dimension = input_states[0].shape[0]
    identity = np.eye(dimension)

    operators = {
        (c, e): cp.Variable((dimension, dimension), hermitian=True)
        for c in range(num_outputs)
        for e in range(num_outputs)
    }
    p_e = cp.Variable(num_outputs, nonneg=True)

    constraints: list[cp.Constraint] = []
    for c in range(num_outputs):
        for e in range(num_outputs):
            constraints.append(operators[(c, e)] >> 0)

    for s in range(num_inputs):
        rho_s = input_states[s]
        for c in range(num_outputs):
            constraints.append(
                cp.sum(
                    [cp.real(cp.trace(operators[(c, e)] @ rho_s)) for e in range(num_outputs)]
                )
                == probabilities[s, c]
            )

    for e in range(num_outputs):
        constraints.append(
            sum(operators[(c, e)] for c in range(num_outputs)) == p_e[e] * identity
        )

    constraints.append(cp.sum(p_e) == 1)

    rho_star = input_states[target_input]
    objective = cp.Maximize(
        cp.sum(
            [
                cp.real(cp.trace(operators[(c, c)] @ rho_star))
                for c in range(num_outputs)
            ]
        )
    )

    problem = cp.Problem(objective, constraints)
    solver_name, status = solve_cvxpy_problem(
        problem,
        preferred_solver=preferred_solver,
        verbose=verbose,
    )

    value = problem.value
    h_min = None
    if value is not None and value > 0 and status in ("optimal", "optimal_inaccurate"):
        h_min = float(-np.log2(value))

    return {
        "solver": solver_name,
        "status": status,
        "p_guess": None if value is None else float(np.real_if_close(value)),
        "H_min": h_min,
    }
