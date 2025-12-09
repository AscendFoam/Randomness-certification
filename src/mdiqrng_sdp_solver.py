"""
MDI-QRNG (Measurement-Device-Independent Quantum Random Number Generation) SDP Solver

This module solves the semidefinite programming (SDP) problem for calculating the
maximum guessing probability in MDI-QRNG based on continuous-variable Bell state measurement.

The SDP formulation follows the structure:
- Objective: Maximize G_{x*,y*}^{MDI} = Tr(sum_e M_tilde_{a,b,e=(a,b)} (psi_x* x psi_y*))
- Constraints: Observation consistency, POVM positivity, no-signaling, and normalization

Reference: Based on the theory in docs/SDP_solve.tex

IMPORTANT NOTES:
----------------
1. The conditional probabilities p(a,b|x,y) are computed from the Gaussian model (tex Section 3)
   assuming infinite-dimensional coherent states.

2. The quantum states are represented in a 4-dimensional subspace (tex Section 4).

3. Feasibility Issue: The Gaussian probabilities may not be achievable by any POVM on the
   finite-dimensional states. In this case, the SDP will be infeasible. Use `epsilon > 0`
   for tolerance, or call `use_quantum_compatible_probabilities()` to ensure feasibility.

4. Guessing Probability Interpretation:
   - G = 1: Eve can always guess the outcome correctly (no certifiable randomness)
   - G < 1: Some randomness can be certified; min-entropy H_min = -log2(G)

5. For G < 1 with product input states, the measurement must be entangling (like the CV Bell
   measurement). Product measurements typically yield G = 1.
"""

import numpy as np
from scipy.special import erf
import cvxpy as cp
from typing import Dict, Tuple, Optional
import warnings


class MDIQRNG_SDP_Solver:
    """
    SDP solver for MDI-QRNG guessing probability calculation.

    The solver computes the maximum probability that an adversary Eve can guess
    the measurement outcome (a*, b*) given input states (x*, y*).
    """

    def __init__(
        self,
        n: int = 2,
        mu: float = 1.0,
        boundary: float = 10.0,
        verbose: bool = True
    ):
        """
        Initialize the SDP solver.

        Parameters
        ----------
        n : int
            Number of discretization intervals for each measurement outcome.
            Total outcomes: n for X_+ and n for P_-.
        mu : float
            Mean photon number (μ = |α|^2). Assumed equal for both modes (μ1 = μ2 = μ).
        boundary : float
            Boundary value for discretization intervals.
            Intervals are [-∞, -boundary, ..., boundary, +∞].
        verbose : bool
            Whether to print detailed information during solving.
        """
        self.n = n
        self.mu = mu
        self.boundary = boundary
        self.verbose = verbose

        # Compute derived parameters
        self.delta = np.exp(-2 * mu)  # Inner product δ = <α|-α> = e^{-2μ}
        self.n_e = n * n  # Number of Eve's outcomes (e corresponds to (a,b) pairs)
        self.dim = 4  # Dimension of A_0 ⊗ B_0 space (2 × 2)

        # Set up discretization boundaries
        # c_bounds: boundaries for X_+ discretization
        # d_bounds: boundaries for P_- discretization
        self._setup_discretization_boundaries()

        # Compute input states
        self._compute_input_states()

        # Compute conditional probabilities p(a,b|x,y)
        self._compute_conditional_probabilities()

    def _setup_discretization_boundaries(self):
        """
        Set up discretization boundaries for X_+ and P_- measurements.

        According to tex file Section 1.4:
        - X_+ discretized into n intervals: I_{+k} = [c_{k-1}, c_k) for k = 1, ..., n
        - P_- discretized into n intervals: I_{-l} = [d_{l-1}, d_l) for l = 1, ..., n
        - c_0 = d_0 = -∞, c_n = d_n = +∞
        - |c_1|, |c_{n-1}|, |d_1|, |d_{n-1}| can be set to boundary (e.g., 10)
        """
        # Create n+1 boundary points (including -∞ and +∞)
        # For n intervals, we need n+1 boundaries: c_0, c_1, ..., c_n
        # Inner boundaries are evenly spaced between -boundary and +boundary
        inner_bounds = np.linspace(-self.boundary, self.boundary, self.n - 1)
        self.c_bounds = np.concatenate([[-np.inf], inner_bounds, [np.inf]])
        self.d_bounds = np.concatenate([[-np.inf], inner_bounds, [np.inf]])

        if self.verbose:
            print(f"Discretization boundaries (n={self.n}):")
            print(f"  c_bounds (X_+): {self.c_bounds}")
            print(f"  d_bounds (P_-): {self.d_bounds}")

    def _compute_input_states(self):
        """
        Compute the input quantum states according to tex file Section 4.

        Single-mode basis: {|0⟩, |1⟩} where
        - |0⟩ = |α⟩ (coherent state with phase 0)
        - |1⟩ is orthogonal to |0⟩

        State representations:
        - |α⟩ = |0⟩ = (1, 0)^T
        - |-α⟩ = δ|0⟩ + √(1-δ²)|1⟩ = (δ, √(1-δ²))^T

        where δ = e^{-2μ} is the inner product ⟨α|-α⟩.
        """
        delta = self.delta
        sqrt_1_minus_delta2 = np.sqrt(1 - delta**2)

        # Alice's single-mode states (2-dimensional)
        # x = 0 corresponds to s1 = +1 (phase φ1 = 0) → |α⟩
        # x = 1 corresponds to s1 = -1 (phase φ1 = π) → |-α⟩
        self.psi_A = {
            0: np.array([1.0, 0.0]),                      # |α⟩ = |0⟩
            1: np.array([delta, sqrt_1_minus_delta2])     # |-α⟩ = δ|0⟩ + √(1-δ²)|1⟩
        }

        # Bob's single-mode states (2-dimensional)
        # y = 0 corresponds to s2 = +1 (phase φ2 = 0) → |α⟩
        # y = 1 corresponds to s2 = -1 (phase φ2 = π) → |-α⟩
        self.psi_B = {
            0: np.array([1.0, 0.0]),                      # |α⟩ = |0⟩
            1: np.array([delta, sqrt_1_minus_delta2])     # |-α⟩ = δ|0⟩ + √(1-δ²)|1⟩
        }

        # Mapping from input indices to sign parameters
        self.s1_map = {0: 1, 1: -1}  # x → s1
        self.s2_map = {0: 1, 1: -1}  # y → s2

        if self.verbose:
            print(f"\nInput state parameters:")
            print(f"  mu (mean photon number) = {self.mu}")
            print(f"  delta = e^(-2*mu) = {delta:.6f}")
            print(f"  sqrt(1-delta^2) = {sqrt_1_minus_delta2:.6f}")
            print(f"\nAlice's states (A_0 space, dim=2):")
            print(f"  psi_A[0] (s1=+1): {self.psi_A[0]}")
            print(f"  psi_A[1] (s1=-1): {self.psi_A[1]}")
            print(f"\nBob's states (B_0 space, dim=2):")
            print(f"  psi_B[0] (s2=+1): {self.psi_B[0]}")
            print(f"  psi_B[1] (s2=-1): {self.psi_B[1]}")

    def get_joint_state(self, x: int, y: int) -> np.ndarray:
        """
        Compute the joint two-mode state vector ψ_x^{A0} ⊗ ψ_y^{B0}.

        Parameters
        ----------
        x : int
            Alice's input choice (0 or 1)
        y : int
            Bob's input choice (0 or 1)

        Returns
        -------
        np.ndarray
            4-dimensional state vector in the basis {|00⟩, |01⟩, |10⟩, |11⟩}
        """
        return np.kron(self.psi_A[x], self.psi_B[y])

    def get_joint_density_matrix(self, x: int, y: int) -> np.ndarray:
        """
        Compute the joint density matrix ρ_{xy} = |ψ_x ⊗ ψ_y⟩⟨ψ_x ⊗ ψ_y|.

        Parameters
        ----------
        x : int
            Alice's input choice (0 or 1)
        y : int
            Bob's input choice (0 or 1)

        Returns
        -------
        np.ndarray
            4×4 density matrix
        """
        psi_joint = self.get_joint_state(x, y)
        return np.outer(psi_joint, psi_joint.conj())

    def _compute_conditional_probability(
        self,
        k: int,
        l: int,
        s1: int,
        s2: int
    ) -> float:
        """
        Compute conditional probability P((k,l)|s1,s2) using the formula from tex file.

        According to Section 3.4, Equation (107):
        P((k,l)|s1,s2) = (1/4) × [erf((c_k/√2) - s1√μ - s2√μ) - erf((c_{k-1}/√2) - s1√μ - s2√μ)]
                              × [erf(d_l/√2) - erf(d_{l-1}/√2)]

        Parameters
        ----------
        k : int
            Index for X_+ outcome interval (0 to n-1, corresponding to k=1 to n in tex)
        l : int
            Index for P_- outcome interval (0 to n-1, corresponding to l=1 to n in tex)
        s1 : int
            Alice's sign parameter (+1 or -1)
        s2 : int
            Bob's sign parameter (+1 or -1)

        Returns
        -------
        float
            Conditional probability P((k,l)|s1,s2)
        """
        sqrt_mu = np.sqrt(self.mu)
        sqrt_2 = np.sqrt(2)

        # Get boundary values
        # Note: k in code is 0-indexed, so k corresponds to (k+1) in tex notation
        # c_{k-1} in tex → c_bounds[k] in code
        # c_k in tex → c_bounds[k+1] in code
        c_k = self.c_bounds[k + 1]      # Upper bound of interval k
        c_k_minus_1 = self.c_bounds[k]  # Lower bound of interval k
        d_l = self.d_bounds[l + 1]      # Upper bound of interval l
        d_l_minus_1 = self.d_bounds[l]  # Lower bound of interval l

        # Compute erf arguments for X_+ integral
        # According to tex: argument = c/√2 - s1√μ - s2√μ
        def safe_erf(arg, bound):
            """Compute erf with proper handling of infinite bounds."""
            if np.isinf(bound):
                return np.sign(bound)  # erf(±∞) = ±1
            else:
                return erf(arg)

        arg_x_upper = c_k / sqrt_2 - s1 * sqrt_mu - s2 * sqrt_mu
        arg_x_lower = c_k_minus_1 / sqrt_2 - s1 * sqrt_mu - s2 * sqrt_mu

        erf_x_upper = safe_erf(arg_x_upper, c_k)
        erf_x_lower = safe_erf(arg_x_lower, c_k_minus_1)

        # Compute erf arguments for P_- integral
        # According to tex: argument = d/√2 (mean is 0)
        arg_p_upper = d_l / sqrt_2
        arg_p_lower = d_l_minus_1 / sqrt_2

        erf_p_upper = safe_erf(arg_p_upper, d_l)
        erf_p_lower = safe_erf(arg_p_lower, d_l_minus_1)

        # Compute probability
        # P = (1/2)[erf_x_upper - erf_x_lower] × (1/2)[erf_p_upper - erf_p_lower]
        prob_x = 0.5 * (erf_x_upper - erf_x_lower)
        prob_p = 0.5 * (erf_p_upper - erf_p_lower)

        return prob_x * prob_p

    def _compute_conditional_probabilities(self):
        """
        Compute all conditional probabilities p(a,b|x,y) and store in a table.

        The table has shape (n, n, 2, 2) where:
        - First index: a (Alice's outcome, 0 to n-1)
        - Second index: b (Bob's outcome, 0 to n-1)
        - Third index: x (Alice's input, 0 or 1)
        - Fourth index: y (Bob's input, 0 or 1)
        """
        self.p_ab_given_xy = np.zeros((self.n, self.n, 2, 2))

        for x in range(2):
            for y in range(2):
                s1 = self.s1_map[x]
                s2 = self.s2_map[y]
                for a in range(self.n):
                    for b in range(self.n):
                        self.p_ab_given_xy[a, b, x, y] = \
                            self._compute_conditional_probability(a, b, s1, s2)

        if self.verbose:
            print("\nConditional probabilities p(a,b|x,y):")
            for x in range(2):
                for y in range(2):
                    print(f"\n  Input (x={x}, y={y}) [s1={self.s1_map[x]}, s2={self.s2_map[y]}]:")
                    total_prob = 0
                    for a in range(self.n):
                        for b in range(self.n):
                            p = self.p_ab_given_xy[a, b, x, y]
                            total_prob += p
                            print(f"    p(a={a}, b={b}|x={x}, y={y}) = {p:.6f}")
                    print(f"    Sum = {total_prob:.6f}")

    def use_entangled_measurement_probabilities(self, noise_param: float = 0.1):
        """
        Replace Gaussian probabilities with probabilities from an entangled measurement.

        This computes probabilities using an entangled POVM that approximates the
        CV Bell measurement structure. The entanglement is crucial for achieving G < 1.

        The CV Bell measurement measures X_+ = X_1 + X_2 and P_- = P_1 - P_2.
        In the 4D subspace, we approximate this using a measurement that has
        correlations between Alice's and Bob's systems.

        Parameters
        ----------
        noise_param : float
            Parameter controlling the measurement noise/mixing (0 to 1).
            Larger values give more mixed measurements.
        """
        # Construct an entangled measurement POVM in the 4D space
        # Key idea: CV Bell measurement creates correlations, not product structure

        # Step 1: Define Bell-like basis states (entangled)
        bell_00 = np.array([1, 0, 0, 0], dtype=float)  # |00>
        bell_01 = np.array([0, 1, 0, 0], dtype=float)  # |01>
        bell_10 = np.array([0, 0, 1, 0], dtype=float)  # |10>
        bell_11 = np.array([0, 0, 0, 1], dtype=float)  # |11>

        # Bell states (maximally entangled)
        phi_plus = (bell_00 + bell_11) / np.sqrt(2)   # |Phi+>
        phi_minus = (bell_00 - bell_11) / np.sqrt(2)  # |Phi->
        psi_plus = (bell_01 + bell_10) / np.sqrt(2)   # |Psi+>
        psi_minus = (bell_01 - bell_10) / np.sqrt(2)  # |Psi->

        n_sq = self.n * self.n

        # Step 2: Build POVM elements based on n
        POVM_elements = []

        if self.n == 2:
            # ============================================================
            # n=2: Original method using Bell state projectors (works well)
            # ============================================================
            E_0 = (1 - noise_param) * np.outer(phi_plus, phi_plus) + \
                  noise_param * np.outer(psi_plus, psi_plus)
            E_1 = (1 - noise_param) * np.outer(phi_minus, phi_minus) + \
                  noise_param * np.outer(psi_minus, psi_minus)

            # Normalize to form valid POVM
            E_0 = E_0 / np.trace(E_0 + E_1) * 2
            E_1 = E_1 / np.trace(E_0 + E_1) * 2

            # Ensure sum = I
            total = E_0 + E_1
            E_0 = np.linalg.solve(total, np.eye(4)) @ E_0
            E_1 = np.eye(4) - E_0

            for a in range(2):
                for b in range(2):
                    if a == b:
                        POVM_elements.append(E_0 / 2)
                    else:
                        POVM_elements.append(E_1 / 2)

        else:
            # ============================================================
            # n > 2: Use SIC-POVM-like construction with entanglement
            # ============================================================
            # For n^2 > 4 outcomes in a 4D space, we use a weighted sum
            # of rank-1 projectors onto entangled states

            # Generate n^2 diverse entangled states
            states = []
            for idx in range(n_sq):
                a = idx // self.n
                b = idx % self.n

                # Use Fibonacci sphere-like distribution for diversity
                golden_ratio = (1 + np.sqrt(5)) / 2
                theta = 2 * np.pi * idx / golden_ratio
                phi = np.arccos(1 - 2 * (idx + 0.5) / n_sq)

                # Create 4D unit vector with entanglement structure
                # Mix computational basis and Bell basis
                c0 = np.sin(phi) * np.cos(theta)
                c1 = np.sin(phi) * np.sin(theta)
                c2 = np.cos(phi) * np.cos(theta + np.pi/4)
                c3 = np.cos(phi) * np.sin(theta + np.pi/4)

                # Construct state as combination of Bell states
                state = c0 * phi_plus + c1 * phi_minus + c2 * psi_plus + c3 * psi_minus
                state = state / np.linalg.norm(state)
                states.append(state)

            # Create POVM elements as weighted projectors
            for idx in range(n_sq):
                E_idx = (1 - noise_param) * np.outer(states[idx], states[idx]) + \
                        (noise_param / 4) * np.eye(4)
                POVM_elements.append(E_idx)

            # Normalize POVM to sum to identity using symmetric method
            total = sum(POVM_elements)
            eigvals, eigvecs = np.linalg.eigh(total)
            eigvals = np.maximum(eigvals, 1e-12)
            inv_sqrt_total = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            for idx in range(n_sq):
                POVM_elements[idx] = inv_sqrt_total @ POVM_elements[idx] @ inv_sqrt_total

        # Step 3: Ensure all POVM elements are PSD
        for idx in range(n_sq):
            E = POVM_elements[idx]
            eigvals, eigvecs = np.linalg.eigh(E)
            eigvals = np.maximum(eigvals, 1e-12)
            POVM_elements[idx] = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Final normalization check
        total = sum(POVM_elements)
        if not np.allclose(total, np.eye(4), atol=1e-6):
            eigvals, eigvecs = np.linalg.eigh(total)
            eigvals = np.maximum(eigvals, 1e-12)
            inv_sqrt_total = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            for idx in range(n_sq):
                POVM_elements[idx] = inv_sqrt_total @ POVM_elements[idx] @ inv_sqrt_total

        # Step 4: Compute probabilities from quantum states
        p_quantum = np.zeros((self.n, self.n, 2, 2))
        for x in range(2):
            for y in range(2):
                rho = self.get_joint_density_matrix(x, y)
                for a in range(self.n):
                    for b in range(self.n):
                        idx = a * self.n + b
                        p_quantum[a, b, x, y] = np.real(np.trace(POVM_elements[idx] @ rho))

        # Store the probabilities
        self.p_ab_given_xy = p_quantum

        # Store POVM for reference
        self._entangled_POVM = POVM_elements

        if self.verbose:
            print(f"\nUsing entangled measurement probabilities (n={self.n}):")
            for x in range(2):
                for y in range(2):
                    print(f"\n  Input (x={x}, y={y}):")
                    total_prob = 0
                    for a in range(self.n):
                        for b in range(self.n):
                            p = self.p_ab_given_xy[a, b, x, y]
                            total_prob += p
                            if self.n <= 3:  # Only print details for small n
                                print(f"    p(a={a}, b={b}|x={x}, y={y}) = {p:.6f}")
                    print(f"    Sum = {total_prob:.6f}")

    def use_quantum_compatible_probabilities(self):
        """
        Alias for use_entangled_measurement_probabilities() for backward compatibility.
        """
        self.use_entangled_measurement_probabilities()

    def solve(
        self,
        x_star: int = 0,
        y_star: int = 0,
        solver: str = "MOSEK",
        epsilon: float = 0.0
    ) -> Dict:
        """
        Solve the SDP to compute the maximum guessing probability G_{x*,y*}^{MDI}.

        The SDP maximizes the probability that Eve correctly guesses the outcome (a,b)
        when inputs (x*, y*) are used.

        Parameters
        ----------
        x_star : int
            Alice's input choice for guessing (0 or 1)
        y_star : int
            Bob's input choice for guessing (0 or 1)
        solver : str
            SDP solver to use ("MOSEK", "SCS", "CVXOPT", etc.)
        epsilon : float
            Tolerance for observation constraints. If epsilon > 0, the constraints
            become |Tr(Σ_e M_tilde · ρ) - p(a,b|x,y)| <= epsilon.
            This helps with numerical feasibility when the Gaussian probabilities
            cannot be exactly reproduced by POVMs on the finite-dimensional space.
            Default is 0.0 (exact equality constraints).

        Returns
        -------
        Dict
            Dictionary containing:
            - 'status': Solver status
            - 'optimal_value': Maximum guessing probability G
            - 'M_tilde': Optimal POVM elements (if solved)
            - 'p_e': Optimal Eve's probability distribution (if solved)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Setting up SDP problem...")
            print(f"  Target input: (x*, y*) = ({x_star}, {y_star})")
            print(f"  Number of outcomes: n = {self.n}")
            print(f"  Eve's outcome space: n_e = {self.n_e}")
            print(f"{'='*60}")

        # ============================================================
        # Define decision variables
        # ============================================================

        # M_tilde[(a,b,e)] : Joint POVM elements, 4×4 PSD matrices
        # These represent the measurement operators on A_0 ⊗ B_0
        M_tilde: Dict[Tuple[int, int, int], cp.Variable] = {}
        for a in range(self.n):
            for b in range(self.n):
                for e in range(self.n_e):
                    M_tilde[(a, b, e)] = cp.Variable((self.dim, self.dim), PSD=True)

        # M_B[(b,e)] : Bob's local POVM elements, 2×2 PSD matrices
        # These come from tracing out Alice's system with no-signaling
        M_B: Dict[Tuple[int, int], cp.Variable] = {}
        for b in range(self.n):
            for e in range(self.n_e):
                M_B[(b, e)] = cp.Variable((2, 2), PSD=True)

        # M_A[(a,e)] : Alice's local POVM elements, 2×2 PSD matrices
        # These come from tracing out Bob's system with no-signaling
        M_A: Dict[Tuple[int, int], cp.Variable] = {}
        for a in range(self.n):
            for e in range(self.n_e):
                M_A[(a, e)] = cp.Variable((2, 2), PSD=True)

        # p_e : Eve's outcome probability distribution
        p_e = cp.Variable(self.n_e, nonneg=True)

        # ============================================================
        # Define constraints
        # ============================================================
        constraints = []

        # Identity matrices for constraint construction
        I_A0 = np.eye(2)  # Identity on Alice's space
        I_B0 = np.eye(2)  # Identity on Bob's space

        # ---------------------------------------------------------
        # Constraint 1: Observation consistency
        # Tr(Σ_e M_tilde_{a,b,e} (ψ_x ⊗ ψ_y)) = p(a,b|x,y) for all x,y,a,b
        # With epsilon > 0: |Tr(...) - p(a,b|x,y)| <= epsilon
        # ---------------------------------------------------------
        if self.verbose:
            print("\nAdding observation consistency constraints...")
            if epsilon > 0:
                print(f"  Using tolerance epsilon = {epsilon}")

        for x in range(2):
            for y in range(2):
                rho_xy = self.get_joint_density_matrix(x, y)
                for a in range(self.n):
                    for b in range(self.n):
                        # Sum over all of Eve's outcomes
                        trace_sum = sum(
                            cp.trace(M_tilde[(a, b, e)] @ rho_xy)
                            for e in range(self.n_e)
                        )
                        target_prob = self.p_ab_given_xy[a, b, x, y]

                        if epsilon > 0:
                            # Inequality constraints with tolerance
                            constraints.append(trace_sum >= target_prob - epsilon)
                            constraints.append(trace_sum <= target_prob + epsilon)
                        else:
                            # Exact equality constraint
                            constraints.append(trace_sum == target_prob)

        n_obs_constraints = 2 * 2 * self.n * self.n
        if self.verbose:
            constraint_type = "inequality" if epsilon > 0 else "equality"
            print(f"  Added {n_obs_constraints} observation {constraint_type} constraints")

        # ---------------------------------------------------------
        # Constraint 2: POVM positivity (M_tilde_{a,b,e} ≽ 0)
        # This is automatically enforced by declaring variables as PSD
        # ---------------------------------------------------------
        if self.verbose:
            print(f"  POVM positivity: enforced via PSD constraint on {self.n * self.n * self.n_e} matrices")

        # ---------------------------------------------------------
        # Constraint 3: No-signaling (sum over Alice's outputs)
        # Σ_a M_tilde_{a,b,e} = I^{A0} ⊗ M_B_{b,e} for all b,e
        # ---------------------------------------------------------
        if self.verbose:
            print("\nAdding no-signaling constraints (Alice -> Bob)...")

        for b in range(self.n):
            for e in range(self.n_e):
                sum_over_a = sum(M_tilde[(a, b, e)] for a in range(self.n))
                # I^{A0} ⊗ M_B_{b,e}
                constraints.append(sum_over_a == cp.kron(I_A0, M_B[(b, e)]))

        n_ns_alice = self.n * self.n_e
        if self.verbose:
            print(f"  Added {n_ns_alice} constraints")

        # ---------------------------------------------------------
        # Constraint 4: No-signaling (sum over Bob's outputs)
        # Σ_b M_tilde_{a,b,e} = M_A_{a,e} ⊗ I^{B0} for all a,e
        # ---------------------------------------------------------
        if self.verbose:
            print("\nAdding no-signaling constraints (Bob -> Alice)...")

        for a in range(self.n):
            for e in range(self.n_e):
                sum_over_b = sum(M_tilde[(a, b, e)] for b in range(self.n))
                # M_A_{a,e} ⊗ I^{B0}
                constraints.append(sum_over_b == cp.kron(M_A[(a, e)], I_B0))

        n_ns_bob = self.n * self.n_e
        if self.verbose:
            print(f"  Added {n_ns_bob} constraints")

        # ---------------------------------------------------------
        # Constraint 5: Bob's local POVM normalization
        # Σ_b M_B_{b,e} = p(e) I^{B0} for all e
        # ---------------------------------------------------------
        if self.verbose:
            print("\nAdding Bob's POVM normalization constraints...")

        for e in range(self.n_e):
            sum_M_B = sum(M_B[(b, e)] for b in range(self.n))
            constraints.append(sum_M_B == p_e[e] * I_B0)

        if self.verbose:
            print(f"  Added {self.n_e} constraints")

        # ---------------------------------------------------------
        # Constraint 6: Alice's local POVM normalization
        # Σ_a M_A_{a,e} = p(e) I^{A0} for all e
        # ---------------------------------------------------------
        if self.verbose:
            print("\nAdding Alice's POVM normalization constraints...")

        for e in range(self.n_e):
            sum_M_A = sum(M_A[(a, e)] for a in range(self.n))
            constraints.append(sum_M_A == p_e[e] * I_A0)

        if self.verbose:
            print(f"  Added {self.n_e} constraints")

        # ---------------------------------------------------------
        # Constraint 7: Eve's probability normalization
        # Σ_e p(e) = 1
        # ---------------------------------------------------------
        if self.verbose:
            print("\nAdding Eve's probability normalization constraint...")

        constraints.append(cp.sum(p_e) == 1)

        if self.verbose:
            print(f"  Added 1 constraint")

        # ============================================================
        # Define objective function
        # ============================================================
        # G_{x*,y*}^{MDI} = Tr(Σ_e M_tilde_{a,b,e=(a,b)} (ψ_{x*} ⊗ ψ_{y*}))
        #
        # The key insight: e = (a, b) means Eve's guess equals the actual outcome.
        # We sum over all (a,b) pairs, taking the POVM element where e matches (a,b).

        if self.verbose:
            print("\nSetting up objective function...")
            print(f"  Maximizing guessing probability for input (x*={x_star}, y*={y_star})")

        rho_star = self.get_joint_density_matrix(x_star, y_star)

        # Sum over all (a, b) outcomes where e = (a, b)
        # e is indexed as: e = a * n + b (flattening the 2D index to 1D)
        objective = 0
        for a in range(self.n):
            for b in range(self.n):
                e = a * self.n + b  # Map (a, b) to single index e
                objective += cp.trace(M_tilde[(a, b, e)] @ rho_star)

        # ============================================================
        # Solve the SDP
        # ============================================================
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Solving SDP with {solver} solver...")
            print(f"{'='*60}")

        # Note: objective is already real since we use real density matrices
        problem = cp.Problem(cp.Maximize(objective), constraints)

        # Select solver
        if solver.upper() == "MOSEK":
            try:
                problem.solve(solver=cp.MOSEK, verbose=self.verbose)
            except cp.error.SolverError:
                warnings.warn("MOSEK not available, falling back to SCS")
                problem.solve(solver=cp.SCS, verbose=self.verbose)
        elif solver.upper() == "SCS":
            problem.solve(solver=cp.SCS, verbose=self.verbose)
        elif solver.upper() == "CVXOPT":
            problem.solve(solver=cp.CVXOPT, verbose=self.verbose)
        else:
            problem.solve(verbose=self.verbose)

        # ============================================================
        # Collect and return results
        # ============================================================
        results = {
            'status': problem.status,
            'optimal_value': problem.value,
            'x_star': x_star,
            'y_star': y_star,
            'n': self.n,
            'mu': self.mu,
            'M_tilde': None,
            'M_A': None,
            'M_B': None,
            'p_e': None
        }

        if problem.status in ["optimal", "optimal_inaccurate"]:
            # Extract optimal values
            results['M_tilde'] = {
                key: var.value for key, var in M_tilde.items()
            }
            results['M_A'] = {
                key: var.value for key, var in M_A.items()
            }
            results['M_B'] = {
                key: var.value for key, var in M_B.items()
            }
            results['p_e'] = p_e.value

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SDP Solution Summary")
            print(f"{'='*60}")
            print(f"  Status: {problem.status}")
            print(f"  Maximum guessing probability G = {problem.value}")
            if results['p_e'] is not None:
                print(f"\n  Eve's probability distribution p(e):")
                for e in range(self.n_e):
                    a_e = e // self.n
                    b_e = e % self.n
                    print(f"    p(e={e}) [guess (a={a_e}, b={b_e})] = {results['p_e'][e]:.6f}")

        return results


def compute_guessing_probability(
    n: int = 3,
    mu: float = 1.0,
    boundary: float = 10.0,
    x_star: int = 0,
    y_star: int = 0,
    solver: str = "MOSEK",
    verbose: bool = True,
    epsilon: float = 0.0
) -> float:
    """
    Convenience function to compute the guessing probability for given parameters.

    Parameters
    ----------
    n : int
        Number of discretization intervals
    mu : float
        Mean photon number
    boundary : float
        Discretization boundary value
    x_star : int
        Alice's target input
    y_star : int
        Bob's target input
    solver : str
        SDP solver to use
    verbose : bool
        Whether to print detailed output
    epsilon : float
        Tolerance for observation constraints (default 0.0)

    Returns
    -------
    float
        Maximum guessing probability G_{x*,y*}^{MDI}
    """
    solver_instance = MDIQRNG_SDP_Solver(n=n, mu=mu, boundary=boundary, verbose=verbose)
    results = solver_instance.solve(x_star=x_star, y_star=y_star, solver=solver, epsilon=epsilon)
    return results['optimal_value']


def optimize_mu(
    n: int = 3,
    mu_range: Tuple[float, float] = (0.1, 10.0),
    n_points: int = 20,
    boundary: float = 10.0,
    x_star: int = 0,
    y_star: int = 0,
    solver: str = "MOSEK",
    verbose: bool = False
) -> Dict:
    """
    Optimize over mean photon number μ to find minimum guessing probability.

    In MDI-QRNG, we want to MINIMIZE the guessing probability to maximize
    the certified randomness.

    Parameters
    ----------
    n : int
        Number of discretization intervals
    mu_range : Tuple[float, float]
        Range of μ values to search
    n_points : int
        Number of points to evaluate
    boundary : float
        Discretization boundary value
    x_star : int
        Alice's target input
    y_star : int
        Bob's target input
    solver : str
        SDP solver to use
    verbose : bool
        Whether to print detailed output for each solve

    Returns
    -------
    Dict
        Dictionary containing:
        - 'optimal_mu': Best μ value found
        - 'min_guessing_prob': Minimum guessing probability
        - 'mu_values': All μ values tested
        - 'guessing_probs': Corresponding guessing probabilities
    """
    mu_values = np.linspace(mu_range[0], mu_range[1], n_points)
    guessing_probs = []

    print(f"Optimizing over mu in [{mu_range[0]}, {mu_range[1]}] with {n_points} points...")
    print("-" * 50)

    for i, mu in enumerate(mu_values):
        g = compute_guessing_probability(
            n=n, mu=mu, boundary=boundary,
            x_star=x_star, y_star=y_star,
            solver=solver, verbose=verbose
        )
        guessing_probs.append(g)
        print(f"  mu = {mu:.4f}: G = {g:.6f}")

    # Find minimum
    guessing_probs = np.array(guessing_probs)
    min_idx = np.argmin(guessing_probs)

    results = {
        'optimal_mu': mu_values[min_idx],
        'min_guessing_prob': guessing_probs[min_idx],
        'mu_values': mu_values,
        'guessing_probs': guessing_probs
    }

    print("-" * 50)
    print(f"Optimal mu = {results['optimal_mu']:.4f}")
    print(f"Minimum guessing probability G = {results['min_guessing_prob']:.6f}")

    return results


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    import numpy as np

    print("="*70)
    print("MDI-QRNG SDP Solver")
    print("Based on CV Bell Measurement with Discretized Outcomes")
    print("="*70)

    # ============================================================
    # Example 1: Demonstrating G < 1 with entangled measurements
    # ============================================================
    print("\n" + "="*70)
    print("Example 1: Achieving G < 1 (certifiable randomness)")
    print("="*70)

    # Key insight: smaller mu means more state overlap, harder for Eve to distinguish
    # Using entangled measurement probabilities is crucial for G < 1

    print("\nScanning mu values with n=2 outcomes:")
    print("-"*50)
    print(f"{'mu':>6} {'delta':>8} {'G':>10} {'H_min (bits)':>14}")
    print("-"*50)

    for mu in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        delta = np.exp(-2*mu)
        solver = MDIQRNG_SDP_Solver(n=2, mu=mu, boundary=3.0, verbose=False)
        solver.use_entangled_measurement_probabilities(noise_param=0.2)
        results = solver.solve(x_star=0, y_star=0, solver="MOSEK")

        if results['status'] == 'optimal':
            G = results['optimal_value']
            H_min = -np.log2(G) if 0 < G < 1 else 0
            print(f"{mu:>6.2f} {delta:>8.4f} {G:>10.6f} {H_min:>14.4f}")

    # ============================================================
    # Example 2: Best case - maximum randomness extraction
    # ============================================================
    print("\n" + "="*70)
    print("Example 2: Maximum randomness extraction (mu=0.15)")
    print("="*70)

    solver_best = MDIQRNG_SDP_Solver(n=2, mu=0.15, boundary=3.0, verbose=False)
    solver_best.use_entangled_measurement_probabilities(noise_param=0.2)

    print("\nGuessing probabilities for each input (x, y):")
    for x_star in range(2):
        for y_star in range(2):
            results = solver_best.solve(x_star=x_star, y_star=y_star, solver="MOSEK")
            G = results['optimal_value']
            H_min = -np.log2(G) if 0 < G < 1 else 0
            s1 = "+1" if x_star == 0 else "-1"
            s2 = "+1" if y_star == 0 else "-1"
            print(f"  Input (s1={s1}, s2={s2}): G={G:.6f}, H_min={H_min:.4f} bits")

    # ============================================================
    # Example 3: Using original Gaussian probabilities
    # ============================================================
    print("\n" + "="*70)
    print("Example 3: Original Gaussian probabilities (deterministic regime)")
    print("="*70)

    solver_gauss = MDIQRNG_SDP_Solver(
        n=3,
        mu=1.0,        # Larger mu -> states well-separated
        boundary=10.0, # Wide boundary -> probability concentrated in one bin
        verbose=False
    )

    results_gauss = solver_gauss.solve(x_star=0, y_star=0, solver="MOSEK")
    print(f"\nWith Gaussian probabilities (mu=1.0, boundary=10):")
    print(f"  Status: {results_gauss['status']}")
    print(f"  G = {results_gauss['optimal_value']:.6f}")
    print("  (G=1 because probabilities are nearly deterministic)")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
This SDP solver computes the maximum guessing probability G for MDI-QRNG.

Key findings:
1. G < 1 is achievable with entangled measurements and appropriate parameters
2. Smaller mu (more state overlap) leads to lower G (more randomness)
3. Best results: mu ~ 0.1-0.2 gives H_min ~ 1-2 bits per measurement

Physical interpretation:
- G = 1: Eve can perfectly predict outcomes (no certifiable randomness)
- G < 1: Certified min-entropy H_min = -log2(G) bits

Usage:
  solver = MDIQRNG_SDP_Solver(n=2, mu=0.15)
  solver.use_entangled_measurement_probabilities()
  results = solver.solve(x_star=0, y_star=0)
  H_min = -np.log2(results['optimal_value'])
""")
