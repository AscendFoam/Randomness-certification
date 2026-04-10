<div align="center">

# Derivation of Discrete Probabilities in Continuous-Variable Bell Measurements

</div>

## 1 Introduction

We consider a standard continuous-variable (CV) Bell measurement where two independent coherent states, $ |\alpha\rangle_{a} $ and $ |\beta\rangle_{b} $ , are injected into the input ports (modes a and b) of a 50:50 lossless beam splitter (BS). At the output ports (modes c and d), ideal homodyne detection is performed to measure the amplitude quadrature $ \hat{Q}_{c} $ and the phase quadrature $ \hat{P}_{d}. $

We define the quadratures such that $ [\hat{Q},\hat{P}]=i $ . The continuous measurement outcomes (q,p) are discretized into a 2D grid. The target domain $ D_{i,j} $ corresponding to the discrete labels (i,j) is given by the intervals $ [q_{i},q_{i}+\Delta q]\times[p_{j},p_{j}+\Delta p] $ . Our objective is to calculate the discrete joint probability $ P_{i,j} $ using two distinct methods and subsequently prove their equivalence.

## 2 Method 1: POVM Formalism in the Truncated Fock Basis

In this approach, we backtrack the measurement operators from the output modes to the input modes to construct the Positive Operator-Valued Measure (POVM) elements.

## 2.1 Continuous and Discretized POVMs

The ideal homodyne measurements at the output ports correspond to projectors onto the quadrature eigenstates. The continuous POVM element in the output space is:

$$
\hat {\Pi} _ {o u t} (q, p) = | q \rangle_ {c} \langle q | \otimes | p \rangle_ {d} \langle p |
$$

Applying the unitary transformation of the beam splitter, $ \hat{U}_{BS} $ , the equivalent continuous POVM acting on the input space is:

$$
\hat {\Pi} _ {i n} (q, p) = \hat {U} _ {B S} ^ {\dagger} \left(| q \rangle_ {c} \langle q | \otimes | p \rangle_ {d} \langle p |\right) \hat {U} _ {B S}
$$

To obtain the discrete POVM element $ \hat{\Pi}_{i,j} $ corresponding to the grid $ D_{i,j} $ , we integrate the continuous operator over the specified intervals:

$$
\hat {\Pi} _ {i, j} = \int_ {q _ {i}} ^ {q _ {i} + \Delta q} d q \int_ {p _ {j}} ^ {p _ {j} + \Delta p} d p \hat {\Pi} _ {i n} (q, p)
$$

## 2.2 Fock Basis Expansion and Truncation

For numerical computation, the infinite-dimensional Hilbert space is truncated at a maximum photon number N. We evaluate the matrix elements of $ \hat{\Pi}_{i,j} $ in the truncated Fock basis $ \left\{ \left| n_{a}, m_{b}\right\rangle \right\} $

By inserting the resolution of the identity for the output modes, the matrix element becomes:

$$
\begin{array}{l} \langle n _ {a}, m _ {b} | \hat {\Pi} _ {i, j} | n _ {a} ^ {\prime}, m _ {b} ^ {\prime} \rangle = \int_ {D _ {i, j}} d q d p \sum_ {k, l, k ^ {\prime}, l ^ {\prime} = 0} ^ {N} \langle n _ {a}, m _ {b} | \hat {U} _ {B S} ^ {\dagger} | k _ {c}, l _ {d} \rangle \\ \times \langle k _ {c} | q \rangle_ {c c} \langle q | k _ {c} ^ {\prime} \rangle \langle l _ {d} | p \rangle_ {d d} \langle p | l _ {d} ^ {\prime} \rangle \\ \times \langle k _ {c} ^ {\prime}, l _ {d} ^ {\prime} | \hat {U} _ {B S} | n _ {a} ^ {\prime}, m _ {b} ^ {\prime} \rangle \\ \end{array}
$$

To evaluate this integral, we require the explicit mathematical forms of both the quadrature wavefunctions $ \langle k_{c}|q\rangle $ $ \langle l_{d}|p\rangle $ , and the beam splitter transition amplitudes $ \langle k_{c}^{\prime},l_{d}^{\prime}|\hat{U}_{BS}|n_{a}^{\prime},m_{b}^{\prime}\rangle. $

## 2.3 Quadrature Wavefunctions and Hermite Polynomials

The inner products $ \langle q|k\rangle $ and $ \langle p|l\rangle $ represent the real-space (amplitude) and momentum-space (phase) wavefunctions of the quantum harmonic oscillator, respectively.

For the amplitude quadrature $ \hat{Q} $ , the wavefunction of the k-th Fock state is given by:

$$
\psi_ {k} (q) = \langle q | k \rangle = \frac {1}{\pi^ {1 / 4} \sqrt {2 ^ {k} k !}} H _ {k} (q) e ^ {- q ^ {2} / 2}
$$

For the phase quadrature $ \hat{P} $ , due to the Fourier transform relationship between position and momentum spaces, a phase factor $ (-i)^{l} $ is introduced. The wavefunction for the l-th Fock state is:

$$
\tilde {\psi} _ {l} (p) = \langle p | l \rangle = \frac {(- i) ^ {l}}{\pi^ {1 / 4} \sqrt {2 ^ {l} l !}} H _ {l} (p) e ^ {- p ^ {2} / 2}
$$

Here, $ H_{n}(x) $ denotes the physicists' Hermite polynomials. They are defined canonically via Rodrigues' formula:

$$
H _ {n} (x) = (- 1) ^ {n} e ^ {x ^ {2}} \frac {d ^ {n}}{d x ^ {n}} \left(e ^ {- x ^ {2}}\right)
$$

For the purpose of algorithmic implementation and numerical integration, the explicit series expansion of the Hermite polynomial is utilized:

$$
H _ {n} (x) = n! \sum_ {m = 0} ^ {\lfloor n / 2 \rfloor} \frac {(- 1) ^ {m} (2 x) ^ {n - 2 m}}{m ! (n - 2 m) !}
$$

where $ \lfloor n/2\rfloor $ is the floor function, returning the greatest integer less than or equal to $ n/2 $ Substituting Eqs.(5) and (6) into Eq.(4) allows the integrand to be expressed entirely in terms of known polynomials and Gaussian functions, which can be integrated over the grid $ D_{i,j} $

## 2.4 Explicit Matrix Elements of the Beam Splitter

To fully determine Eq.(4), we also need the unitary operator $ \hat{U}_{BS} $ in the Fock basis. A 50:50 lossless beam splitter is modeled by:

$$
\hat {U} _ {B S} = \exp \left[ \frac {\pi}{4} \left(\hat {a} ^ {\dagger} \hat {b} - \hat {a} \hat {b} ^ {\dagger}\right) \right]
$$

In the Heisenberg picture, this operator transforms the creation operators as:

$$
\hat {U} _ {B S} \hat {a} ^ {\dagger} \hat {U} _ {B S} ^ {\dagger} = \frac {\hat {a} ^ {\dagger} + \hat {b} ^ {\dagger}}{\sqrt {2}} \equiv \hat {c} ^ {\dagger}
$$

$$
\hat {U} _ {B S} \hat {b} ^ {\dagger} \hat {U} _ {B S} ^ {\dagger} = \frac {\hat {a} ^ {\dagger} - \hat {b} ^ {\dagger}}{\sqrt {2}} \equiv \hat {d} ^ {\dagger}
$$

We apply $ \hat{U}_{BS} $ to the input Fock state $ |k_{a},l_{b}\rangle=\frac{(\hat{a}^{\dagger})^{k}(\hat{b}^{\dagger})^{l}}{\sqrt{k!l!}}|0,0\rangle $:

$$
\begin{array}{l} \hat {U} _ {B S} \left| k _ {a}, l _ {b} \right\rangle = \frac {1}{\sqrt {k ! l !}} \left(\hat {U} _ {B S} \hat {a} ^ {\dagger} \hat {U} _ {B S} ^ {\dagger}\right) ^ {k} \left(\hat {U} _ {B S} \hat {b} ^ {\dagger} \hat {U} _ {B S} ^ {\dagger}\right) ^ {l} \hat {U} _ {B S} | 0, 0 \rangle \\ = \frac {1}{\sqrt {k ! l ! 2 ^ {k + l}}} \left(\hat {c} ^ {\dagger} + \hat {d} ^ {\dagger}\right) ^ {k} \left(\hat {c} ^ {\dagger} - \hat {d} ^ {\dagger}\right) ^ {l} | 0, 0 \rangle \\ \end{array}
$$

Expanding the binomials:

$$
\hat {U} _ {B S} \left| k _ {a}, l _ {b} \right\rangle = \frac {1}{\sqrt {k ! l ! 2 ^ {k + l}}} \sum_ {p = 0} ^ {k} \sum_ {q = 0} ^ {l} \binom {k} {p} \binom {l} {q} (- 1) ^ {l - q} \left(\hat {c} ^ {\dagger}\right) ^ {p + q} \left(\hat {d} ^ {\dagger}\right) ^ {k + l - p - q} | 0, 0 \rangle
$$

Defining the output photon numbers $ n=p+q $ and $ m=k+l-p-q $ (where $ n+m=k+l $ due to energy conservation), and applying the creation operators to the vacuum $ (\hat{c}^{\dagger})^{n}(\hat{d}^{\dagger})^{m}|0,0\rangle = \sqrt{n!m!}|n_{c},m_{d}\rangle $ , we project onto the output basis. Substituting $ q=n-p $ yields the explicit transition amplitude:

$$
\langle n _ {c}, m _ {d} | \hat {U} _ {B S} | k _ {a}, l _ {b} \rangle = \delta_ {n + m, k + l} \left[ \frac {n ! m ! k ! l !}{2 ^ {k + l}} \right] ^ {1 / 2} \sum_ {p = \max (0, n - l)} ^ {\min (k, n)} \frac {(- 1) ^ {l - n + p}}{p ! (k - p) ! (n - p) ! (l - n + p) !}
$$

## 2.5 Probability Calculation

The discrete probability is finally obtained by taking the trace of the POVM element with the input density matrix $ \rho_{in}=|\alpha\rangle\langle\alpha|_{a}\otimes|\beta\rangle\langle\beta|_{b} $:

$$
P _ {i, j} ^ {(1)} = \operatorname {T r} \left(\rho_ {i n} \hat {\Pi} _ {i, j}\right) = \sum_ {n, m, n ^ {\prime}, m ^ {\prime} = 0} ^ {N} \langle n _ {a} ^ {\prime}, m _ {b} ^ {\prime} | \rho_ {i n} | n _ {a}, m _ {b} \rangle \langle n _ {a}, m _ {b} | \hat {\Pi} _ {i, j} | n _ {a} ^ {\prime}, m _ {b} ^ {\prime} \rangle
$$

Due to the truncation at N, this method yields an approximation which converges to the exact value as $ N\to\infty. $

## 3 Method 2: Direct Phase-Space Integration

This method utilizes the structural properties of coherent states under linear optical transformations to directly evaluate the joint probability distribution.

## 3.1 Evolution of the Coherent State

A fundamental property of a 50:50 beam splitter is that the output of two coherent states remains a separable product of coherent states. The transformation is given by $ \hat{c}=(\hat{a}+\hat{b}) / \sqrt{2} $ and $ \hat{d}=(\hat{a}-\hat{b}) / \sqrt{2} $ . The output state is:

$$
| \psi_ {o u t} \rangle = \hat {U} _ {B S} | \alpha \rangle_ {a} | \beta \rangle_ {b} = \left| \frac {\alpha + \beta}{\sqrt {2}} \right\rangle_ {c} \otimes \left| \frac {\alpha - \beta}{\sqrt {2}} \right\rangle_ {d} \equiv | \gamma \rangle_ {c} \otimes | \delta \rangle_ {d}
$$

Since the output state is a tensor product, the measurements of $ \hat{Q}_{c} $ and $ \hat{P}_{d} $ are statistically independent.

## 3.2 Joint Probability Density Function (PDF)

The probability density function for measuring q on mode c and p on mode d is:

$$
f (q, p) = \left| \langle q | \gamma \rangle_ {c} \right| ^ {2} \times \left| \langle p | \delta \rangle_ {d} \right| ^ {2}
$$

For a coherent state $ |\gamma\rangle $ , the measurement of the quadrature $ \hat{Q} $ yields a Gaussian distribution with variance 1/2. Thus:

$$
\left| \langle q | \gamma \rangle_ {c} \right| ^ {2} = \frac {1}{\sqrt {\pi}} \exp \left[ - \left(q - \sqrt {2} \operatorname {R e} (\gamma)\right) ^ {2} \right]
$$

$$
| \langle p | \delta \rangle_ {d} | ^ {2} = \frac {1}{\sqrt {\pi}} \exp \left[ - \left(p - \sqrt {2} \operatorname {I m} (\delta)\right) ^ {2} \right]
$$

Substituting $ \gamma $ and $ \delta $ , the continuous joint PDF is strictly analytically defined without any approximations.

## 3.3 Probability Calculation

The discrete probability $ P_{i,j}^{(2)} $ is calculated by integrating the PDF over the target grid:

$$
P _ {i, j} ^ {(2)} = \int_ {q _ {i}} ^ {q _ {i} + \Delta q} d q \int_ {p _ {j}} ^ {p _ {j} + \Delta p} d p f (q, p)
$$

This integral resolves into the product of two standard error functions (erf).

## 4 Proof of Equivalence

To prove that Method 1 and Method 2 are mathematically identical in the infinite dimensional limit $ ( N \to\infty) $ , we start with the trace expression from Method 1:

$$
P _ {i, j} ^ {(1)} = \operatorname {T r} \left(\rho_ {i n} \hat {\Pi} _ {i, j}\right)
$$

Expanding $ \hat{\Pi}_{i,j} $ using its integral definition:

$$
P _ {i, j} ^ {(1)} = \operatorname {T r} \left[ \rho_ {i n} \int_ {D _ {i, j}} d q d p \hat {U} _ {B S} ^ {\dagger} \left(| q \rangle_ {c} \langle q | \otimes | p \rangle_ {d} \langle p |\right) \hat {U} _ {B S} \right]
$$

Since the integral is linear, it commutes with the trace operation. Furthermore, utilizing the cyclic property of the trace $ \operatorname{Tr} (\hat{A}\hat{B}\hat{C})=\operatorname{Tr}(\hat{C}\hat{A}\hat{B}) $ , we can rearrange the operators:

$$
P _ {i, j} ^ {(1)} = \int_ {D _ {i, j}} d q d p \operatorname {T r} \left[ \left(\hat {U} _ {B S} \rho_ {i n} \hat {U} _ {B S} ^ {\dagger}\right) \left(| q \rangle_ {c} \langle q | \otimes | p \rangle_ {d} \langle p |\right) \right]
$$

The term $ \hat{U}_{BS}\rho_{in}\hat{U}_{BS}^{\dagger} $ is precisely the density matrix of the output state, $ \rho_{out} $ . As derived in Method 2, for an input state $ \rho_{in}=|\alpha\rangle\langle\alpha|_{a}\otimes|\beta\rangle\langle\beta|_{b} $ , the output density matrix is:

$$
\rho_ {o u t} = | \gamma \rangle \langle \gamma | _ {c} \otimes | \delta \rangle \langle \delta | _ {d}
$$

Substituting $ \rho_{out} $ into the trace expression:

$$
P _ {i, j} ^ {(1)} = \int_ {D _ {i, j}} d q d p \operatorname {T r} \left[ \left(| \gamma \rangle \langle \gamma | _ {c} \otimes | \delta \rangle \langle \delta | _ {d}\right) \left(| q \rangle_ {c} \langle q | \otimes | p \rangle_ {d} \langle p |\right) \right]
$$

Evaluating the trace over the continuous spatial basis yields the inner products:

$$
\begin{array}{l} P _ {i, j} ^ {(1)} = \int_ {D _ {i, j}} d q d p \langle q | \gamma \rangle_ {c c} \langle \gamma | q \rangle_ {c} \cdot \langle p | \delta \rangle_ {d d} \langle \delta | p \rangle_ {d} \\ = \int_ {D _ {i, j}} d q d p | \langle q | \gamma \rangle_ {c} | ^ {2} | \langle p | \delta \rangle_ {d} | ^ {2} \\ = \int_ {D _ {i, j}} d q d p f (q, p) \\ \end{array}
$$

This final expression is strictly identical to $ P_{i,j}^{(2)}。 $

Conclusion: We conclude that $ P_{i,j}^{(1)}\equiv P_{i,j}^{(2)} $ when $ N\rightarrow\infty $ . The disparity between the two methods in practical scenarios strictly stems from the non-unitary nature of the operator $ \sum_{n=0}^{N}|n\rangle\langle n|<\mathbf{I} $ applied during the Fock basis truncation in Method 1. For states exhibiting purely Gaussian statistics (e.g., coherent states), Method 2 represents the exact analytical solution and circumvents truncation errors entirely.