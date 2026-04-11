<div align="center">

# QRNG with uncharacterized APD

</div>

January 5,2026

## 1 Introduction

In this note, we consider a quantum random number generator (QRNG) with uncharacterized APD. This means that, the POVM elements of APD are diagonal, but the diagonal terms are unknown. We give the randomness quantification by numerical method.

## 2 Security Analysis

The numerical method for general measurement-device-independent (MDI) QRNG is formulated as an optimization problem searching for the maximum possible guessing probability over all possible measurement performed by the adversary. We assume D test states, N measurement outcomes and M dimensional quantum states and POVM. Then the unknown POVM is decomposed into $ N^{D} $ groups,

$$
M _ {y} = \sum_ {\lambda_ {0} = 1} ^ {N} \sum_ {\lambda_ {1} = 1} ^ {N} \dots \sum_ {\lambda_ {D} = 1} ^ {N} M _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}}.
$$

such that $ \max_{y}[\operatorname{tr}(\rho_{x}M_{y}^{\lambda_{0}\lambda_{1}\dots\lambda_{D}})]=\operatorname{tr}(\rho_{x}M_{\lambda_{x}}^{\lambda_{0}\lambda_{1}\dots\lambda_{D}}),(x\in\{1,2,\dots,D\}) $ . Then we have the following theorem.

Theorem 1. Given the input state $ \rho=\sum_{x}q_{x}\rho_{x} $ $ (x\in\{1,2,\ldots,D\}) $ , the probability that Eve's can successfully guess the output of $ \{M_{y}\}_{y} $ $ (y\in\{0,1,\ldots,N\}) $ i.e., the guessing probability $ p_{\mathrm{guess}}=2^{-H_{\mathrm{min}}} $ , has an upper bound

$$
p _ {\mathrm {g u e s s}} \leq \max _ {M _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}}} \sum_ {x = 1} ^ {D} q _ {x} \sum_ {\lambda_ {0} = 1} ^ {N} \sum_ {\lambda_ {1} = 1} ^ {N} \dots \sum_ {\lambda_ {D} = 1} ^ {N} \operatorname {t r} \left(\rho_ {x} M _ {\lambda_ {x}} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}}\right)
$$

Then the upper bound of guessing probability is computed by a semi-definite

program (SDP)

$$
\begin{array}{l} \max _ {M _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}}} \quad \sum_ {x = 1} ^ {D} q _ {x} \sum_ {\lambda_ {0} = 1} ^ {N} \sum_ {\lambda_ {1} = 1} ^ {N} \dots \sum_ {\lambda_ {D} = 1} ^ {N} \operatorname {t r} \left(\rho_ {x} M _ {\lambda_ {x}} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}}\right) \\ \mathrm {s . t .} \quad M _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}} = \left(M _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}}\right) ^ {\dagger} \\ M _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}} \geq 0 \\ \sum_ {y} M _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}} = \frac {1}{M} \operatorname {t r} \left(\sum_ {y} M _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}}\right) I \\ \sum_ {\lambda_ {0} = 1} ^ {N} \sum_ {\lambda_ {1} = 1} ^ {N} \dots \sum_ {\lambda_ {D} = 1} ^ {N} \operatorname {t r} \left(\rho_ {x} M _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}}\right) = p (y | x), \\ \end{array}
$$

where the first two constraints are trivial, the third constraint comes from normalization of the POVM, and the last constraint means the POVM should be compatible with the actual experimental statistics. Then one can calculate the upper bound of guessing probability by substituting the statistics $ p ( y | x ) $ collected from the experiment into Eq. (3).

Now we consider a special case where the measurement device is not fully untrusted. We assume it is a phase-insensitive detection, i.e., the POVM elements are diagonal in Fock basis. Under this assumption, Eq. (3) becomes a linear program. We denote diagonal terms of $ \rho_{x} $ and $ M_{y}^{\lambda_{0}\lambda_{1}\cdots\lambda_{D}} $ as $ \vec{\rho}_{x} $ and $ \vec{M}_{y}^{\lambda_{0}\lambda_{1}\cdots\lambda_{D}} $ , respectively. Then,

$$
\begin{array}{l} \max _ {\vec {M} _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}} \geq 0} \quad \sum_ {x = 1} ^ {D} q _ {x} \sum_ {\lambda_ {0} = 1} ^ {N} \sum_ {\lambda_ {1} = 1} ^ {N} \dots \sum_ {\lambda_ {D} = 1} ^ {N} \vec {\rho} _ {x} \cdot \vec {M} _ {\lambda_ {x}} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}} \\ \mathrm {s . t .} \quad \sum_ {y} \vec {M} _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}} = \frac {1}{M} \sum_ {y} \| \vec {M} _ {y} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}} \| _ {1} \vec {1} \\ \sum_ {\lambda_ {0} = 1} ^ {N} \sum_ {\lambda_ {1} = 1} ^ {N} \dots \sum_ {\lambda_ {D} = 1} ^ {N} \vec {\rho} _ {x} \cdot \vec {M} _ {\lambda_ {x}} ^ {\lambda_ {0} \lambda_ {1} \dots \lambda_ {D}} = p (y | x), \\ \end{array}
$$