<div align="center">

# Measurement-device-independent entanglement and randomness estimation in quantum networks

</div>

Ivan Šupić, $ ^{1,*} $ Paul Skrzypczyk, $ ^{2} $ and Daniel Cavalcanti $ ^{1} $

$ ^{1} $ICFO-Institut de Ciencies Fotoniques, The Barcelona Institute of Science and Technology, 08860 Castelldefels (Barcelona), Spain $ ^{2} $H. H. Wills Physics Laboratory, University of Bristol, Tyndall Avenue, Bristol BS8 1TL, United Kingdom (Received 5 March 2017; published 26 April 2017)

Detection of entanglement in quantum networks consisting of many parties is one of the important steps toward building quantum communication and computation networks. We consider a scenario where the measurement devices used for this certification are uncharacterized. In this case, it is well known that by using quantum states as inputs for the measurement devices it is possible to detect any entangled state (a situation known as measurement-device-independent entanglement witnessing). Here we go beyond entanglement detection and provide methods to estimate the amount of entanglement in a quantum network. We also consider the task of randomness certification and show that randomness can be certified in a variety of cases, including single-partite experiments or setups using only separable states.

DOI: 10.1103/PhysRevA.95.042340

## I. INTRODUCTION

Quantum entanglement is one of the most intriguing aspects of quantum theory [1]. Implications of its existence on the foundations of quantum theory were already emphasized in the seminal work of Einstein et al. [2]. In the past few decades entanglement has been at the core of the power of quantum theory for different information-processing tasks [1]. Moreover, it allows quantum systems to display nonlocal correlations, which can be certified by the violation of Bell inequalities [3,4]. Thus, whenever a Bell inequality violation is observed, one certifies that the underlying quantum system is entangled. Importantly, this certification is done without the need to know which measurements are being performed (in contrast to the case of entanglement witnesses [1]). We can thus say that the observation of nonlocal correlations certifies entanglement in a device-independent way [4].

The relation between Bell nonlocality and entanglement is one of the open questions in the foundations of quantum theory. While nonlocal correlations can only be obtained by performing local measurements on an entangled state, not every entangled state can lead to nonlocality. As was first shown by Werner [5], the probability distributions obtained by local measurements applied to certain entangled quantum states can be simulated by a purely classical model called a local-hidden-variable model. Werner's result was later generalized in many ways [6-17].

Since the standard Bell scenario exposes a gap between nonlocality and entanglement, one may ask if it is possible to come up with some alternative scenario in which every entangled state will exhibit nonlocal correlations. One possibility is to make use of many copies of the shared entangled state. It has been proven that some entangled states which are not nonlocal in the standard Bell scenario can violate a Bell inequality in this scenario [18,19], a phenomenon known as superactivation. Whether every entangled state can be superactivated in this way is again an open problem. Another possibility is to

consider a quantum network consisting of many copies of the same state shared among many parties [20-22]. Similar to superactivation, some states that are local in the standard Bell scenario become nonlocal in the network scenario. However, the general relation between entanglement and nonlocality is also unknown in this case. Another alternative Bell scenario, historically preceding the others, was suggested in Ref. [23] (see also Ref. [11]) and is known as the hidden nonlocality scenario. In it the parties are allowed to perform preprocessing on their shared state before applying their measurements. Even though some local entangled states can become nonlocal in this scenario, it has recently been shown that there are entangled states which can never exhibit hidden nonlocality; i.e., they remain local after arbitrary preprocessing [24]. Finally, it is possible to combine all the nonstandard Bell scenarios, but still there is no conclusive statement about the relation between entanglement and nonlocality in what would be called a multicopy with preprocessing scenario.

There is, however, one modification of the standard Bell scenario that can reveal nonlocal correlations from every entangled state. It consists of using measurement devices that receive quantum systems as inputs [25] (see also Refs. [26-30] for further developments and Refs. [31,32,34] for experimental demonstrations). The observation of nonlocal correlations in this scenario can be seen as an entanglement test with uncharacterized measurements, which motivated the name of measurement-device-independent nonlocality.

The aim of this paper is to shed further light on some aspects of this scenario with quantum inputs, explore its power for entanglement detection and quantification in quantum networks, and finally to study randomness certification in this scenario.

## II. MEASUREMENT-DEVICE-INDEPENDENT ENTANGLEMENT CERTIFICATION

We start by reviewing some of the main results on entanglement certification in the Bell scenario with quantum inputs and by proposing some improvements on this task. We consider two separated parties, Alice and Bob, sharing

a bipartite system in an unknown state $ \rho^{A B}\in\mathcal{L}(\mathcal{H}_{A}\otimes\mathcal{H}_{B}) $ They want to certify if their system is entangled but do not know how their measurement devices work. Buscemi proposed a solution to this problem [25]: At each round of the experiment Alice and Bob encode their measurement choices in quantum states $ \psi_{x}\in\mathcal{L}(\mathcal{H}_{A_{0}}) $ and $ \psi_{y}\in\mathcal{L}(\mathcal{H}_{B_{0}}) $ respectively, which they use as inputs for their measurement devices. After receiving the quantum inputs, the measurement devices provide classical outputs a and b according to the probability distributions

$$
p (a, b | \psi_ {x}, \psi_ {y}) = \operatorname {T r} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B B _ {0}}\right) \left(\psi_ {x} ^ {A _ {0}} \otimes \rho^ {A B} \otimes \psi_ {y} ^ {B _ {0}}\right) \right],
$$

where $ M_{a}^{A_{0}A} $ and $ M_{b}^{BB_{0}} $ are the measurement operators applied by the measurement devices on the corresponding input systems and their shares of state $ \rho^{AB} $ . Buscemi proved that if $ \{\psi_{x/y}\}_{x/y} $ correspond to tomographically complete sets of input states $ ^{1} $ in $ \mathcal{H}_{A/B} $ , and each box performs a Bell state measurement, then every entangled state $ \rho^{AB} $ produces a set of probability distributions $ \{p(a,b|\psi_{x},\psi_{y})\}_{a,b,x,y} $ that cannot be reproduced with any separable state.

Building on Buscemi's result, the authors of Ref. [26] provided a method of constructing Bell tests with quantum inputs from every entanglement witness and named them measurement-device-independent entanglement witnesses (MDIEW). However, when it comes to practical entanglement detection, MDIEW are useful only when one has a good guess about which entangled state should be detected, and can thus start from an entanglement witness which is able to detect its entanglement. This problem was first addressed in Ref. [33], where it was shown that, given multiple copies of the state, a universal witness can be found. More recently, in Ref. [34] a solution at the single-copy level was given, by showing that the quantum inputs scenario can be cast as a semidefinite programming (SDP) optimization problem [35], readily solvable with available software. We present here the said SDP in a slightly different form.

The starting point is the fact that the joint outcome probability distribution can be written in the following way:

$$
p (a, b | \psi_ {x}, \psi_ {y}) = \operatorname {T r} \left[ \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \left(\psi_ {x} ^ {A _ {0}} \otimes \psi_ {y} ^ {B _ {0}}\right) \right],
$$

where $ \tilde{M}_{a,b} $ is an effective POVM operator defined by

$$
\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} = \operatorname {T r} _ {A B} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B B _ {0}}\right) \left(\mathbb {I} ^ {A _ {0}} \otimes \rho^ {A B} \otimes \mathbb {I} ^ {B _ {0}}\right) \right].
$$

By construction, the effective positive-operator-valued measure (POVM) $ \tilde{M}_{a,b}^{A_{0}B_{0}} $ satisfies a number of conditions, which can be thought of as playing the role of "no-signaling" conditions. In particular,

$$
\begin{array}{l} \sum_ {a} \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} = \mathbb {I} ^ {A _ {0}} \otimes \tilde {M} _ {b} ^ {B _ {0}}, \\ \sum_ {b} \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} = \tilde {M} _ {a} ^ {A _ {0}} \otimes \mathbb {I} ^ {B _ {0}}, \\ \end{array}
$$

where $ \tilde{M}_{b}^{B_{0}}\equiv \operatorname{Tr}_{B}[M_{b}^{B B_{0}}(\rho^{B}\otimes\mathbb{I}^{B_{0}})]\geqslant 0 $ and $ \tilde{M}_{a}^{A_{0}}\equiv $ $ \operatorname{Tr}_{A}[M_{a}^{A_{0}A}(\mathbb{I}^{A_{0}}\otimes \rho^{A})]\geqslant 0 $ are effective local POVMs for Alice and Bob (i.e., such that $ \sum_{a}\tilde{M}_{a}^{A_{0}}=\mathbb{I}^{A_{0}} $ and $ \sum_{b}\tilde{M}_{b}^{B_{0}}=\mathbb{I}^{B_{0}}). $ We will write $ \{\tilde{M}_{a,b}^{A_{0}B_{0}}\}_{a,b}\in \mathcal{M} $ to denote the fact that the effective POVM satisfies these conditions, i.e.,

$$
\begin{array}{l} \mathcal {M} = \left\{\left\{\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b} \mid \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \geqslant 0, \sum_ {a} \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} = \mathbb {I} ^ {A _ {0}} \otimes \tilde {M} _ {b} ^ {B _ {0}}, \right. \\ \left. \sum_ {b} \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} = \tilde {M} _ {a} ^ {A _ {0}} \otimes \mathbb {I} ^ {B _ {0}}, \sum_ {a} \tilde {M} _ {a} ^ {A _ {0}} = \mathbb {I} ^ {A _ {0}}, \sum_ {b} \tilde {M} _ {b} ^ {B _ {0}} = \mathbb {I} ^ {B _ {0}} \right\}. \\ \end{array}
$$

Now, if the shared state $ \rho^{AB} $ is separable, i.e., $ \rho^{AB}=\sum_{\lambda} p_{\lambda}\rho_{\lambda}^{A}\otimes \rho_{\lambda}^{B} $ , then (3) becomes

$$
\begin{array}{l} \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} = \sum_ {\lambda} p _ {\lambda} \operatorname {T r} _ {A} \left[ M _ {a} ^ {A _ {0} A} \left(\mathbb {I} ^ {A _ {0}} \otimes \rho_ {\lambda} ^ {A}\right) \right] \\ \otimes \operatorname {T r} _ {B} \left[ M _ {b} ^ {B B _ {0}} \left(\rho_ {\lambda} ^ {B} \otimes \mathbb {I} ^ {B _ {0}}\right) \right] \\ = \sum_ {\lambda} p _ {\lambda} \tilde {M} _ {a | \lambda} ^ {A _ {0}} \otimes \tilde {M} _ {b | \lambda} ^ {B _ {0}}, \\ \end{array}
$$

where $ \tilde{M}_{a|\lambda}^{A_{0}} \equiv \operatorname{Tr}_{A}[M_{a}^{A_{0}A}(\mathbb{I}^{A_{0}}\otimes \rho_{\lambda}^{A})]$ and $ \tilde{M}_{b|\lambda}^{B_{0}} \equiv $ $ \operatorname{Tr}_{B}[M_{b}^{BB_{0}}(\rho_{\lambda}^{B}\otimes \mathbb{I}^{B_{0}})] $ are effective local POVMs for Alice and Bob. Consequently, the fact that $ \rho^{AB} $ is separable implies that the operator $ \tilde{M}_{a,b}^{A_{0}B_{0}} $ is a separable operator for all a and b.

Alice and Bob can thus check the separability of $ \rho^{A B} $ by solving the following feasibility problem:

given $ \{p(a,b|\psi_{x},\psi_{y})\}_{a,b,x,y} $

find

$$
\left\{\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b}
$$

s.t.

$$
\begin{array}{l} p (a, b | \psi_ {x}, \psi_ {y}) = \operatorname {T r} \left[ \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \left(\psi_ {x} ^ {A _ {0}} \otimes \psi_ {y} ^ {B _ {0}}\right) \right] \quad \forall a, b, x, y, \\ \left\{\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b} \in \mathcal {S} \\ \end{array}
$$

where s.t. stands for such that, and $ \mathcal{S} $ denotes the subset of $ \mathcal{M} $ which are separable operators, i.e.,

$$
\begin{array}{l} \mathcal {S} = \left\{\left\{\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b} \mid \left\{\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b} \in \mathcal {M}, \right. \\ \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} = \sum_ {\lambda} \tau_ {a | \lambda} \otimes \chi_ {b | \lambda}, \tau_ {a | \lambda} \geqslant 0, \chi_ {b | \lambda} \geqslant 0 \Bigg \}. \\ \end{array}
$$

This problem is in principle hard to solve, due to the lack of an efficient characterization of the set of separable operators. However, one can relax the constraint of separability and impose instead that each operator $ \tilde{M}_{a,b}^{A_{0}B_{0}} $ is positive under partial transpose (PPT). In the feasibility problem above, this amounts to replacing the condition $ \tilde{M}_{a,b}^{A_{0}B_{0}}=\sum_{\lambda}\tau_{a|\lambda}\otimes\chi_{b|\lambda},\tau_{a|\lambda}\geqslant0 $ $ \chi_{b|\lambda}\geqslant0 $ by $ (\tilde{M}_{a,b}^{A_{0}B_{0}})^{T_{A_{0}}}\geqslant0 $ $ \forall a,b. $ With this replacement, the problem becomes a feasibility SDP optimization problem, which can then be solved efficiently.

Notice, however, that this relaxation is not able to detect PPT-entangled states. A second, more stringent, relaxation of the set of separable operators is the set of the operators having a $ k $ - symmetric extension [36]. Imposing that the operators $ \tilde{M}_{a,b}^{A_{0}B_{0}} $ have a $ k $ symmetric extension amounts to demanding

that there exists a $ ( k+1) $ -partite operator $ \tilde{N}_{a,b}^{A_{0}B_{0}\dots B_{k-1}} \geqslant0 $ such that $ \tilde{N}_{a,b}^{A_{0}B_{i}}=\tilde{M}_{a,b}^{A_{0}B_{0}}\forall i. $ For every fixed k, the above feasibility optimization problem with this replacement is again a SDP feasibility problem, which now can also detect PPT entangled states [37]. Finally, we note that by increasing the order k of the extension, we obtain stronger SDP tests that converge to the separability test above in the limit of $ k \to\infty. $

Finally, it is an important fact that once the sets of quantum inputs used are tomographically complete, then the probabilities $ p ( a,b|\psi_{x},\psi_{y}) $ allow for an exact reconstruction of the effective POVM elements $ \tilde{M}_{a,b}^{A_{0}B_{0}} $ , using quantum process tomography. In this special case, any available entanglement criterion, not just those that can be checked via SDP, can be used to determine if it is a separable operator, leading directly to a conclusion of whether the shared state was entangled.

## III. MEASUREMENT-DEVICE-INDEPENDENT ENTANGLEMENT ESTIMATION

One step beyond certifying the presence of entanglement in a system is to estimate how much entanglement it contains. There are many different entanglement measures, but they are in general not easy to compute even if one knows the full state of the system (for more details on entanglement measures, see Ref. [38]). In fact the problem of deciding if a given quantum state is entangled is non-deterministic polynomial (NP)-hard, which implies hardness of computing entanglement measures [39]. Quantification of entanglement in a measurement-device-independent scenario was the subject of Ref. [40], where the authors define the best possible payoff in a semiquantum games that a state can achieve as an entanglement measure. In what follows, we show how to place measurement-device-independent bounds on two well-known entanglement quantifiers, the robustness [41] and the negativity [42].

## A. MDI lower bound on the robustness of entanglement

A physically well-motivated way of quantifying entanglement is through its robustness to noise [41], defined as the amount of noise one can add to an entangled state before it becomes separable. In mathematical terms, the generalized robustness $ r_{g} $ of a state $ \rho $ is given by

$$
r _ {S} \left(\rho^ {A B}\right) = \min _ {r, \sigma^ {A B}} r
$$

$$
\begin{array}{l} \mathrm {s . t .} \quad \frac {\rho^ {A B} + r \sigma^ {A B}}{1 + r} \in \mathrm {s e p}, \\ \sigma^ {A B} \in S \\ \end{array}
$$

where S is a subset of quantum states, which defines the type of robustness, and sep denotes the set of separable states. Typical choices for S include the set of all quantum states (generalized robustness), the set of separable states (robustness), or the maximally mixed state (random robustness).

In a similar way, we define the robustness of MDI nonlocality $ \tilde{r}_{S}^{MDI} $ as the minimum amount of noise that has to be added to the set of probability distributions $ \{p(a,b|\psi_{x},\psi_{y})\}_{a,b,x,y} $ before it can be reproduced by a separable state, where the noise comes from the set S. Formally, the MDI-nonlocality

robustness is the solution of the following optimization problem:

$$
\begin{array}{l} \tilde {r} _ {S} ^ {M D I} [ p (a, b | \psi_ {x}, \psi_ {y}) ] = \min _ {\{\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \} _ {a, b}, \{\tilde {N} _ {a, b} ^ {A _ {0} B _ {0}} \} _ {a, b}} r \\ \mathrm {s. t.} \quad \frac {p (a , b | \psi_ {x} , \psi_ {y}) + r \pi (a , b | \psi_ {x} , \psi_ {y})}{1 + r} \\ = \operatorname {T r} \left[ \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \left(\psi_ {x} ^ {A _ {0}} \otimes \psi_ {y} ^ {B _ {0}}\right) \right], \\ \pi (a, b | \psi_ {x}, \psi_ {y}) = \operatorname {T r} \left[ \tilde {N} _ {a, b} ^ {A _ {0} B _ {0}} \left(\psi_ {x} ^ {A _ {0}} \otimes \psi_ {y} ^ {B _ {0}}\right) \right] \quad \forall a, b, x, y, \\ \left\{\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b} \in \mathscr {S}, \quad \left\{\tilde {N} _ {a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b} \in \mathcal {M} _ {S}. \tag {11} \\ \end{array}
$$

where $ \mathcal{M}_{S} $ is the set of effective POVMs associated to the noise S. For example, for the generalized robustness, when the set S corresponds to all quantum states, then $ \mathcal{M}_{S}=\mathcal{M}. $ Similarly, for the robustness, when the set S corresponds to all separable states, then $ \mathcal{M}_{S}=\mathcal{S}. $ Finally, for the random robustness, when $ S=\{\mathbb{I}^{AB}/d_{A}d_{B}\} $ , then $ \mathcal{M}_{S}=\{\{\tilde{M}_{a,b}^{A_{0}B_{0}}\}_{a,b}|\{\tilde{M}_{a,b}^{A_{0}B_{0}}\}_{a,b}\in $ $ \mathcal{M},\tilde{M}_{a,b}^{A_{0}B_{0}}=\tilde{M}_{a}^{A_{0}}\otimes \tilde{M}_{b}^{B_{0}} \} $

We now show that $ \tilde{r}_{S}^{MDI} $ is a lower bound to the robustness of entanglement $ r_{S} $ of the underlining state being measured. To see this, consider that the robustness of the state $ \rho^{AB} $ is given by $ r_{S}^{*} $ . This means that there exists a state $ \sigma^{*AB}\in S $ for which the state $ (\rho+r_{S}^{*}\sigma^{*})/(1+r_{S}^{*}) $ is separable. Thus for any POVMs $ \{M_{a}^{A_{0}A}\}_{a} $ and $ \{M_{b}^{BB_{0}}\}_{b} $ satisfying (1),

$$
\begin{array}{l} \tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} = \operatorname {T r} _ {A B} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B B _ {0}}\right) \right. \\ \times \left(\mathbb {I} ^ {A _ {0}} \otimes \frac {\rho^ {A B} + r _ {S} ^ {*} \sigma^ {* A B}}{1 + r _ {S} ^ {*}} \otimes \mathbb {I} ^ {B _ {0}}\right) \Bigg ] \\ \end{array}
$$

and

$$
\tilde {N} _ {a, b} ^ {A _ {0} B _ {0}} = \operatorname {T r} _ {A B} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B B _ {0}}\right) \left(\mathbb {I} ^ {A _ {0}} \otimes \sigma^ {* A B} \otimes \mathbb {I} ^ {B _ {0}}\right) \right],
$$

are feasible for the problem (11) (i.e., satisfy all the constraints) and achieve the value $ r=r_{S}^{*} $ , as can be verified by direct substitution. Since $ \{\tilde{M}_{a,b}^{A_{0}B_{0}}\}_{a,b} $ and $ \{\tilde{N}_{a,b}^{A_{0}B_{0}}\}_{a,b} $ given by (12) and (13) do not necessarily provide an optimal solution to the problem (11), then

$$
\tilde {r} _ {S} ^ {M D I} \left[ p \left(a, b \mid \psi_ {x}, \psi_ {y}\right) \right] \leqslant r _ {S} ^ {*} \left(\rho^ {A B}\right).
$$

This bound can be easily interpreted. If the measured state has no entanglement, i.e., $ r_{S}^{*}(\rho^{AB})=0 $ , then the probability distribution obtained by measuring it trivially has a separable realiszation, so $ \tilde{r}_{S}^{MDI}[p(a,b|\psi_{x},\psi_{y})]=0 $ . On the other hand, if Alice and Bob detect that $ \tilde{r}_{S}^{MDI}[p(a,b|\psi_{x},\psi_{y})]>0 $ , then they immediately conclude that the underlining state is entangled, and moreover can place a lower bound on the amount of entanglement, as measured by the robustness (with respect to S), that is necessary to explain the data.

## B. MDI lower bound on the negativity

Another widely used entanglement measure of entanglement is the negativity [42]. Analogous to the deviceindependent estimation of negativity [43], it is possible to

put a lower bound on the negativity in a measurement-device-independent way. The negativity $ \mathcal{N} $ of some state $ \rho^{AB} $ is defined as the sum of the absolute values of the nonpositive eigenvalues of the partially transposed state $ \rho^{T_{A}} $ . It has been shown in Ref. [42] that it admits the following representation:

$$
\begin{array}{l} \mathcal {N} \left(\rho^ {A B}\right) = \min _ {\rho_ {+}, \rho_ {-}} \quad \operatorname {T r} \left[ \rho_ {-} \right], \\ \mathrm {s . t .} \quad \rho^ {A B} = \rho_ {+} - \rho_ {-}, \\ \rho_ {\pm} ^ {T _ {A}} \geqslant 0. \\ \end{array}
$$

Having in mind the decomposition $ \rho^{AB}=\rho_{+}-\rho_{-} $ it is possible to write the observed probabilities from the quantum inputs scenario in the following way:

$$
\begin{array}{l} p (a, b | \psi_ {x}, \psi_ {y}) = \operatorname {T r} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B B _ {0}}\right) \left(\psi_ {x} ^ {A _ {0}} \otimes \rho^ {A B} \otimes \psi_ {y} ^ {B _ {0}}\right) \right] \\ = q _ {+} \left(a, b | \psi_ {x}, \psi_ {y}\right) - q _ {-} \left(a, b | \psi_ {x}, \psi_ {y}\right), \\ \end{array}
$$

where

$$
\begin{array}{l} q _ {+} (a, b | \psi_ {x}, \psi_ {y}) = \operatorname {T r} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B B _ {0}}\right) \left(\psi_ {x} ^ {A _ {0}} \otimes \rho_ {+} \otimes \psi_ {y} ^ {B _ {0}}\right) \right], \\ q _ {-} (a, b | \psi_ {x}, \psi_ {y}) = \operatorname {T r} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B B _ {0}}\right) \left(\psi_ {x} ^ {A _ {0}} \otimes \rho_ {-} \otimes \psi_ {y} ^ {B _ {0}}\right) \right]. \\ \end{array}
$$

According to (15), the negativity can be obtained by minimizing $ \operatorname{Tr}[\rho_{-}], $ which can be written as

$$
\begin{array}{l} \sum_ {a, b} q _ {-} \left(a, b \mid \psi_ {x}, \psi_ {y}\right) \\ = \operatorname {T r} \left[ \left(\sum_ {a} M _ {a} ^ {A _ {0} A} \otimes \sum_ {b} M _ {b} ^ {B B _ {0}}\right) \left(\psi_ {x} ^ {A _ {0}} \otimes \rho_ {-} \otimes \psi_ {y} ^ {B _ {0}}\right) \right] \\ = \operatorname {T r} \left[ \rho_ {-} \right]. \tag {16} \\ \end{array}
$$

In order to estimate the negativity by an SDP optimization, it is necessary to understand the form an effective POVM $ \tilde{M}_{a,b} $ corresponding to a PPT state. We recall that for an arbitrary state an effective POVM must be positive and satisfy no-signaling principle, as encoded in (5), and for a separable state it also has to be a separable operator, as encoded in (8). An effective POVM corresponding to a PPT state, besides satisfying the no-signaling constraints, must also be a PPT operator. To see this, consider partial transpose of an effective POVM:

$$
\begin{array}{l} \left(\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}}\right) ^ {T _ {A _ {0}}} = \operatorname {T r} _ {A B} \left\{\left[ \left(M _ {a} ^ {A _ {0} A}\right) ^ {T _ {A _ {0}}} \otimes M _ {b} ^ {B B _ {0}} \right] \left(\mathbb {I} ^ {A _ {0}} \otimes \rho^ {A B} \otimes \mathbb {I} ^ {B _ {0}}\right) \right\}, \\ = \operatorname {T r} _ {A B} \left\{\left[ \left(M _ {a} ^ {A _ {0} A}\right) ^ {T} \otimes M _ {b} ^ {B B _ {0}} \right] \right. \\ \times \left[ \mathbb {I} ^ {A _ {0}} \otimes \left(\rho^ {A B}\right) ^ {T _ {A}} \otimes \mathbb {I} ^ {B _ {0}} \right] \}. \\ \end{array}
$$

The second equality follows from the fact that $ \operatorname{Tr}[A^{T}B]=\operatorname{Tr}[A B^{T}]. $ Since the full transpose is a CPTP map, $ \left(M_{a}^{A_{0}A}\right)^{T} $ is a positive operator, and thus $ \left(\tilde{M}_{a,b}^{A_{0}B_{0}}\right)^{T_{A_{0}}} $ is positive if the state $ \rho^{AB} $ is PPT, which is exactly the claim we wanted to prove. We will denote by $ \mathcal{P} $ the set of effective POVMs that are also PPT, i.e.,

$$
\mathcal {P} = \left\{\left\{\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b} \mid \left| \left\{\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b} \in \mathcal {M}, \left(\tilde {M} _ {a, b} ^ {A _ {0} B _ {0}}\right) ^ {T _ {A _ {0}}} \geqslant 0 \right\}.
$$

Now we have all the ingredients for the formulation of the SDP whose solution lower bounds the negativity of a

state compatible with some observed probability distribution $ p ( a,b|\psi_{x},\psi_{y}) $ in the quantum inputs scenario:

$$
\begin{array}{l} \mathcal {N} ^ {M D I} [ p (a, b | \psi_ {x}, \psi_ {y}) ] = \min _ {\left\{M _ {\pm , a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b}} \sum_ {a, b} q _ {-} (a, b | \psi_ {x}, \psi_ {y}), \\ \mathrm {s . t .} \quad p (a, b | \psi_ {x}, \psi_ {y}) = q _ {+} (a, b | \psi_ {x}, \psi_ {y}) - q _ {-} (a, b | \psi_ {x}, \psi_ {y}), \\ q _ {\pm} \left(a, b | \psi_ {x} ^ {A _ {0}}, \psi_ {y} ^ {B _ {0}}\right) = \operatorname {T r} \left[ M _ {\pm , a, b} ^ {A _ {0} B _ {0}} \left(\psi_ {x} \otimes \psi_ {y}\right) \right], \quad \forall a, b \\ \left\{M _ {\pm , a, b} ^ {A _ {0} B _ {0}} \right\} _ {a, b} \in \mathcal {P}. \tag {18} \\ \end{array}
$$

## IV. MULTIPARTITE CASE

In this section, we will generalize the previous entanglement detection and quantification techniques to the multipartite scenario. In Buscemi's paper [25], there is an outline of the proof that all multipartite entangled states exhibit some kind of nonlocality when queried with quantum inputs. Moreover, the approach via entanglement witnesses is also explained in Ref. [26] (see also Ref. [44]), but as in the bipartite case the witness is tailored for a specific state. Here, as before, we are interested in the detection of multipartite entanglement without a priori knowledge of the system under study.

In the bipartite case, we saw that the problem reduces to finding a separable effective POVM that returns the observed data when applied to the chosen set of inputs. This generalizes to the multipartite case as follows: Given a certain type of separability, we want to find the properties of the effective POVM which by acting on the given set of quantum inputs returns the observed probability distribution. As expected, we will show that the effective POVM should have the same type of separability properties as the underlying state. For the sake of simplicity, we will consider the tripartite scenario, with the generalization to more parties being straightforward.

The scenario involves three parties, Alice, Bob, and Charlie, each of whom can input quantum systems in the states $ \psi_{x},\psi_{y} $ , and $ \psi_{z} $ respectively in their measuring devices, that subsequently provide classical outputs a, b, and c. The experiment is characterized by the set of joint probabilities of the form

$$
\begin{array}{l} p (a, b, c | \psi_ {x}, \psi_ {y}, \psi_ {z}) = \operatorname {T r} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B _ {0} B} \otimes M _ {c} ^ {C _ {0} C}\right) \right. \\ \times \left(\rho^ {A B C} \otimes \psi_ {x} ^ {A _ {0}} \otimes \psi_ {y} ^ {B _ {0}} \otimes \psi_ {z} ^ {C _ {0}}\right) ], \\ \end{array}
$$

where $ M_{a}^{A_{0}A} $ is a POVM Alice applies to the input $ \psi_{x} $ and her share of the state $ \rho^{ABC} $ , and analogously for $ M_{b}^{B_{0}B} $ and $ M_{c}^{C_{0}C} $ . In the same way as in the bipartite scenario, it is useful to define an effective POVM

$$
\begin{array}{l} \tilde {M} _ {a, b, c} ^ {A _ {0} B _ {0} C _ {0}} = \operatorname {T r} _ {A B C} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B _ {0} B} \otimes M _ {c} ^ {C _ {0} C}\right) \right. \\ \times \left(\rho^ {A B C} \otimes \mathbb {I} ^ {A _ {0} B _ {0} C _ {0}}\right) ] \\ \end{array}
$$

which allows one to write

$$
p (a, b, c | \psi_ {x}, \psi_ {y}, \psi_ {z}) = \operatorname {T r} \left[ \tilde {M} _ {a, b, c} ^ {A _ {0} B _ {0} C _ {0}} \left(\psi_ {x} ^ {A _ {0}} \otimes \psi_ {y} ^ {B _ {0}} \otimes \psi_ {z} ^ {C _ {0}}\right) \right].
$$

As in the bipartite case, the effective POVM elements satisfy a number of constraints by construction, which play the role

of no-signaling conditions. For example,

$$
\sum_ {a} \tilde {M} _ {a, b, c} ^ {A _ {0} B _ {0} C _ {0}} = \mathbb {I} ^ {A _ {0}} \otimes \tilde {M} _ {b, c} ^ {B _ {0} C _ {0}},
$$

with $ \tilde{M}_{b,c}^{B_{0}C_{0}} $ another effective POVM for Bob and Charlie. We will denote the set of effective POVMs which satisfy all such conditions in the tripartite case by $ \mathcal{M}^{ABC}. $

In what follows, we will show that the entanglement properties of the effective POVM elements (20) are the same as the entanglement properties of the shared state $ \rho^{A B C}. $

Fully separable states can be written in the form $ \rho^{A B C}=\sum_{\lambda} p_{\lambda} \rho_{\lambda}^{A} \otimes \rho_{\lambda}^{B} \otimes \rho_{\lambda}^{C} $ and if Alice, Bob, and Charlie share such a state the corresponding effective POVM elements (20) will also be fully separable operators

$$
\tilde {M} _ {a, b, c} ^ {A _ {0} B _ {0} C _ {0}} = \sum_ {\lambda} p _ {\lambda} M _ {a | \lambda} ^ {A _ {0}} \otimes M _ {b | \lambda} ^ {B _ {0}} \otimes M _ {c | \lambda} ^ {C _ {0}},
$$

where $ M_{a|\lambda}^{A_{0}}=\operatorname{Tr}_{A}[M_{a}^{A_{0}A}(\mathbb{I}^{A_{0}}\otimes \rho_{\lambda}^{A})]$ is an effective POVM, and analogously for $ M_{b|\lambda}^{B_{0}} $ and $ M_{c|\lambda}^{C_{0}} $ . Analogous to the bipartite case, we define a subset $ \mathcal{S}^{A|B|C} $ of all effective tripartite POVMs $ \mathcal{M}^{ABC} $ , which are also fully separable, $ \mathcal{S}^{A|B|C}=\{\{\tilde{M}_{a,b,c}^{A_{0}B_{0}C_{0}}\}_{a,b,c}|\{\tilde{M}_{a,b,c}^{A_{0}B_{0}C_{0}}\}_{a,b,c}\in $ $ \mathcal{M}^{ABC},\tilde{M}_{a,b,c}^{A_{0}B_{0}C_{0}}=\sum_{\lambda}\tau_{a|\lambda}\otimes \chi_{b|\lambda}\otimes \omega_{c|\lambda},\tau_{a|\lambda}\geqslant 0,\chi_{b|\lambda}\geqslant $ $ 0,\omega_{c|\lambda}\geqslant 0\} $ . With this in place, full separability of the shared state can thus be cast by the following feasibility problem:

$$
\begin{array}{l} \mathrm {g i v e n} \quad \left\{p (a, b, c | \psi_ {x}, \psi_ {y}, \psi_ {z}) \right\} _ {a, b, c, x, y, z}, \\ \mathrm {f i n d} \quad \left\{\tilde {M} _ {a, b, c} ^ {A _ {0} B _ {0} C _ {0}} \right\} _ {a, b, c} \\ \mathrm {s. t.} \quad p (a, b, c | \psi_ {x}, \psi_ {y}, \psi_ {z}) = \operatorname {T r} \left[ \tilde {M} _ {a, b, c} ^ {A _ {0} B _ {0} C _ {0}} \left(\psi_ {x} ^ {A _ {0}} \otimes \psi_ {y} ^ {B _ {0}} \otimes \psi_ {z} ^ {C _ {0}}\right) \right] \\ \forall a, b, c, x, y, z \\ \left\{\tilde {M} _ {a, b, c} ^ {A _ {0} B _ {0} C _ {0}} \right\} _ {a, b, c} \in \mathscr {S} ^ {A | B | C}. \tag {24} \\ \end{array}
$$

In a way similar to the bipartite scenario, it is necessary to choose an appropriate relaxation of the set of fully separable tripartite operators in order to turn this feasibility problem into an SDP. The set of operators which are PPT across all biparititons is one choice. A second option is to use the multipartite generalisation of the k-shareability hierarchy of SDPs [45].

Tripartite states have a richer entanglement structure than bipartite states, such that even if the problem (24) confirms that there is some entanglement in the system, a full entanglement characterization is not yet complete. It can happen, for example, that the entanglement is shared only between two parties. States which have such entanglement structure are called separable across a certain bipartition. For example, the state $ \rho^{A B C}=\sum_{\lambda} p_{\lambda} \rho_{\lambda}^{A B}\otimes \rho_{\lambda}^{C} $ is separable with respect to the bipartition $ A B | C $ . A state is biseparable if it can be written as a convex combination of states that are separable with respect to different bipartitions:

$$
\begin{array}{l} \rho^ {A B C} = \sum_ {\lambda} p _ {\lambda} ^ {A | B C} \rho_ {\lambda} ^ {A} \otimes \rho_ {\lambda} ^ {B C} + \sum_ {\mu} p _ {\mu} ^ {B | A C} \rho_ {\mu} ^ {B} \otimes \rho_ {\mu} ^ {A C} \\ + \sum_ {\nu} p _ {\nu} ^ {C | A B} \rho_ {\nu} ^ {A B} \otimes \rho_ {\nu} ^ {C}. \\ \end{array}
$$

The strongest form of entanglement that can be present in a tripartite system is genuine multipartite entanglement (GME). A state $ \rho^{A B C} $ is genuinely multipartite entangled if it is not biseparable.

Let us assume that the state shared among Alice, Bob, and Charlie is biseparable (25). In that case, the effective POVM reads

$$
\begin{array}{l} \tilde {M} _ {a, b, c} ^ {A _ {0} B _ {0} C _ {0}} = \sum_ {\lambda} p _ {\lambda} ^ {A | B C} M _ {a | \lambda} ^ {A _ {0}} \otimes M _ {b, c | \lambda} ^ {B _ {0} C _ {0}} \\ + \sum_ {\mu} p _ {\mu} ^ {B | A C} M _ {b | \mu} ^ {B _ {0}} \otimes M _ {a, c | \mu} ^ {A _ {0} C _ {0}} \\ + \sum_ {\nu} p _ {\nu} ^ {A B | C} M _ {a, b | \nu} ^ {A _ {0} B _ {0}} \otimes M _ {c | \nu} ^ {C _ {0}}, \\ \end{array}
$$

where

$$
\begin{array}{l} M _ {a | \lambda} ^ {A _ {0}} = \operatorname {T r} _ {A} \left[ M _ {a} ^ {A _ {0} A} \left(\psi_ {x} ^ {A _ {0}} \otimes \rho_ {\lambda} ^ {A}\right) \right], \\ M _ {b, c | \lambda} ^ {B _ {0} C _ {0}} = \operatorname {T r} _ {B, C} \left[ \left(M _ {b} ^ {B _ {0} B} \otimes M _ {c} ^ {C C _ {0}}\right) \left(\psi_ {y} ^ {B _ {0}} \otimes \rho_ {\lambda} ^ {B C} \otimes \psi_ {z} ^ {C _ {0}}\right) \right], \\ \end{array}
$$

and analogously for all other operators. Thus, the fact that the state $ \rho^{A B C} $ is biseparable implies that the operators (26) are also biseparable.

With this structure in mind, it is possible to construct a feasibility problem to test whether an observed probability distribution in the quantum input scenario can be obtained with a biseparable state:

find

$$
\begin{array}{l} \mathrm {n} \quad \left\{p \left(a, b, c \mid \psi_ {x}, \psi_ {y}, \psi_ {z}\right) \right\} _ {a, b, c, x, y, z}, \\ \mathrm {d} \quad \left\{\tilde {M} _ {a | b, c} ^ {A _ {0} B _ {0} C _ {0}}, \tilde {M} _ {b | a, c} ^ {A _ {0} B _ {0} C _ {0}}, \tilde {M} _ {c | a, b} ^ {A _ {0} B _ {0} C _ {0}} \right\} _ {a, b, c} \\ \mathrm {t.} \quad p (a, b, c | \psi_ {x}, \psi_ {y}, \psi_ {z}) = \operatorname {T r} \left[ \tilde {M} _ {a, b, c} ^ {A _ {0} B _ {0} C _ {0}} \left(\psi_ {x} ^ {A _ {0}} \otimes \psi_ {y} ^ {B _ {0}} \otimes \psi_ {z} ^ {C _ {0}}\right) \right] \\ \forall a, b, c, x, y, z, \\ \tilde {M} _ {a, b, c} ^ {A _ {0} B _ {0} C _ {0}} = \tilde {M} _ {a | b, c} ^ {A _ {0} B _ {0} C _ {0}} + \tilde {M} _ {b | a, c} ^ {A _ {0} B _ {0} C _ {0}} + \tilde {M} _ {c | a, b} ^ {A _ {0} B _ {0} C _ {0}} \quad \forall a, b, c, \\ \left\{\tilde {M} _ {a | b, c} ^ {A _ {0} B _ {0} C _ {0}} \right\} _ {a, b, c} \in \mathcal {S} ^ {A | B, C}, \left\{\tilde {M} _ {b | a, c} ^ {A _ {0} B _ {0} C _ {0}} \right\} _ {a, b, c} \in \mathcal {S} ^ {B | A, C}, \\ \left\{\tilde {M} _ {c | a, b} ^ {A _ {0} B _ {0} C _ {0}} \right\} _ {a, b, c} \in \mathcal {S} ^ {C | A, B}, \tag {27} \\ \end{array}
$$

where $ \mathcal{S}^{A|B,C} $ denotes the subset of effective tripartite POVMs $ \mathcal{M}^{A B C} $ that are also separable across the bipartition $ A|B,C $ and analogously for $ \mathcal{S}^{B|A,C} $ and $ \mathcal{S}^{C|A,B} $ . Once more, by replacing the sets $ \mathcal{S} $ by the set of PPT operators or operators having k-symmetric extension, the above problem becomes an instance of a SDP.

Quantification of multipartite entanglement can be performed in a manner similar to that in the bipartite scenario. Namely, one can lower bound the robustness of genuine multipartite entanglement, or simply robustness of multipartite entanglement, by defining MDI multipartite nonlocality robustness analogously to (11).

## V. RANDOMNESS FROM QUANTUM INPUTS

Nonlocal correlations, as proven by Bell's theorem [3], cannot be explained by any classical, deterministic model or a convex combinations of such models. Consequently, a violation of a Bell inequality can be used to certify that the data generated are intrinsically random. This reasoning led to the development of the protocols for so-called device-independent

randomness certification [46-49]. In these protocols, the amount of (global) randomness stemming from some Bell experiment is characterized by the guessing probability $ G_{x,y} $ with which an external eavesdropper can guess a pair of outcomes observed by Alice and Bob when they make measurements x and y. A lower bound on the guessing probability is

$$
G _ {x ^ {*}, y ^ {*}} = \max _ {a, b} p \left(a, b \mid x ^ {*}, y ^ {*}\right),
$$

and this is the best that an external observer uncorrelated with Alice and Bob can guess. However, an eavesdropper, usually named Eve, can have side information—a system that is correlated (or even entangled) with the state of Alice and Bob. In principle, she could have even provided all the measuring devices and can possibly achieve much better guessing probability than the lower bound (28). Thus the aim of a device-independent randomness estimation protocol is to quantify the randomness of Alice's (and/or Bob's) measurement outcomes by optimizing over all possible eavesdropping strategies of Eve compatible with the obtained Bell inequality violation. The scenario assumes that by sharing a tripartite state, and performing some measurement on her share, Eve steers the state of Alice and Bob. Her strategy is to perform a measurement such that her outcome, denoted by e, will give her the highest probability to guess the pair of outcomes for one particular choice of measurements for Alice and Bob. In such a scenario, calculating Eve's guessing probability, if the obtained value of a Bell expression $ \sum_{a,b,x,y} I_{a,b,x,y} p(a,b|x,y) $ is equal to $ \beta $ , it can be cast as the following optimization problem:

$$
\begin{array}{l} G _ {x ^ {*}, y ^ {*}} = \max _ {\left\{p (a, b, e | x, y) \right\} _ {a, b, e, x, y}} \sum_ {e} p (a, b, e = (a, b) | x ^ {*}, y ^ {*}), \\ \mathrm {s . t} \quad \sum_ {a, b, e, x, y} I _ {a, b, x, y} p (a, b, e | x, y) = \beta ; \\ \left\{p (a, b, e | x, y, z) \right\} _ {a, b, e, x, y} \in \mathcal {Q}, \\ \end{array}
$$

where $ \mathcal{Q} $ denotes the set of quantum behaviours, i.e., the set of all $ \{p(a,b,e|x,y)\}_{a,b,e,x,y} $ that can arise by performing local measurements on a tripartite quantum state. This program gives the highest probability with which Eve's outcome e is the same as Alice's and Bob's outcomes, a and b, for some specific pair of inputs $ x^{*} $ and $ y^{*} $ , with the constraints that the overall probabilities must be compatible with quantum mechanics and the observed violation of a Bell inequality. In general, this problem cannot be solved exactly, due to the set $ \mathcal{Q} $ having no known simple characterization (in particular since it implicitly contains all behaviors compatible with any quantum state and measurements in a Hilbert space of any dimension); however, by using the Navascues-Pironio-Acin (NPA) hierarchy of SDP relaxations of the quantum set of behaviours [50], computable upper bounds can be placed on the guessing probability.

One generalization of this protocol to the quantum-input scenario under the name measurement-device-independent randomness certification has been introduced in Ref. [51]. Analogous to the ability to detect entanglement of all entangled states, even those that do not violate any Bell inequality, the authors of Ref. [51] prove that it is possible to extract randomness from local entangled states in a measurementdevice-independent way. They use the analog of the program (29) with the constraint that the probabilities $ p ( a,b|\psi_{x},\psi_{y}) $

violate the inequality corresponding to a specific MDIEW. As already noted in the previous text, a MDIEW is usually constructed with respect to a specific entangled state. It can nevertheless be used to check if some other entangled state in principle can be useful for randomness extraction. However, as the source providing Alice's and Bob's shared state is uncharacterized, it may not be clear which witness should be used, and therefore it is desirable to have a method to certify randomness that does not rely on a specific MDIEW. Another way to certify randomness in the quantum input scenario is the subject of Ref. [52]. In this approach, the source has to prepare a tomographically complete set of inputs which are used to perform quantum process tomography of the measurement device. Randomness is generated by measuring one of the prepared quantum inputs by the characterized POVM. A method to quantify the amount of randomness is presented for two-outcome POVMs. This approach was used to experimentally generate randomness in a measurementdevice-independent manner [53].

In the following, we will show the way to quantify the amount of randomness resulting from an experiment with quantum inputs without assuming the underlying state. This can be seen as the generalization of the approach of Refs. [54,55] from the standard Bell scenario to the quantum inputs scenario. The protocol works for measurements with an arbitrary number of outputs and the set of quantum inputs does not have to be tomographically complete, which makes it more general than Ref. [52].

Before presenting the more general approach to randomness estimation in the quantum input scenario, let us consider in more detail the essential nature of this scenario, which is the fact that before guessing the measurement outcome Eve has to guess the input state. Due to this, it is possible to extract randomness not only from local entangled states as observed in Ref. [51] but also from a single black box, i.e., without the use of entanglement.

This leads us to the change of scenario: Now we have only one party, Alice, who has a characterized device which prepares quantum input states $ |\psi_{x}\rangle $ . Alice measures these states using an uncharacterized black box, modeled by the POVM $ \{M_{a}\}_{a} $ and obtains some outcomes a. By repeating the process, she can calculate the set of probabilities $ p(a|\psi_{x}) $ to get an outcome a when the quantum input is $ |\psi_{x}\rangle $ . The question is how random Alice's outputs are for Eve. In some cases, Alice can be sure that her outcomes are genuinely random. If the set of quantum inputs is tomographical complete, Alice can perform process tomography and exactly learn which POVM her black box is applying. In the special case when the obtained POVM is extremal, Eve cannot be correlated with Alice's experiment because extremal POVMs cannot be decomposed as a convex combination of other POVMs. Therefore, Eve's guessing probability is obtained simply from (28) (restricted to a single party Alice, instead of Alice and Bob). An extremal dimensional POVMs cannot have more than $ d^{2} $ outcomes [56], which means that when preparing qubit quantum inputs Alice at best can get 2 bits of randomness. One example is the case when Alice prepares the following informationally complete set of quantum inputs $ \{\frac{\mathbb{I}}{2},|0\rangle ,\frac{|0\rangle +|1\rangle}{\sqrt{2}},\frac{|0\rangle +i|1\rangle}{\sqrt{2}}\} $ and observes the probability distribution which corresponds to measuring these inputs with an extremal four-outcome tetrahedral POVM. The

probabilities to get any of the four outcomes when measuring the input which is in the maximally mixed state $ \frac{\mathbb{I}}{2} $ is equal to 0.25, which corresponds to 2 bits of randomness.

In the general case, when the set of quantum inputs is not tomographically complete or the applied POVM is not extremal, it is still possible to construct an SDP-based estimation of randomness analogous to (29). Since the measuring device is uncharacterized, it has to be assumed that it was possibly prepared by Eve. In that case, Eve can be quantumly correlated with the box. That is, she can prepare an ancillary entangled pair of particles $ \rho^{AE} $ and place half inside the box while keeping the other half. In each round, Alice's box then performs a joint measurement $ M_{a}^{A_{0}A} $ , where a corresponds to Alice's outcome, and Eve performs a measurement $ N_{e}^{E} $ , with outcome e. Thus, the probability for Alice to get outcome a and Eve to get e when the quantum input is $ |\psi_{x}\rangle^{A_{0}} $ is

$$
\begin{array}{l} p (a, e | \psi_ {x}) = \operatorname {T r} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes N _ {e} ^ {E}\right) \left(\psi_ {x} ^ {A _ {0}} \otimes \rho^ {A E}\right) \right] \\ = \operatorname {T r} \left[ \tilde {M} _ {a, e} ^ {A _ {0}} \psi_ {x} ^ {A} \right], \\ \end{array}
$$

where $ \tilde{M}_{a,e} $ is an effective POVM which satisfies the relation

$$
\tilde {M} _ {a, e} ^ {A _ {0}} = \operatorname {T r} _ {A E} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes N _ {e} ^ {E}\right) \left(\mathbb {I} ^ {A _ {0}} \otimes \rho^ {A E}\right) \right].
$$

From this relation, it follows that the effective POVM satisfies the relation

$$
\sum_ {a} \tilde {M} _ {a, e} ^ {A _ {0}} = p (e) \mathbb {I} ^ {A _ {0}},
$$

where p(e) $ \geqslant0 $ . As in the above, this can be seen as the no-signaling constraint from Alice to Eve.

With the above in place, the optimization problem which bounds the guessing probability of Eve is

$$
\begin{array}{l} G _ {x ^ {*}} = \max _ {\{\tilde {M} _ {a, e} ^ {A _ {0}} \} _ {a, e}} \operatorname {T r} \sum_ {e} \tilde {M} _ {a = e, e} ^ {A _ {0}} \psi_ {x ^ {*}} ^ {A _ {0}}, \\ \mathrm {s . t} \quad \operatorname {T r} \sum_ {e} \tilde {M} _ {a, e} ^ {A _ {0}} \psi_ {x} ^ {A _ {0}} = p (a | \psi_ {x}), \quad \forall x, a; \\ \sum_ {a} \tilde {M} _ {a, e} ^ {A _ {0}} = p (e) \mathbb {I} ^ {A _ {0}}, \quad \sum_ {e} p (e) = 1, \\ \end{array}
$$

The objective function maximizes the probability for Eve to guess the outcome a when Alice measures the quantum input $ \psi_{x}^{A_{0}} $ , by optimizing over all possible effective POVMs $ \tilde{M}_{a,e}^{A_{0}} $ . The first constraint ensures that effective POVMs are in accordance with the observed probability distribution, while the second imposes no signaling and completeness of the measurement. For any probability distribution obtained by measuring some set of quantum inputs, the above optimization problem, which is an SDP, gives the guessing probability and thus the randomness of the outcomes.

A similar analysis can be applied in the bipartite case. As mentioned above, in the standard Bell scenario one does not need to consider a specific Bell inequality in order for randomness estimation, but rather can use the full nonlocal behavior (set of correlations) obtained in a Bell experiment [54,55]. In what follows, we will generalize this method, providing an alternative way to quantify randomness in a measurement-device-independent manner. Specifically, we will show that it is possible to certify randomness even when Alice and Bob share a separable state.

In the scenario with quantum inputs, Eve again distributes a state to Alice and Bob with which she is entangled. By performing a local POVM $ N_{e} $ on her share, and conditioned on the outcome e, she prepares an subnormalized state $ \rho_{e}^{AB} $ whose norm is equal to the probability for Eve to obtain the outcome, $ \mathrm{Tr}\rho_{e}^{AB}=p(e) $ . The full joint probability is given by

$$
\begin{array}{l} p (a, b, e | \psi_ {x}, \psi_ {y}) = \operatorname {T r} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B B _ {0}}\right) \left(\psi_ {x} ^ {A _ {0}} \otimes \rho_ {e} ^ {A B} \otimes \psi_ {y} ^ {B _ {0}}\right) \right] \\ = \operatorname {T r} \left[ \tilde {M} _ {a, b, e} ^ {A _ {0} B _ {0}} \left(\psi_ {x} ^ {A _ {0}} \otimes \psi_ {y} ^ {B _ {0}}\right) \right], \tag {34} \\ \end{array}
$$

where $ \tilde{M}_{a,b,e} $ is an effective POVM defined by

$$
\tilde {M} _ {a, b, e} ^ {A _ {0} B _ {0}} = \operatorname {T r} _ {A B} \left[ \left(M _ {a} ^ {A _ {0} A} \otimes M _ {b} ^ {B B _ {0}}\right) \left(\mathbb {I} ^ {A _ {0}} \otimes \rho_ {e} ^ {A B} \otimes \mathbb {I} ^ {B _ {0}}\right) \right].
$$

These effective POVMs, apart from satisfying the completeness relation $ \sum_{a,b,e} \tilde{M}_{a,b,e}^{A_{0}B_{0}}=\mathbb{I}^{A_{0}B_{0}} $ , also satisfy the no-signaling constraints

$$
\begin{array}{l} \sum_ {a} \tilde {M} _ {a, b, e} ^ {A _ {0} B _ {0}} = \mathbb {I} ^ {A _ {0}} \otimes \tilde {M} _ {b, e} ^ {B _ {0}}, \\ \sum_ {b} \tilde {M} _ {a, b, e} ^ {A _ {0} B _ {0}} = \tilde {M} _ {a, e} ^ {A _ {0}} \otimes \mathbb {I} ^ {B _ {0}}, \\ \end{array}
$$

where $ \{ \tilde{M}_{b,e}^{B_{0}} \}_{b} $ and $ \{ \tilde{M}_{a,e}^{A_{0}} \}_{a} $ are subnormalized POVMs (for Bob and Alice respectively), for all values of e, with the same normalization for each e, i.e., $ \sum_{b} \tilde{M}_{b,e}^{B_{0}}=p(e)\mathbb{I}^{B_{0}} $ and $ \sum_{a} \tilde{M}_{a,e}^{A_{0}}=p(e)\mathbb{I}^{A_{0}}. $

Like in the standard Bell scenario, Eve's optimal strategy is to perform a measurement such that the outcome e will be equal to the pair $ ( a,b) $ with as high a probability as possible, for some specific pair of quantum inputs $ \psi_{x^{*}}^{A_{0}} $ and $ \psi_{y^{*}}^{B_{0}} $ . Eve's optimal guessing probability is then the solution of the following SDP:

$$
\begin{array}{l} G _ {x ^ {*}, y ^ {*}} ^ {M D I} = \max _ {\left\{\tilde {M} _ {a, b, e} \right\} _ {a, b, e} ^ {A _ {0} B _ {0}}} \operatorname {T r} \sum_ {e} \tilde {M} _ {a, b, e} ^ {A _ {0} B _ {0}} \left(\psi_ {x ^ {*}} ^ {A _ {0}} \otimes \psi_ {y ^ {*}} ^ {B _ {0}}\right), \\ \mathrm {s . t .} \quad \operatorname {T r} \sum_ {e} \tilde {M} _ {a, b, e} ^ {A _ {0} B _ {0}} \left(\psi_ {x} ^ {A _ {0}} \otimes \psi_ {y} ^ {B _ {0}}\right) \\ = p (a, b | \psi_ {x}, \psi_ {y}), \forall x, y, a, b, \\ \tilde {M} _ {a, b, e} ^ {A _ {0} B _ {0}} \geqslant 0 \quad \forall a, b, e, \\ \sum_ {a} \tilde {M} _ {a, b, e} ^ {A _ {0} B _ {0}} = \mathbb {I} ^ {A _ {0}} \otimes \tilde {M} _ {b, e} ^ {B _ {0}} \quad \forall b, e, \\ \sum_ {b} \tilde {M} _ {a, b, e} ^ {A _ {0} B _ {0}} = \tilde {M} _ {a, e} ^ {A _ {0}} \otimes \mathbb {I} ^ {B _ {0}} \quad \forall a, e, \\ \sum_ {b} \tilde {M} _ {b, e} ^ {B _ {0}} = p (e) \mathbb {I} ^ {B _ {0}} \quad \forall e, \\ \sum_ {a} \tilde {M} _ {a, e} ^ {A _ {0}} = p (e) \mathbb {I} ^ {A _ {0}} \quad \forall e, \\ \sum p (e) = 1. \tag {37} \\ \end{array}
$$

The objective function is the total probability for Eve to guess Alice's and Bob's outputs for some specific pair of quantum inputs $ \psi_{x^{*}}^{A_{0}} $ and $ \psi_{y^{*}}^{B_{0}} $ . The first constraint imposes consistency with the observed behavior in the experiment, while the remaining constraints ensure a valid effective POVM (which is normalized and no signaling).

As an example, this program can be used to obtain the optimal guessing probability compatible with the probability

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026032223135444b898677a7745ee%2Fcrop_1_1774192456928.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=LI2IMzIfcPMGP2iISJg6k3SBkbU%3D&Expires=1774797256' alt='OCR图片'/></div>

<div align="center">

FIG. 1. Min-entropy $ (-\log G_{x^{*}y^{*}}) $ vs noise w for the probability distribution that arises by performing Bell state measurements on the two-qubit Werner state, and quantum inputs along the vertices of a tetrahedron on the Bloch sphere. For all $ w\neq0 $ (i.e., whenever the state is not equal to the maximally mixed state), then randomness can be certified. This graph was produced using the CVX [57] package for MATLAB and the toolbox QETLAB [58].

</div>

distribution which arises by Alice and Bob performing Bell state measurements on a shared Werner state $ |\Psi\rangle \langle \Psi|= w |\Phi^{+}\rangle \langle \Phi^{+}|+(1-w) \frac{\mathbb{I}}{4} $ and with quantum inputs corresponding to the vertices of tetrahedron on the Bloch sphere. The resulting guessing probability in terms of parameter w is presented in Fig. 1.

The correlations obtained on the maximally entangled state $ ( w=1) $ allow us to extract four bits of randomness because the guessing probability is $ \frac{1}{16} $ . As in Ref. [51] some randomness can be observed from all entangled Werner states, even those admitting a local model. What may be particularly surprising is that actually all Werner states except for the maximally mixed state manifest some randomness. As commented earlier, intuitively this can be explained by the fact that Eve cannot with certainty guess which quantum input was used, which makes the probability distribution random even when there is no nonlocality at all.

## VI. DISCUSSION

In this work we have provided insights into entanglement and randomness detection and quantification in the

measurement-device-independent scenario. As explained, this scenario differs from the well-known device-independent scenario by the fact that the parties possess a characterized device that can prepare quantum system in some defined quantum states. The scenario in which some parties do not trust their sources and measurements but have a characterized preparation device is not so uncommon in quantum information processing and has been used for constructing protocols for universal blind quantum computation [59]. In particular, we showed how one can estimate the values of two widely used entanglement measures, robustness-based quantifiers of entanglement, and entanglement negativity. Furthermore, we showed how entanglement detection and quantification can be performed in quantum networks (i.e., in situations involving multiple parties, not just two).

On the other hand, we showed how possessing a characterized preparation device can decrease adversarial power in guessing measurement outputs. Already a single party which can prepare specific states and measure them with a black box can extract two bits of randomness. Two parties sharing some quantum state can extract randomness even when they do not share any entanglement.

There are a number of interesting directions for future work. First is to study whether the results presented here for the measurement-device-independent scenario can be adapted to other device-independent scenarios. A second interesting avenue is to explore the prospects with regard to full quantum state recovery. A third direction is to focus on mixed quantum inputs—in which can one can imagine that the eavesdropper holds a purification. It is interesting to ask how this affects entanglement detection, and randomness estimation.

Note added. After finishing this work, we learned about independent work of Rosset et al. [30] also dealing with measurement-device-independent entanglement quantification.

## ACKNOWLEDGMENTS

This work was supported by the Ramon y Cajal fellowship program, Spanish MINECO(SEVERO OCHOA program for Centres of Excellence in R&D SEV-2015-0522 and National Plan Project QIBEQI FIS2016-80773-P), the AXA Chair in Quantum Information Science, Generalitat de Catalunya (SGR875 and CERCA Programme), Fundació Privada Cellex and ERC CoG QITBOX, and the Royal Society University Research Fellowship program.

[1] R. Horodecki, P. Horodecki, M. Horodecki, and K. Horodecki Quantum entanglement, Rev. Mod. Phys. 81, 865 (2009).

[2] A. Einstein, B. Podolsky, and N. Rosen, Can quantummechanical description of physical reality be considered complete? Phys. Rev. 47, 777 (1935).

[3] J. S. Bell, On the Einstein Podolsky Rosen paradox, Physics 1, 195 (1964).

[4] N. Brunner, D. Cavalcanti, S. Pironio, V. Scarani, and S. Wehner, Bell nonlocality, Rev. Mod. Phys. 86, 419 (2014).

[5] R. F. Werner, Quantum states with Einstein-Podolsky-Rosen correlations admitting a hidden-variable model, Phys. Rev. A 40, 4277 (1989).

[6] J. Barrett, Nonsequential positive-operator-valued measurements on entangled mixed states do not always violate a Bell inequality, Phys. Rev. A 65, 042302 (2002).

[7] A. Acín, N. Gisin, and B. Toner, Grothendieck's constant and local models for noisy entangled quantum states, Phys. Rev. A 73, 062105 (2006).

[8] M. L. Almeida, P. Pironio, J. Barrett, G. Tóth, and A. Acín, Noise Robustness of the Nonlocality of Entangled Quantum States, Phys. Rev. Lett. 99, 040403 (2007).

[9] H. M. Wiseman, S. J. Jones, and A. C. Doherty, Steering, Entanglement, Nonlocality, and the Einstein-Podolsky-Rosen Paradox, Phys. Rev. Lett. 98, 140402 (2007).

[10] S. Jevtic, M. J. W. Hall, M. R. Anderson, M. Zwierz, and H. M. Wiseman, Einstein-Podolsky-Rosen steering and the steering ellipsoid, J. Opt. Soc. Am. B 32, A40 (2015).

[11] F. Hirsch, M. T. Quintino, J. Bowles, and N. Brunner, Genuine Hidden Quantum Nonlocality, Phys. Rev. Lett. 111, 160402 (2013).

[12] R. Augusiak, M. Demianowicz, and A. Acín, Local hiddenvariable models for entangled quantum states, J. Phys. A: Math. Theor. 47, 424002 (2014).

[13] J. Bowles, F. Hirsch, M. T. Quintino, and N. Brunner, Genuinely multipartite entangled quantum states with fully local hidden variable models and hidden multipartite nonlocality, Phys. Rev. A 93, 022121 (2016).

[14] F. Hirsch, M. T. Quintino, T. Vértesi, M. F. Pusey, and N. Brunner, Algorithmic Construction of Local Hidden Variable Models for Entangled Quantum States, Phys. Rev. Lett. 117, 190402 (2016).

[15] D. Cavalcanti, L. Guerini, R. Rabelo, and P. Skrzypczyk, General Method for Constructing Local-Hidden-Variable Models for Multiqubit Entangled States, Phys. Rev. Lett. 117, 190401 (2016).

[16] M. Oszmaniec, L. Guerini, P. Wittek, and A. Acín, Simulating positive-operator-valued measures with projective measurements, arXiv:1609.06139.

[17] F. Hirsch, M. Tulio Quintino, T. Vértesi, M. Navascués, and N. Brunner, Better local hidden variable models for two-qubit Werner states and an upper bound on the Grothendieck constant KG(3), arXiv:1609.06114.

[18] C. Palazuelos, Superactivation of Quantum Nonlocality, Phys. Rev. Lett. 109, 190401 (2012).

[19] D. Cavalcanti, A. Acín, N. Brunner, and T. Vértesi, All quantum states useful for teleportation are nonlocal resources, Phys. Rev. A 87, 042104 (2013).

[20] A. Sen De, U. Sen, C. Brukner, V. Buzek, and M. Zukowski, Entanglement swapping of noisy states: A kind of superadditivity in nonclassicality, Phys. Rev. A 72, 042310 (2005).

[21] D. Cavalcanti, M. L. Almeida, V. Scarani, and A. Acín, Quantum networks reveal quantum nonlocality, Nat. Commun. 2, 184 (2011).

[22] D. Cavalcanti, R. Rabelo, and V. Scarani, Nonlocality Tests Enhanced by a Third Observer, Phys. Rev. Lett. 108, 040402 (2012).

[23] S. Popescu, Bell's Inequalities and Density Matrices: Revealing Hidden Nonlocality, Phys. Rev. Lett. 74, 2619 (1995).

[24] F. Hirsch, M. T. Quintino, J. Bowles, T. Vértesi, and N. Brunner, Entanglement without hidden nonlocality, New J. Phys. 18, 113019 (2016).

[25] F. Buscemi, All Entangled Quantum States are Nonlocal, Phys. Rev. Lett. 108, 200401 (2012).

[26] C. Branciard, D. Rosset, Y.-C. Liang, and N. Gisin, Measurement-Device-Independent Entanglement Witnesses for all Entangled Quantum States, Phys. Rev. Lett. 110, 060405 (2013).

[27] D. Rosset, C. Branciard, N. Gisin, and Y.-C. Liang, Correlations of entangled quantum states cannot be classically simulated, New J. Phys. 15, 053025 (2013).

[28] E. G. Cavalcanti, M. J. W. Hall, and H. M. Wiseman, Entanglement verification and steering when Alice and Bob cannot be trusted, Phys. Rev. A 87, 032306 (2013).

[29] M. J. W. Hall, Trust-free verification of steering: Why you can't cheat a quantum referee, in Reality and Measurement in Algebraic Quantum Theory, edited by M. Ozawa (Springer, Berlin, 2017).

[30] D. Rosset et al. (unpublished).

[31] M. Nawareg, S. Muhammad, E. Amselem, and M. Bourennane Experimental measurement-device-independent entanglement detection, Sci. Rep. 5, 8048 (2015).

[32] P. Xu, X. Yuan, L.-K. Chen, H. Lu, X.-C. Yao, X. Ma, Y.-A. Chen, and J.-W. Pan, Implementation of a Measurement- Device-Independent Entanglement Witness, Phys. Rev. Lett. 112, 140506 (2014).

[33] A. Mallick and S. Ghosh, On witnessing arbitrary bipartite entanglement in a measurement device independent way, arXiv:1506.03985 [quant-ph].

[34] E. Verbanis, A. Martin, D. Rosset, C. C. W. Lim, R. T. Thew, and H. Zbinden, Resource-Efficient Measurement Device Independent Entanglement Witness, Phys. Rev. Lett. 116, 190501 (2016).

[35] S. Boyd and L. Vandenberghe, Convex Optimization (Cambridge University Press, Cambridge, UK, 2004).

[36] A. C. Doherty, P. A. Parrilo, and F. M. Spedalieri, Distinguishing Separable and Entangled States, Phys. Rev. Lett. 88, 187904 (2002).

[37] A. C. Doherty, P. A. Parrilo, and F. M. Spedalieri, Complete family of separability criteria, Phys. Rev. A 69, 022308 (2004).

[38] M. B. Plenio and S. Virmani, An introduction to entanglement measures, Quant. Inf. Comput. 7, 1 (2007).

[39] L. Gurvits, Classical complexity and quantum entanglement, J. Comput. Sys. Sci. 69, 448 (2004); L. M. Ioannou, Computational complexity of the quantum separability problem, Quant. Inf. Comput. 7, 335 (2007); S. Gharibian, Strong NP-hardness of the quantum separability problem, ibid. 10, 343 (2010); Y. Huang, Computing quantum discord is NP-hard, New J. Phys. 16, 033027 (2014).

[40] F. Shahandeh, M. J. W. Hall, and T. C. Ralph, Measurement- Device-Independent Approach to Entanglement Measures, Phys. Rev. Lett. 118, 150505 (2017).

[41] G. Vidal and R. Tarrach, Robustness of entanglement, Phys. Rev. A 59, 141 (1999).

[42] G. Vidal and R. F. Werner, A computable measure of entanglement, Phys. Rev. A 65, 032314 (2002).

[43] T. Moroder, J.-D. Bancal, Y.-C. Liang, M. Hofmann, and O. Gühne, Device-Independent Entanglement Quantification and Related Applications, Phys. Rev. Lett. 111, 030501 (2013); G. Tóth, T. Moroder, and O. Gühne, Evaluating Convex Roof Entanglement Measures, ibid. 114, 160501 (2015).

[44] Q. Zhao, X. Yuan, and X. Ma, Efficient measurement-device-independent detection of multipartite entanglement structure, Phys. Rev. A 94, 012343 (2016).

[45] A. C. Doherty, Entanglement and shareability of quantum states, J. Phys. A: Math. Theor. 47, 424004 (2014).

[46] R. Colbeck, Quantum and relativistic protocols for secure multi-party computation, PhD. thesis, University of Cambridge, Cambridge, UK, 2009.

[47] R. Colbeck and A. Kent, Private randomness expansion with untrusted devices, J. Phys. A 44, 095305 (2011).

[48] S. Pironio et al., Random numbers certified by Bell's theorem, Nature (London) 464, 1021 (2010).

[49] A. Acín and L. Masanes, Certified randomness in quantum physics, Nature (London) 540, 213 (2016).

[50] M. Navascués, S. Pironio, and A. Acín, Bounding the Set of Quantum Correlations, Phys. Rev. Lett. 98, 010401 (2007); A convergent hierarchy of semidefinite programs characterizing the set of quantum correlations, New J. Phys. 10, 073013 (2008).

[51] A. Chaturvedi and M. Banik, Measurement-device-independent randomness from local entangled states, Europhys. Lett. 112, 30003 (2015).

[52] Z. Cao, H. Zhou, and X. Ma, Loss-tolerant measurement-device-independent quantum random number generation, New J. Phys. 17, 125011 (2015).

[53] Y.-Q. Nie, J.-Y. Guan, H. Zhou, Q. Zhang, X. Ma, J. Zhang, and J.-W. Pan, Experimental measurement-device-independent quantum random-number generation, Phys. Rev. A 94, 060301 (2016).

[54] O. Nieto-Silleras, S. Pironio, and J. Silman, Using complete measurement statistics for optimal device-independent randomness evaluation, New J. Phys. 16, 013035 (2014).

[55] J.-D. Bancal, L. Sheridan, and V. Scarani, More randomess from the same data, New J. Phys. 16, 033011 (2014).

[56] G. M. D'Ariano, P. Lo Presti, and P. Perinotti, Classical randomness in quantum measurements, J. Phys. A 38, 5979 (2005).

[57] M. Grant and S. Boyd, CVX: Matlab Software for Disciplined Convex Programming, version 2.1, 2004 http://cvxr.com/cvx.

[58] N. Johnston, QETLAB: A MATLAB toolbox for quantum entanglement, version 0.9, 2016 http://www.qetlab.com

[59] A. Broadbent, J. Fitzsimons, and E. Kashefi, Universal blind quantum computation, in Proceedings of the 50th Annual IEEE Symposium on Foundations of Computer Science (FOCS 2009) (IEEE, Piscataway, NJ, 2009), pp. 517-526.