# Iterated Quantum Prisoner's Dilemma — EWL, Memory and Strategy Tournaments

A computational framework for studying classical and quantum strategies in the
Iterated Prisoner's Dilemma using the Eisert–Wilkens–Lewenstein (EWL) protocol.

The package provides tools for analyzing repeated strategic interactions under
different degrees of quantum entanglement, with particular emphasis on memory,
adaptive behavior, Markov Reward Processes, strategy tournaments, expected
sentence values.

The package reproduces the published results and provides additional analyses developed for the associated Ph.D. thesis called "Dinámica de decisiones y juegos en sistemas
cuánticos entrelazados"

## Overview

The Iterated Prisoner's Dilemma extends the standard Prisoner's Dilemma by
allowing two players to interact repeatedly. Consequently, the action selected
in a given round may depend on information obtained from previous rounds.

This package extends this idea to the quantum domain using the EWL framework.
At every round, the players apply local unitary operations to an entangled
two-qubit system. Measurement produces one of four observable outcomes,

$$
\mathcal{S}=\{CC,CD,DC,DD\},
$$

which determines both the sentence assigned to each player and the information
available for subsequent strategic decisions.

The framework makes it possible to study the interaction between:

- classical reciprocal strategies,
- quantum local operations,
- entanglement,
- quantum interference,
- memory,
- adaptive strategies,
- stochastic dynamics,
- and long-term strategic performance.

---

## EWL Quantum Game

The two-player quantum interaction follows the
Eisert–Wilkens–Lewenstein protocol.

Starting from the state

$$
|00\rangle,
$$


the entangling operator $J(\gamma)$ is applied, where

$$
0\leq\gamma\leq\frac{\pi}{2}.
$$

The players then independently apply local operations

$$
U_A(\theta_A,\phi_A),
\qquad
U_B(\theta_B,\phi_B),
$$

followed by the inverse entangling operation.

The final state is

$$
|\psi_f\rangle =
J^\dagger
(U_A\otimes U_B)
J|00\rangle.
$$

Measurement in the computational basis produces the probabilities

$$
(p_{CC},p_{CD},p_{DC},p_{DD}).
$$

The classical actions are embedded in the EWL strategy space as

$$
C=U(0,0),
\qquad
D=U(\pi,0).
$$

---

## Sentence Convention

Instead of expressing strategic performance through the traditional payoff
matrix, this implementation uses years of sentence.

For player $A$,

$$
\mathbf{c}_A=
\begin{pmatrix}
1\\
5\\
0\\
3
\end{pmatrix},
$$

and for player $B$,

$$
\mathbf{c}_B=
\begin{pmatrix}
1\\
0\\
5\\
3
\end{pmatrix},
$$

using the state ordering

$$
(CC,CD,DC,DD).
$$

Therefore:

| Outcome | Player A | Player B |
|---|---:|---:|
| $CC$ | 1 | 1 |
| $CD$ | 5 | 0 |
| $DC$ | 0 | 5 |
| $DD$ | 3 | 3 |

Unlike a conventional payoff formulation, **lower values correspond to better
strategic performance**.

---

## Strategy Classes

The package contains both classical and quantum strategies.

### Classical strategies

The tournament population includes:

- Tit for Tat (TFT)
- Always Defect (ALLD)
- Always Cooperate (ALLC)
- Random
- Grudger
- Tit for Two Tats (TF2T)

These strategies provide the classical reference population for the iterated
game.

### Quantum strategies

The quantum population contains stationary, adaptive, reciprocal, and
discrete-switching strategies implemented through local EWL operations.

The package currently uses the labels

$$
Q_1,Q_2,Q_3,Q_4,Q_5,Q_9.
$$

These strategies explore different mechanisms for modifying the local quantum
operation according to previous observations.

In particular, adaptive strategies may modify the angular parameter $\theta$
during the interaction, allowing the player's position in the EWL strategy
space to evolve according to the observed behavior of the opponent.

The discrete switching strategy $Q_9$ restricts its response to

$$
\theta \in \left\{\frac{\pi}{4},\frac{3\pi}{4}\right\}.
$$

---

## TFTQ Strategies

The package also investigates quantum generalizations of Tit for Tat.

Classical TFT begins by cooperating and subsequently reproduces the previous
action of the opponent.

Within the EWL parameterization, this corresponds to selecting between

$$
\theta_1=0
\qquad\text{and}\qquad
\theta_2=\pi.
$$

TFTQ preserves the reciprocal decision rule but replaces these extreme
responses with parameterized quantum operations.

The analyzed pairs include

$$
(\theta_1,\theta_2)
\in
\left\{(0,\pi),
\left(\frac{\pi}{16},\frac{15\pi}{16}\right),
\left(\frac{\pi}{8},\frac{7\pi}{8}\right),
\left(\frac{\pi}{4},\frac{3\pi}{4}\right)
\right\}.
$$

This experiment evaluates how the accumulated expected sentence changes when
the response angles and the strategic phase are modified.

---

## Markov Reward Process Formulation

For memory-one strategies, the observable dynamics can be represented as a
Markov process over

$$
\mathcal{S}=\{CC,CD,DC,DD\}.
$$

The transition matrix is

$$
P=
\begin{pmatrix}
P_{CC\rightarrow CC} & \cdots & P_{CC\rightarrow DD}\\
\vdots & \ddots & \vdots\\
P_{DD\rightarrow CC} & \cdots & P_{DD\rightarrow DD}
\end{pmatrix}.
$$

If $\mathbf{p}_t$ is the probability distribution over the four observable
states at round $t$, then

$$
\mathbf{p}_{t+1}=\mathbf{p}_tP.
$$

The expected sentence of player $i$ at round $t$ is

$$
E_i(t)=\mathbf{p}_t\mathbf{c}_i.
$$

The expected cumulative sentence over $T$ rounds is

$$
C_i(T)=\sum_{t=0}^{T-1}
\mathbf{p}_0P^t\mathbf{c}_i.
$$

The package compares these analytical predictions with direct stochastic
simulation of the repeated game.

---

## Stationary Behavior

When the induced Markov chain admits a unique stationary distribution,

$$
\boldsymbol{\pi}=\boldsymbol{\pi}P,
$$

with

$$
\sum_s\pi_s=1,
$$

the asymptotic expected sentence per round is

$$
\overline{c}_{i,\infty}=\boldsymbol{\pi}\mathbf{c}_i.
$$

This provides a long-time characterization of memory-one strategy profiles.

---

## Extended Memory

Not every strategy is Markovian on the observable state space
$\{CC,CD,DC,DD\}$.

Strategies such as Tit for Two Tats may require information from more than one
previous round. In these cases, the current outcome alone does not contain
enough information to determine the next strategic response.

The package therefore distinguishes between:

- memory-one dynamics,
- finite extended-memory dynamics,
- adaptive quantum strategies,
- and direct trajectory simulation.

A finite-memory strategy can also be represented as a Markov process by
augmenting the state space to include the required history.

---

## Round-Robin Tournament

The package performs exhaustive round-robin tournaments between the classical
and quantum strategy populations.

For each strategy $S_i$, its performance against $S_j$ is stored in a sentence
matrix,

$$
\mathbf{C}=[C_{ij}],
$$

where $C_{ij}$ is the mean sentence per round obtained by $S_i$ when playing
against $S_j$.

Each cell therefore represents an **individual matchup**.

The global performance of $S_i$ is

$$
\overline{C}_i=\frac{1}{|\mathcal{S}_T|}
\sum_j C_{ij}.
$$

This distinction is important:

- $C_{ij}$ describes a particular matchup;
- $\overline{C}_i$ describes performance against the complete tournament
  population.

Since sentence is minimized, the strategy with the smallest
$\overline{C}_i$ occupies the highest position in the ranking.

---

## Classical-Only Benchmark

A separate analysis considers only the classical-versus-classical block of
the tournament.

For

$$
S_i\in\mathcal{S}_C,
$$

the classical-group score is calculated from interactions exclusively with
other classical strategies.

This provides a reference closer to the logic of traditional Iterated
Prisoner's Dilemma tournaments and allows reciprocal strategies such as TFT,
TF2T, and Grudger to be evaluated independently of the quantum population.

---

## Entanglement Sweep

The package evaluates how strategic performance changes with the EWL
entanglement parameter

$$
0\leq\gamma\leq\frac{\pi}{2}.
$$

The analysis includes:

- mean sentence versus $\gamma$,
- tournament ranking versus $\gamma$,
- classical and quantum group averages,
- outcome frequencies versus $\gamma$,
- and changes in the relative ordering of strategies.

In the generated figures:

- **solid lines** represent classical strategies;
- **dashed lines** represent quantum strategies.

This convention makes the two strategy groups visually distinguishable while
preserving the individual strategy labels.

---

## Outcome Distribution Analysis

Strategic performance cannot be fully characterized from the frequency of
mutual cooperation alone.

The package therefore calculates the four outcome frequencies

$$
(f_{CC},f_{CD},f_{DC},f_{DD}),
$$

which satisfy

$$
f_{CC}+f_{CD}+f_{DC}+f_{DD}=1.
$$

For player $A$, the mean sentence can be reconstructed as

$$
\overline{C}_A =
f_{CC}
+
5f_{CD}
+
3f_{DD},
$$

because

$$
c_A(DC)=0.
$$

The four quantities have different strategic interpretations:

- $f_{CC}$ — mutual cooperation,
- $f_{CD}$ — player $A$ cooperates while $B$ defects,
- $f_{DC}$ — player $A$ defects while $B$ cooperates,
- $f_{DD}$ — mutual defection.

This decomposition makes it possible to determine **why** a strategy obtains
a particular mean sentence rather than using cooperation frequency as a
surrogate for performance.

---

## Main Analyses Included

The package generates numerical data and figures for:

1. EWL outcome probabilities.
2. Finite-horizon Markov Reward Process dynamics.
3. Analytical versus simulation validation.
4. Stationary distributions and asymptotic sentences.
5. Quantum interference diagnostics.
6. Adaptive quantum strategies.
7. Extended-memory and non-Markovian behavior.
8. TFTQ parameter sensitivity.
9. Classical and quantum round-robin tournaments.
10. Individual matchup sentence matrices.
11. Global strategy rankings.
12. Classical-only tournament benchmarks.
13. Entanglement-dependent strategy reordering.
14. Classical versus quantum group averages.
15. $CC$, $CD$, $DC$, and $DD$ outcome-frequency analysis.

---

## Output Structure

The package produces three main types of output.



## Reproducibility

The numerical experiments are designed so that the reported figures and
tables can be regenerated from the included source code.

For stochastic tournament experiments, fixed random seeds should be used when
exact numerical reproducibility is required.



## Scientific Scope

This package is intended as a research framework for investigating the
interaction between memory and quantum resources in repeated strategic
decision-making.

Its main purpose is not to claim a universal advantage of quantum strategies
over classical strategies. Instead, it provides a controlled computational
environment for determining how specific decision policies interact with
entanglement, interference, memory, and opponent behavior.

The resulting framework connects repeated quantum games with stochastic
processes, Markov Reward Processes, adaptive decision rules, and tournament
analysis.

---

## Citation

If this software is used in academic work, please cite the associated Ph.D. thesis

---

## License

Add the selected software license here before public release.
