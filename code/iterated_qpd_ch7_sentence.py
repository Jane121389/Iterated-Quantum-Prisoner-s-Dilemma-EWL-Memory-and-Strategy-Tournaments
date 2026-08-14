#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================
Dinámica de la toma de decisiones en el Dilema del Prisionero Cuántico Iterativo.

Este script respalda las Secciones 7.3--7.5 de la tesis:
  7.3 Representación mediante un Proceso de Recompensa de Markov (MRP)
  7.4 Estrategias clásicas y cuánticas
  7.5 Memoria extendida y dinámica no markoviana

Los cálculos principales utilizan NumPy para que el análisis sea exacto y
reproducible sin requerir un SDK cuántico. Qiskit es opcional y se utiliza
únicamente para validar el circuito y las probabilidades EWL frente a la
implementación matricial.

Resultados generados:
  figures/*.png
  tables/*.csv
  summary.txt

Uso:
    python iterated_qpd_ch7.py
    python iterated_qpd_ch7.py --output results_ch7
    python iterated_qpd_ch7.py --shots 5000 --seed 1234
    python iterated_qpd_ch7.py --validate-qiskit   # opcional, requiere Qiskit

Convenciones:
  Orden de los estados: (CC, CD, DC, DD)
  Primer símbolo = jugador A, segundo símbolo = jugador B
  Convención de condena (años): orden de los estados (CC,CD,DC,DD)
  EWL entangler: J(gamma) = exp[-i gamma/2 (D tensor D)]
  D = [[0,1],[-1,0]]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Constantes y objetos cuánticos básicos
# -----------------------------------------------------------------------------

STATE_LABELS: Tuple[str, ...] = ("CC", "CD", "DC", "DD")
STATE_INDEX: Dict[str, int] = {s: i for i, s in enumerate(STATE_LABELS)}
BASIS = np.eye(4, dtype=complex)
RNG_DEFAULT_SEED = 20260809

SENTENCE_A = np.array([1.0, 5.0, 0.0, 3.0], dtype=float)
SENTENCE_B = np.array([1.0, 0.0, 5.0, 3.0], dtype=float)

# Los valores menores corresponden a un mejor desempeño estratégico.
# Correspondencia en el orden de estados (CC, CD, DC, DD):
# A -> (1, 5, 0, 3) años; B -> (1, 0, 5, 3) años.

D_GATE = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
I4 = np.eye(4, dtype=complex)
K_DD = np.kron(D_GATE, D_GATE)

KET00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
RHO0 = np.outer(KET00, KET00.conj())


def U_ewl(theta: float, phi: float = 0.0) -> np.ndarray:
    """
    Estrategia local de dos parámetros del esquema EWL.

    El parámetro theta controla la mezcla entre las acciones clásicas
    cooperación y defección, mientras que phi introduce una fase relativa
    que permite explorar estrategias genuinamente cuánticas.
    """
    # Se calculan una sola vez los factores trigonométricos que aparecen
    # en la parametrización estándar de la estrategia U(theta, phi).
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array(
        [[np.exp(1j * phi) * c, s],
         [-s, np.exp(-1j * phi) * c]],
        dtype=complex,
    )


def J_ewl(gamma: float) -> np.ndarray:
    """
    Operador de entrelazamiento del protocolo EWL:

        J(gamma) = exp[-i gamma/2 (D tensor D)].

    gamma controla la intensidad del entrelazamiento: gamma=0 corresponde
    al régimen no entrelazado y gamma=pi/2 al caso de entrelazamiento máximo
    dentro de esta parametrización.
    """
    # Como (D tensor D)^2 = I, la exponencial matricial puede evaluarse de
    # forma analítica mediante cos(gamma/2) e i sin(gamma/2), evitando una
    # exponenciación numérica general.
    return np.cos(gamma / 2.0) * I4 - 1j * np.sin(gamma / 2.0) * K_DD


def ewl_statevector(UA: np.ndarray, UB: np.ndarray, gamma: float) -> np.ndarray:
    """
    Calcula exactamente el estado final del protocolo EWL:

        |psi_f> = J^dagger (U_A tensor U_B) J |00>.

    UA y UB son las estrategias locales de los jugadores A y B.
    """
    # Construye el entrelazador para el valor de gamma seleccionado.
    J = J_ewl(gamma)

    # Primera etapa del protocolo: prepara el estado entrelazado J|00>.
    psi = J @ KET00

    # Los jugadores aplican simultáneamente sus estrategias locales.
    # np.kron(UA, UB) representa el producto tensorial U_A tensor U_B.
    psi = np.kron(UA, UB) @ psi

    # El desentrelazamiento J^dagger transforma las correlaciones cuánticas
    # en amplitudes asociadas con los resultados observables CC, CD, DC y DD.
    psi = J.conj().T @ psi
    return psi


def ewl_probabilities(UA: np.ndarray, UB: np.ndarray, gamma: float) -> np.ndarray:
    """
    Obtiene las probabilidades de medición en el orden (CC, CD, DC, DD).

    Se aplica directamente la regla de Born:
        P(s) = |<s|psi_f>|^2.
    """
    psi = ewl_statevector(UA, UB, gamma)

    # El módulo cuadrado de cada amplitud compleja proporciona la
    # probabilidad de observar cada resultado del Dilema del Prisionero.
    p = np.abs(psi) ** 2

    # Elimina residuos numéricos extremadamente pequeños y renormaliza para
    # garantizar que la suma de probabilidades sea exactamente uno.
    p[np.abs(p) < 1e-15] = 0.0
    return p / p.sum()


def decohered_intermediate_probabilities(
    UA: np.ndarray, UB: np.ndarray, gamma: float
) -> np.ndarray:
    """
    Baseline where coherences are removed after local strategies and before J^†.
    This is useful only as an interference diagnostic.
    """
    J = J_ewl(gamma)
    rho = J @ RHO0 @ J.conj().T
    UAB = np.kron(UA, UB)
    rho = UAB @ rho @ UAB.conj().T
    # Se anulan todos los términos fuera de la diagonal de rho. Esta operación
    # elimina las coherencias en la base computacional y permite comparar el
    # resultado EWL coherente contra una referencia sin interferencia cuántica.
    rho = np.diag(np.diag(rho))
    rho = J.conj().T @ rho @ J
    p = np.real(np.diag(rho))
    p = np.maximum(p, 0.0)
    return p / p.sum()


# -----------------------------------------------------------------------------
# Definición de estrategias
# -----------------------------------------------------------------------------


def opponent_last_action(state: str, player: str) -> str:
    """Opponent's action encoded in an outcome state."""
    if player == "A":
        return state[1]
    if player == "B":
        return state[0]
    raise ValueError("player must be 'A' or 'B'")


def own_last_action(state: str, player: str) -> str:
    return state[0] if player == "A" else state[1]


@dataclass
class MemoryOnePolicy:
    name: str
    selector: Callable[[str, str], np.ndarray]

    def unitary(self, last_state: str, player: str) -> np.ndarray:
        return self.selector(last_state, player)


def policy_allc() -> MemoryOnePolicy:
    return MemoryOnePolicy("ALLC", lambda s, p: U_ewl(0.0, 0.0))


def policy_alld() -> MemoryOnePolicy:
    return MemoryOnePolicy("ALLD", lambda s, p: U_ewl(np.pi, 0.0))


def policy_tft() -> MemoryOnePolicy:
    def selector(s: str, player: str) -> np.ndarray:
        a = opponent_last_action(s, player)
        return U_ewl(0.0, 0.0) if a == "C" else U_ewl(np.pi, 0.0)
    return MemoryOnePolicy("TFT", selector)


def policy_pavlov() -> MemoryOnePolicy:
    """Win-stay, lose-shift using R/T as win and S/P as lose."""
    def selector(s: str, player: str) -> np.ndarray:
        own = own_last_action(s, player)
        opp = opponent_last_action(s, player)
        win = (own == "C" and opp == "C") or (own == "D" and opp == "C")
        next_a = own if win else ("D" if own == "C" else "C")
        return U_ewl(0.0, 0.0) if next_a == "C" else U_ewl(np.pi, 0.0)
    return MemoryOnePolicy("Pavlov", selector)


def policy_stationary_q(theta: float = np.pi / 2, phi: float = 0.0, name: str = "Q1") -> MemoryOnePolicy:
    U = U_ewl(theta, phi)
    return MemoryOnePolicy(name, lambda s, p: U)


def policy_quantum_tft(theta_c: float = np.pi / 4, theta_d: float = 3*np.pi/4,
                       phi: float = 0.0, name: str = "QTFT") -> MemoryOnePolicy:
    """Política cuántica de memoria uno que alterna entre dos valores no clásicos de theta."""
    def selector(s: str, player: str) -> np.ndarray:
        a = opponent_last_action(s, player)
        return U_ewl(theta_c if a == "C" else theta_d, phi)
    return MemoryOnePolicy(name, selector)


# -----------------------------------------------------------------------------
# Construcción y análisis del MRP
# -----------------------------------------------------------------------------


def transition_matrix(
    policy_A: MemoryOnePolicy,
    policy_B: MemoryOnePolicy,
    gamma: float,
) -> np.ndarray:
    """
    Construye la matriz de transición 4x4 inducida por el circuito cuántico.

    Para estrategias de memoria uno, el estado de la ronda anterior
    s_t en {CC, CD, DC, DD} determina las operaciones locales que se aplican
    en la ronda siguiente. Por tanto, cada elemento de P representa

        P[i,j] = Pr(s_{t+1}=j | s_t=i).

    A diferencia de una cadena de Markov puramente fenomenológica, aquí las
    probabilidades de transición se derivan de la regla de Born aplicada al
    circuito EWL.
    """
    P = np.zeros((4, 4), dtype=float)

    # Cada fila de P corresponde a uno de los cuatro posibles resultados de
    # la ronda anterior. A partir de ese estado se seleccionan U_A y U_B.
    for i, s in enumerate(STATE_LABELS):
        UA = policy_A.unitary(s, "A")
        UB = policy_B.unitary(s, "B")

        # Estas cuatro probabilidades son precisamente las probabilidades de
        # transición hacia CC, CD, DC y DD en la siguiente ronda.
        P[i, :] = ewl_probabilities(UA, UB, gamma)

    # Corrección de redondeo y normalización: toda fila de una matriz de
    # transición debe satisfacer sum_j P[i,j] = 1.
    P = np.maximum(P, 0.0)
    P /= P.sum(axis=1, keepdims=True)
    return P


def initial_distribution(
    policy_A: MemoryOnePolicy,
    policy_B: MemoryOnePolicy,
    gamma: float,
    initial_assumed_state: str = "CC",
) -> np.ndarray:
    """
    First-round distribution using the policies as if the pre-game state were CC.
    This implements the common convention that reciprocal strategies begin cooperatively.
    """
    UA = policy_A.unitary(initial_assumed_state, "A")
    UB = policy_B.unitary(initial_assumed_state, "B")
    return ewl_probabilities(UA, UB, gamma)


def propagate(p0: np.ndarray, P: np.ndarray, T: int) -> np.ndarray:
    """Propaga la distribución de estados desde p_0 hasta p_T."""
    out = np.zeros((T + 1, len(p0)))
    out[0] = p0

    # Implementa iterativamente p_{t+1} = p_t P. Por ello, después de t
    # rondas se tiene la relación equivalente p_t = p_0 P^t.
    for t in range(T):
        out[t + 1] = out[t] @ P
    return out


def finite_horizon_value(P: np.ndarray, cost: np.ndarray, T: int) -> np.ndarray:
    """
    Calcula el valor acumulado esperado para un horizonte finito T:

        V_0 = sum_{t=0}^{T-1} P^t R,

    donde R es aquí el vector de años de condena asociado a cada estado.
    """
    V = np.zeros_like(cost, dtype=float)
    Pt = np.eye(P.shape[0])

    # Se acumula P^t R para t=0,...,T-1. De esta forma V[i] representa
    # la condena total esperada al comenzar en el estado i.
    for _ in range(T):
        V += Pt @ cost
        Pt = Pt @ P
    return V


def cumulative_expected_cost(p0: np.ndarray, P: np.ndarray, cost: np.ndarray, T: int) -> float:
    return float(p0 @ finite_horizon_value(P, cost, T))


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Calcula la distribución estacionaria pi que satisface

        pi P = pi,    sum_i pi_i = 1.

    Por ello pi es un autovector izquierdo de P asociado al autovalor 1.
    """
    # Buscar un autovector izquierdo de P equivale a buscar un autovector
    # derecho de P.T. Se selecciona el autovalor numéricamente más cercano a 1.
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    v = np.real(eigvecs[:, idx])
    # Sign ambiguity; use abs if tiny numerical negatives occur.
    if v.sum() < 0:
        v = -v
    v = np.maximum(v, 0.0)
    if v.sum() < 1e-14:
        # Como alternativa, se resuelve el sistema lineal imponiendo la normalización.
        A = np.vstack([P.T - np.eye(P.shape[0]), np.ones(P.shape[0])])
        b = np.concatenate([np.zeros(P.shape[0]), [1.0]])
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        v = np.maximum(np.real(v), 0.0)
    return v / v.sum()


def spectral_gap(P: np.ndarray) -> float:
    # La brecha espectral 1-|lambda_2| mide qué tan separado se encuentra el
    # autovalor estacionario lambda_1=1 del segundo autovalor dominante. En
    # cadenas ergódicas, una brecha mayor suele asociarse con convergencia más
    # rápida hacia la distribución estacionaria.
    eigvals = np.linalg.eigvals(P)
    mags = np.sort(np.abs(eigvals))[::-1]
    if len(mags) < 2:
        return 0.0
    return float(max(0.0, 1.0 - mags[1]))


def monte_carlo_mrp(
    p0: np.ndarray,
    P: np.ndarray,
    cost_A: np.ndarray,
    cost_B: np.ndarray,
    T: int,
    runs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo cumulative sentence costs over runs; one cost per sampled state per round."""
    GA = np.zeros(runs)
    GB = np.zeros(runs)
    # Cada realización genera explícitamente una trayectoria aleatoria de la
    # cadena. Esta simulación sirve para contrastar el valor analítico del MRP
    # contra un muestreo directo de las transiciones inducidas por P.
    for r in range(runs):
        s = int(rng.choice(4, p=p0))
        for _ in range(T):
            GA[r] += cost_A[s]
            GB[r] += cost_B[s]
            s = int(rng.choice(4, p=P[s]))
    return GA, GB


# -----------------------------------------------------------------------------
# Simulaciones adaptativas y con memoria extendida
# -----------------------------------------------------------------------------


@dataclass
class AdaptiveThetaPolicy:
    name: str
    theta: float
    phi: float = 0.0
    delta: float = np.pi / 10
    mode: str = "bidirectional"  # cooperation, defection, bidirectional
    history_theta: List[float] = field(default_factory=list)

    def reset(self, theta0: Optional[float] = None) -> None:
        if theta0 is not None:
            self.theta = float(theta0)
        self.history_theta = [float(self.theta)]

    def unitary(self) -> np.ndarray:
        return U_ewl(self.theta, self.phi)

    def update(self, observed_state: str, player: str) -> None:
        # La estrategia adaptativa modifica theta después de observar la acción
        # del oponente. Esto introduce retroalimentación: la operación cuántica
        # empleada en una ronda depende de información adquirida previamente.
        opp = opponent_last_action(observed_state, player)
        if self.mode == "cooperation":
            if opp == "C":
                self.theta = max(0.0, self.theta - self.delta)
        elif self.mode == "defection":
            if opp == "D":
                self.theta = min(np.pi, self.theta + self.delta)
        elif self.mode == "bidirectional":
            # Ante una defección del oponente, theta aumenta hacia el extremo
            # asociado con D; ante cooperación disminuye hacia el extremo C.
            self.theta += self.delta * (1.0 if opp == "D" else -1.0)
            self.theta = float(np.clip(self.theta, 0.0, np.pi))
        else:
            raise ValueError(f"Unknown adaptive mode: {self.mode}")
        self.history_theta.append(float(self.theta))


def simulate_adaptive_pair(
    adaptive_A: AdaptiveThetaPolicy,
    opponent_B: MemoryOnePolicy,
    gamma: float,
    T: int,
    rng: np.random.Generator,
    assumed_initial_state: str = "CC",
) -> Dict[str, np.ndarray]:
    """Direct history-dependent simulation: A adaptive, B memory-one."""
    adaptive_A.reset()
    states: List[int] = []
    costs_A: List[float] = []
    costs_B: List[float] = []
    prev_state = assumed_initial_state

    for _ in range(T):
        # A utiliza el valor actual de su parámetro adaptativo theta_t, mientras
        # que B selecciona una estrategia de memoria uno a partir del resultado
        # observado en la ronda anterior.
        UA = adaptive_A.unitary()
        UB = opponent_B.unitary(prev_state, "B")

        # El circuito cuántico determina la distribución del siguiente resultado.
        p = ewl_probabilities(UA, UB, gamma)
        idx = int(rng.choice(4, p=p))
        state = STATE_LABELS[idx]
        states.append(idx)
        costs_A.append(SENTENCE_A[idx])
        costs_B.append(SENTENCE_B[idx])
        adaptive_A.update(state, "A")
        prev_state = state

    return {
        "states": np.array(states, dtype=int),
        "cost_A": np.array(costs_A, dtype=float),
        "cost_B": np.array(costs_B, dtype=float),
        "theta_A": np.array(adaptive_A.history_theta, dtype=float),
    }


def tf2t_next_action(history: Sequence[str], player: str) -> str:
    """
    Estrategia Tit for Two Tats (TF2T): coopera salvo que el oponente haya
    desertado en las dos rondas inmediatamente anteriores.

    Esta dependencia de dos resultados previos es precisamente la que impide
    describir la dinámica mediante una cadena de Markov sobre solo cuatro
    estados {CC, CD, DC, DD}.
    """
    if len(history) < 2:
        return "C"
    opp_last_two = [opponent_last_action(s, player) for s in history[-2:]]
    return "D" if opp_last_two == ["D", "D"] else "C"


def next_distribution_tf2t_vs_allc(
    history: Sequence[str], gamma: float, tf2t_player: str = "A"
) -> np.ndarray:
    action = tf2t_next_action(history, tf2t_player)
    U_tf2t = U_ewl(0.0 if action == "C" else np.pi, 0.0)
    U_allc = U_ewl(0.0, 0.0)
    if tf2t_player == "A":
        return ewl_probabilities(U_tf2t, U_allc, gamma)
    return ewl_probabilities(U_allc, U_tf2t, gamma)


def augmented_tf2t_transition_matrix(gamma: float) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    Construye una representación de Markov aumentada de 16 estados para
    TF2T(A) contra ALLC(B).

    Como TF2T necesita recordar dos rondas, se redefine el estado como

        X_t = (s_{t-1}, s_t).

    Existen 4 x 4 = 16 estados posibles. La transición adopta la forma
    (a,b) -> (b,c), donde c se genera mediante las probabilidades del circuito
    cuántico. Al incluir la memoria dentro del estado, la propiedad de Markov
    vuelve a ser válida en este espacio aumentado.
    """
    aug_states = [(a, b) for a in STATE_LABELS for b in STATE_LABELS]
    idx = {x: i for i, x in enumerate(aug_states)}
    P16 = np.zeros((16, 16), dtype=float)
    U_allc = U_ewl(0.0, 0.0)

    for x in aug_states:
        history = [x[0], x[1]]
        a = tf2t_next_action(history, "A")
        UA = U_ewl(0.0 if a == "C" else np.pi, 0.0)
        probs = ewl_probabilities(UA, U_allc, gamma)
        i = idx[x]
        for k, c in enumerate(STATE_LABELS):
            y = (x[1], c)
            P16[i, idx[y]] += probs[k]
    return aug_states, P16


# -----------------------------------------------------------------------------
# Validación opcional con Qiskit
# -----------------------------------------------------------------------------


def validate_with_qiskit(gamma: float, theta_A: float, phi_A: float,
                         theta_B: float, phi_B: float) -> Dict[str, object]:
    """Valida el vector de estado exacto de NumPy frente a un circuito de Qiskit, si está disponible."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, Operator
    except Exception as exc:
        return {"available": False, "message": f"Qiskit unavailable: {exc}"}

    UA = U_ewl(theta_A, phi_A)
    UB = U_ewl(theta_B, phi_B)
    J = J_ewl(gamma)

    qc = QuantumCircuit(2)
    # En Qiskit, el orden tensorial de Operator para ambos qubits es q1 ⊗ q0.
    # Para hacerlo compatible con nuestra base |A B>, donde A es el primer qubit o el más significativo,
    # se aplican directamente los operadores completos de 4x4 usando el mismo orden matricial de la base.
    qc.unitary(Operator(J), [0, 1], label="J")
    qc.unitary(Operator(np.kron(UA, UB)), [0, 1], label="UAxUB")
    qc.unitary(Operator(J.conj().T), [0, 1], label="Jdg")

    sv_q = np.asarray(Statevector.from_instruction(qc).data)
    sv_np = ewl_statevector(UA, UB, gamma)

    # Global phase alignment.
    overlap = np.vdot(sv_np, sv_q)
    if abs(overlap) > 1e-14:
        sv_q = sv_q * np.exp(-1j * np.angle(overlap))

    return {
        "available": True,
        "max_statevector_error": float(np.max(np.abs(sv_np - sv_q))),
        "numpy_probabilities": np.abs(sv_np) ** 2,
        "qiskit_probabilities": np.abs(sv_q) ** 2,
    }


# -----------------------------------------------------------------------------
# Funciones auxiliares para las gráficas
# -----------------------------------------------------------------------------


def save_heatmap(matrix: np.ndarray, xlabels: Sequence[str], ylabels: Sequence[str],
                 title: str, path: Path, fmt: str = ".3f") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(xlabels)), labels=xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), labels=ylabels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Probability")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_line_plot(x, ys: Dict[str, np.ndarray], xlabel: str, ylabel: str,
                   title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    for label, y in ys.items():
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if len(ys) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Experimentos correspondientes a las subsecciones de la tesis
# -----------------------------------------------------------------------------


def run_experiments(output: Path, shots: int, seed: int, qiskit_validation: bool) -> None:
    figdir = output / "figures"
    tabdir = output / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    summary: List[str] = []
    summary.append("Chapter 7 computational results — sentence-cost convention\n")
    summary.append(f"Seed: {seed}\nMonte Carlo runs: {shots}\n")

    # ------------------------------------------------------------------
    # 7.3: Matriz de transición del MRP, dinámica analítica y Monte Carlo
    # Se usa QTFT vs Q1 para que gamma modifique efectivamente el núcleo estocástico.
    # ------------------------------------------------------------------
    A = policy_quantum_tft(name="QTFT")
    B = policy_stationary_q(np.pi/2, np.pi/8, name="Q1_phi")
    gamma_ref = np.pi / 4
    P = transition_matrix(A, B, gamma_ref)
    p0 = initial_distribution(A, B, gamma_ref)

    pd.DataFrame(P, index=STATE_LABELS, columns=STATE_LABELS).to_csv(
        tabdir / "transition_matrix_QTFT_vs_Q1_gamma_pi4.csv"
    )
    save_heatmap(
        P, STATE_LABELS, STATE_LABELS,
        r"Matriz de transición inducida: QTFT vs Q1, $\gamma=\pi/4$",
        figdir / "fig_7_3_transition_matrix.png",
    )

    T = 50
    distributions = propagate(p0, P, T)
    save_line_plot(
        np.arange(T + 1),
        {STATE_LABELS[i]: distributions[:, i] for i in range(4)},
        "Ronda", "Probabilidad del estado",
        r"Evolución de $\mathbf{p}_t=\mathbf{p}_0P^t$",
        figdir / "fig_7_3_state_probabilities.png",
    )

    GA_exact = cumulative_expected_cost(p0, P, SENTENCE_A, T)
    GB_exact = cumulative_expected_cost(p0, P, SENTENCE_B, T)
    GA_mc, GB_mc = monte_carlo_mrp(p0, P, SENTENCE_A, SENTENCE_B, T, shots, rng)
    mrp_validation = pd.DataFrame({
        "player": ["A", "B"],
        "analytical_cumulative_years": [GA_exact, GB_exact],
        "mc_mean_years": [GA_mc.mean(), GB_mc.mean()],
        "mc_std_years": [GA_mc.std(ddof=1), GB_mc.std(ddof=1)],
        "absolute_difference_years": [abs(GA_mc.mean()-GA_exact), abs(GB_mc.mean()-GB_exact)],
    })
    mrp_validation.to_csv(tabdir / "mrp_analytical_vs_monte_carlo.csv", index=False)

    # Figura: resultado analítico frente a la media de Monte Carlo y su error estándar.
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    x = np.arange(2)
    analytic = np.array([GA_exact, GB_exact])
    mcmean = np.array([GA_mc.mean(), GB_mc.mean()])
    sem = np.array([GA_mc.std(ddof=1), GB_mc.std(ddof=1)]) / np.sqrt(shots)
    width = 0.35
    ax.bar(x - width/2, analytic, width, label="MRP analítico")
    ax.bar(x + width/2, mcmean, width, yerr=sem, capsize=4, label="Monte Carlo")
    ax.set_xticks(x, ["Jugador A", "Jugador B"])
    ax.set_ylabel(f"Condena acumulada esperada en {T} rondas (años)")
    ax.set_title("Validación MRP: condena acumulada esperada vs muestreo directo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig_7_3_mrp_validation.png", dpi=220)
    plt.close(fig)

    # Barrido de gamma: probabilidades estacionarias, condena y brecha espectral.
    # Barrido desde el régimen clásico/no entrelazado (gamma=0) hasta el
    # entrelazamiento máximo (gamma=pi/2), con 61 puntos para observar cómo el
    # parámetro cuántico modifica la dinámica estocástica de largo plazo.
    gammas = np.linspace(0.0, np.pi/2, 61)
    stat = np.zeros((len(gammas), 4))
    avgA = np.zeros(len(gammas))
    avgB = np.zeros(len(gammas))
    gaps = np.zeros(len(gammas))
    for k, g in enumerate(gammas):
        Pg = transition_matrix(A, B, g)
        pi = stationary_distribution(Pg)
        stat[k] = pi
        avgA[k] = pi @ SENTENCE_A
        avgB[k] = pi @ SENTENCE_B
        gaps[k] = spectral_gap(Pg)

    pd.DataFrame({
        "gamma": gammas,
        "pi_CC": stat[:, 0], "pi_CD": stat[:, 1],
        "pi_DC": stat[:, 2], "pi_DD": stat[:, 3],
        "sentence_A_stationary_years": avgA,
        "sentence_B_stationary_years": avgB,
        "spectral_gap": gaps,
    }).to_csv(tabdir / "stationary_sweep_gamma.csv", index=False)

    save_line_plot(
        gammas,
        {STATE_LABELS[i]: stat[:, i] for i in range(4)},
        r"Entrelazamiento $\gamma$", "Probabilidad estacionaria",
        "Distribución estacionaria inducida por el canal EWL",
        figdir / "fig_7_3_stationary_distribution_vs_gamma.png",
    )
    save_line_plot(
        gammas,
        {"Jugador A": avgA, "Jugador B": avgB},
        r"Entrelazamiento $\gamma$", "Condena estacionaria esperada por ronda (años)",
        "Condena esperada de largo plazo en función del entrelazamiento",
        figdir / "fig_7_3_stationary_sentence_vs_gamma.png",
    )
    save_line_plot(
        gammas, {"Spectral gap": gaps},
        r"Entanglement $\gamma$", r"$1-|\lambda_2|$",
        "Brecha espectral de la matriz de transición inducida",
        figdir / "fig_7_3_spectral_gap_vs_gamma.png",
    )

    summary.append("7.3 MRP validation\n")
    summary.append(mrp_validation.to_string(index=False) + "\n")
    summary.append(f"Transition row sums min/max: {P.sum(axis=1).min():.12f}, {P.sum(axis=1).max():.12f}\n")
    summary.append(f"Stationary distribution at gamma=pi/4: {stationary_distribution(P)}\n")

    # ------------------------------------------------------------------
    # 7.4: Comparación de estrategias y trayectorias adaptativas
    # ------------------------------------------------------------------
    policy_pairs = [
        (policy_tft(), policy_allc(), "TFT vs ALLC"),
        (policy_tft(), policy_alld(), "TFT vs ALLD"),
        (policy_quantum_tft(), policy_stationary_q(np.pi/2, np.pi/8, "Q1_phi"), "QTFT vs Q1"),
        (policy_stationary_q(np.pi/4, np.pi/6, "Q_low"), policy_stationary_q(3*np.pi/4, np.pi/6, "Q_high"), "Q_low vs Q_high"),
    ]
    sentence_rows = []
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    for pa, pb, label in policy_pairs:
        vals = []
        for g in gammas:
            Pg = transition_matrix(pa, pb, g)
            pi = stationary_distribution(Pg)
            vals.append(pi @ SENTENCE_A)
            sentence_rows.append({"pair": label, "gamma": g, "stationary_sentence_A_years": vals[-1]})
        ax.plot(gammas, vals, label=label)
    ax.set_xlabel(r"Entrelazamiento $\gamma$")
    ax.set_ylabel("Condena estacionaria esperada de A por ronda (años)")
    ax.set_title("Condena esperada a través del régimen de entrelazamiento")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig_7_4_strategy_sentence_vs_gamma.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(sentence_rows).to_csv(tabdir / "strategy_sentence_vs_gamma.csv", index=False)

    # Diagnóstico de interferencia: evolución coherente frente a desfasada para estrategias no clásicas seleccionadas.
    int_rows = []
    coherent_costs = []
    decoh_costs = []
    UA = U_ewl(np.pi/3, np.pi/5)
    UB = U_ewl(2*np.pi/3, np.pi/7)
    for g in gammas:
        pc = ewl_probabilities(UA, UB, g)
        pdh = decohered_intermediate_probabilities(UA, UB, g)
        rc = pc @ SENTENCE_A
        rd = pdh @ SENTENCE_A
        coherent_costs.append(rc)
        decoh_costs.append(rd)
        int_rows.append({
            "gamma": g,
            "coherent_sentence_A_years": rc,
            "dephased_sentence_A_years": rd,
            "l1_probability_difference": np.abs(pc-pdh).sum(),
        })
    save_line_plot(
        gammas,
        {"EWL coherente": np.array(coherent_costs), "Referencia desfasada": np.array(decoh_costs)},
        r"Entrelazamiento $\gamma$", "Condena esperada de A (años)",
        "Diagnóstico de interferencia: evolución coherente vs desfasada",
        figdir / "fig_7_4_interference_diagnostic.png",
    )
    pd.DataFrame(int_rows).to_csv(tabdir / "interference_diagnostic.csv", index=False)

    # Trayectorias adaptativas contra TFT; se fija la semilla para garantizar reproducibilidad.
    adaptive_specs = [
        ("Q2 cooperation-driven", np.pi, "cooperation"),
        ("Q3 defection-driven", 0.0, "defection"),
        ("Q4 bidirectional", np.pi/2, "bidirectional"),
    ]
    T_ad = 80
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    adaptive_summary = []
    for n, theta0, mode in adaptive_specs:
        local_rng = np.random.default_rng(seed + {"Q2 cooperation-driven": 101, "Q3 defection-driven": 202, "Q4 bidirectional": 303}[n])
        ap = AdaptiveThetaPolicy(name=n, theta=theta0, phi=0.0, mode=mode)
        result = simulate_adaptive_pair(ap, policy_tft(), np.pi/4, T_ad, local_rng)
        ax.plot(np.arange(T_ad + 1), result["theta_A"] / np.pi, label=n)
        adaptive_summary.append({
            "strategy": n,
            "theta_initial_over_pi": theta0/np.pi,
            "theta_final_over_pi": result["theta_A"][-1]/np.pi,
            "mean_sentence_A_years": result["cost_A"].mean(),
            "cooperation_fraction_A_outcome": np.mean([STATE_LABELS[i][0] == "C" for i in result["states"]]),
        })
    ax.set_xlabel("Ronda")
    ax.set_ylabel(r"Parámetro adaptativo $\theta_t/\pi$")
    ax.set_title(r"Trayectorias adaptativas en el espacio EWL ($\gamma=\pi/4$)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig_7_4_adaptive_theta_trajectories.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(adaptive_summary).to_csv(tabdir / "adaptive_strategy_summary.csv", index=False)
    summary.append("\n7.4 Adaptive strategies\n")
    summary.append(pd.DataFrame(adaptive_summary).to_string(index=False) + "\n")

    # ------------------------------------------------------------------
    # 7.5: Pérdida explícita de la propiedad de Markov para TF2T en S y representación aumentada de 16 estados
    # ------------------------------------------------------------------
    # Demostración explícita de la pérdida de la propiedad de Markov en el
    # espacio original de cuatro estados. Ambas historias terminan en DD, por
    # lo que tienen el mismo estado actual, pero TF2T toma decisiones distintas:
    # H1=[CC,DD] -> solo existe una defección consecutiva de B -> A coopera.
    # H2=[CD,DD] -> B desertó en las dos últimas rondas -> A deserta.
    # Si las distribuciones siguientes difieren, entonces conocer únicamente
    # s_t=DD no es suficiente para determinar Pr(s_{t+1}|s_t).
    h1 = ["CC", "DD"]
    h2 = ["CD", "DD"]
    p_h1 = next_distribution_tf2t_vs_allc(h1, gamma_ref, "A")
    p_h2 = next_distribution_tf2t_vs_allc(h2, gamma_ref, "A")
    # La distancia de variación total cuantifica cuánto difieren ambas
    # distribuciones. Un valor distinto de cero prueba numéricamente que las
    # probabilidades futuras dependen de la historia y no solo del estado DD.
    tv = 0.5 * np.abs(p_h1 - p_h2).sum()
    markov_break = pd.DataFrame({
        "next_state": STATE_LABELS,
        "P_next_given_history_CC_DD": p_h1,
        "P_next_given_history_CD_DD": p_h2,
    })
    markov_break.to_csv(tabdir / "tf2t_markov_breakdown_same_current_state.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(4)
    width = 0.36
    ax.bar(x-width/2, p_h1, width, label="Historia (CC, DD)")
    ax.bar(x+width/2, p_h2, width, label="Historia (CD, DD)")
    ax.set_xticks(x, STATE_LABELS)
    ax.set_ylabel("Probabilidad del siguiente estado")
    ax.set_title(f"TF2T: mismo estado actual DD, historias diferentes (TV={tv:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig_7_5_markov_breakdown_tf2t.png", dpi=220)
    plt.close(fig)

    aug_states, P16 = augmented_tf2t_transition_matrix(gamma_ref)
    labels16 = [f"{a}|{b}" for a, b in aug_states]
    pd.DataFrame(P16, index=labels16, columns=labels16).to_csv(
        tabdir / "tf2t_augmented_transition_matrix_16.csv"
    )
    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    im = ax.imshow(P16, aspect="auto")
    ax.set_xticks(range(16), labels16, rotation=90, fontsize=7)
    ax.set_yticks(range(16), labels16, fontsize=7)
    ax.set_title(r"Representación de Markov aumentada para TF2T vs ALLC ($|\mathcal{S}_2|=16$)")
    fig.colorbar(im, ax=ax, label="Probabilidad de transición")
    fig.tight_layout()
    fig.savefig(figdir / "fig_7_5_augmented_transition_matrix.png", dpi=220)
    plt.close(fig)

    # Verifica las propiedades de matriz de Markov para P16.
    row_sum_error = float(np.max(np.abs(P16.sum(axis=1) - 1.0)))

    depths = np.arange(0, 7)
    state_counts = np.where(depths == 0, 1, 4 ** depths)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(depths, state_counts, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Profundidad de memoria $m$")
    ax.set_ylabel(r"Tamaño del espacio aumentado $|\mathcal{S}_m|$")
    ax.set_title(r"Crecimiento del espacio para memoria finita: $|\mathcal{S}_m|=4^m$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "fig_7_5_memory_state_space_growth.png", dpi=220)
    plt.close(fig)
    pd.DataFrame({"memory_depth": depths, "state_space_size": state_counts}).to_csv(
        tabdir / "memory_state_space_growth.csv", index=False
    )

    summary.append("\n7.5 Extended memory / non-Markovian test\n")
    summary.append(f"History 1 {h1} -> next distribution {p_h1}\n")
    summary.append(f"History 2 {h2} -> next distribution {p_h2}\n")
    summary.append(f"Total variation distance: {tv:.8f}\n")
    summary.append(f"Augmented 16-state matrix max row-sum error: {row_sum_error:.3e}\n")

    # ------------------------------------------------------------------
    # Validación opcional con Qiskit
    # ------------------------------------------------------------------
    if qiskit_validation:
        qres = validate_with_qiskit(
            gamma=np.pi/4,
            theta_A=np.pi/3, phi_A=np.pi/7,
            theta_B=2*np.pi/3, phi_B=np.pi/8,
        )
        summary.append("\nOptional Qiskit validation\n")
        summary.append(str(qres) + "\n")

    # Manifiesto compacto que indica qué figura respalda cada subsección.
    manifest = pd.DataFrame([
        ("7.3.2", "fig_7_3_transition_matrix.png", "Quantum-induced transition kernel P"),
        ("7.3.2", "fig_7_3_state_probabilities.png", "Evolution p_t = p_0 P^t"),
        ("7.3.4", "fig_7_3_mrp_validation.png", "Analytical finite-horizon MRP vs Monte Carlo"),
        ("7.3.5", "fig_7_3_stationary_distribution_vs_gamma.png", "Stationary distribution versus entanglement"),
        ("7.3.5", "fig_7_3_stationary_sentence_vs_gamma.png", "Long-run expected sentence versus entanglement"),
        ("7.4", "fig_7_4_strategy_sentence_vs_gamma.png", "Classical/quantum strategy comparison using expected sentence"),
        ("7.4.3", "fig_7_4_adaptive_theta_trajectories.png", "Adaptive strategy trajectories"),
        ("7.4", "fig_7_4_interference_diagnostic.png", "Coherent versus dephased EWL diagnostic"),
        ("7.5.2", "fig_7_5_markov_breakdown_tf2t.png", "Same current state, history-dependent next distribution"),
        ("7.5.3", "fig_7_5_augmented_transition_matrix.png", "16-state Markov embedding for memory two"),
        ("7.5.4", "fig_7_5_memory_state_space_growth.png", "Growth |S_m|=4^m"),
    ], columns=["subsection", "figure", "supports"])
    manifest.to_csv(tabdir / "figure_manifest.csv", index=False)

    (output / "summary.txt").write_text("".join(summary), encoding="utf-8")

    print(f"Results written to: {output.resolve()}")
    print(f"Figures: {figdir.resolve()}")
    print(f"Tables : {tabdir.resolve()}")
    print("\nKey validation:")
    print(mrp_validation.to_string(index=False))
    print(f"TF2T non-Markov total variation distance = {tv:.6f}")
    print(f"Augmented P16 max row-sum error = {row_sum_error:.3e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chapter 7 Iterated Quantum Prisoner's Dilemma experiments")
    p.add_argument("--output", type=Path, default=Path("results_ch7"), help="Output directory")
    p.add_argument("--shots", type=int, default=5000, help="Monte Carlo realizations for MRP validation")
    p.add_argument("--seed", type=int, default=RNG_DEFAULT_SEED, help="Random seed")
    p.add_argument("--validate-qiskit", action="store_true", help="Also validate one circuit with Qiskit if installed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args.output, args.shots, args.seed, args.validate_qiskit)
