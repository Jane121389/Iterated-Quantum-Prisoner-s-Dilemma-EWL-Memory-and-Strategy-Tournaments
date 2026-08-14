#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sección 7.7 — Sensibilidad paramétrica de las estrategias TFTQ
================================================================

Este programa ejecuta exclusivamente los experimentos numéricos de la
Sección 7.7 de la tesis:

    "Sensibilidad paramétrica de las estrategias TFTQ"

No contiene verificaciones contra valores escritos en la tesis ni rutinas
del torneo round-robin de la Sección 7.8. Su propósito es generar, desde la
formulación matemática, las distribuciones, matrices de transición, condenas
acumuladas, tablas y figura correspondientes al análisis TFTQ.

La familia TFTQ conserva la lógica de Tit-for-Tat (TFT):

    - primera ronda: cooperación;
    - si el oponente cooperó en la ronda anterior -> usar theta_1;
    - si el oponente delató en la ronda anterior  -> usar theta_2.

La diferencia con TFT clásico es que las respuestas no tienen que estar
restringidas a theta = 0 y theta = pi. En este experimento se consideran:

    (theta_1, theta_2) =
        (0, pi)
        (pi/16, 15pi/16)
        (pi/8, 7pi/8)
        (pi/4, 3pi/4)

La fase del jugador A se fija en

    phi_A = 0,

mientras que para el jugador B se realiza el barrido

    phi_B = k*pi/20,    k = 0,...,9.

CORRECCIÓN IMPORTANTE
---------------------
El entrelazamiento se mantiene MÁXIMO durante TODA la dinámica:

    gamma = pi/2.

Por tanto, gamma = pi/2 se utiliza tanto para:

    1. la distribución inicial v_1;
    2. TODAS las filas de la matriz de transición P.

No se utiliza gamma = 0 para las rondas posteriores.

Convención de estados
---------------------
Orden:

    (CC, CD, DC, DD)

El primer símbolo corresponde al jugador A y el segundo al jugador B.

Convención de condenas
----------------------
Se utilizan años de condena:

    c_A = (1, 5, 0, 3)
    c_B = (1, 0, 5, 3)

Una condena menor representa un mejor desempeño.

Observable principal
--------------------
Para T = 10 rondas se calcula la condena acumulada esperada:

    C_i(T)
      = sum_{t=1}^T E_i(t)
      = v_1^T [I + P + ... + P^(T-1)] c_i.

El cálculo es exacto y utiliza NumPy; no requiere shots ni muestreo Monte Carlo.

Salidas
-------
El programa genera:

    figures/
        fig_7_7_tftq_condena_acumulada_10_rondas.png

    tables/
        tftq_accumulated_sentence_phiB.csv
        tftq_endpoint_summary.csv
        transition_matrix_<config>_<phiB>.csv
        initial_distribution_<config>_<phiB>.csv

    summary.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. CONSTANTES Y CONVENCIONES
# =============================================================================

STATE_LABELS: Tuple[str, ...] = ("CC", "CD", "DC", "DD")
N_STATES = 4

# Condenas en años, usando el orden (CC, CD, DC, DD).
SENTENCE_A = np.array([1.0, 5.0, 0.0, 3.0], dtype=float)
SENTENCE_B = np.array([1.0, 0.0, 5.0, 3.0], dtype=float)

# En todo el experimento de la Sección 7.7 se mantiene entrelazamiento máximo.
GAMMA_MAX = np.pi / 2.0

# Fase fija del jugador A.
PHI_A = 0.0

# Horizonte utilizado en la tesis.
DEFAULT_ROUNDS = 10

# Tolerancia para limpieza numérica.
NUMERICAL_EPS = 1.0e-15


# =============================================================================
# 2. OBJETOS CUÁNTICOS DEL ESQUEMA EWL
# =============================================================================

I4 = np.eye(4, dtype=complex)

D_GATE = np.array(
    [
        [0.0, 1.0],
        [-1.0, 0.0],
    ],
    dtype=complex,
)

K_DD = np.kron(D_GATE, D_GATE)

# |00> equivale al resultado CC antes de aplicar las operaciones.
KET00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)


def U_ewl(theta: float, phi: float = 0.0) -> np.ndarray:
    """
    Operación local restringida U(theta, phi) del esquema EWL.

    U(theta,phi) =
        [[ exp(i phi) cos(theta/2),   sin(theta/2)],
         [-sin(theta/2),              exp(-i phi) cos(theta/2)]]

    Casos clásicos:
        C = U(0,0)
        D = U(pi,0)
    """
    theta = float(theta)
    phi = float(phi)

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    return np.array(
        [
            [np.exp(1j * phi) * c, s],
            [-s, np.exp(-1j * phi) * c],
        ],
        dtype=complex,
    )


def J_ewl(gamma: float) -> np.ndarray:
    """
    Entrelazador EWL:

        J(gamma) = exp[-i gamma/2 (D x D)].

    Como (D x D)^2 = I, la exponencial se evalúa analíticamente.
    """
    gamma = float(gamma)

    return (
        np.cos(gamma / 2.0) * I4
        - 1j * np.sin(gamma / 2.0) * K_DD
    )


def ewl_probabilities(
    theta_a: float,
    phi_a: float,
    theta_b: float,
    phi_b: float,
    gamma: float = GAMMA_MAX,
) -> np.ndarray:
    """
    Calcula exactamente las probabilidades EWL en el orden:

        (CC, CD, DC, DD).

    El circuito implementado es:

        |psi_f>
          = J(gamma)^†
            [U_A(theta_A,phi_A) x U_B(theta_B,phi_B)]
            J(gamma)|00>.
    """
    J = J_ewl(gamma)

    psi = J @ KET00
    psi = np.kron(
        U_ewl(theta_a, phi_a),
        U_ewl(theta_b, phi_b),
    ) @ psi
    psi = J.conj().T @ psi

    probabilities = np.abs(psi) ** 2
    probabilities[np.abs(probabilities) < NUMERICAL_EPS] = 0.0
    probabilities = np.maximum(probabilities, 0.0)

    total = float(probabilities.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError("No fue posible normalizar las probabilidades.")

    probabilities /= total

    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        raise FloatingPointError("Las probabilidades EWL no suman uno.")

    return probabilities


# =============================================================================
# 3. DEFINICIÓN DE LA FAMILIA TFTQ
# =============================================================================

@dataclass(frozen=True)
class TFTQConfiguration:
    """
    Una parametrización de TFTQ.

    theta_cooperate:
        theta_1, utilizado si el oponente cooperó en la ronda anterior.

    theta_defect:
        theta_2, utilizado si el oponente delató en la ronda anterior.
    """

    theta_cooperate: float
    theta_defect: float
    label: str


TFTQ_CONFIGURATIONS: Tuple[TFTQConfiguration, ...] = (
    TFTQConfiguration(
        0.0,
        np.pi,
        r"$(0,\pi)$",
    ),
    TFTQConfiguration(
        np.pi / 16.0,
        15.0 * np.pi / 16.0,
        r"$(\pi/16,15\pi/16)$",
    ),
    TFTQConfiguration(
        np.pi / 8.0,
        7.0 * np.pi / 8.0,
        r"$(\pi/8,7\pi/8)$",
    ),
    TFTQConfiguration(
        np.pi / 4.0,
        3.0 * np.pi / 4.0,
        r"$(\pi/4,3\pi/4)$",
    ),
)


def opponent_action(previous_state: str, player: str) -> str:
    """
    Obtiene la acción previa del oponente.

    Ejemplo:
        previous_state = "CD"

        jugador A observa que B hizo D;
        jugador B observa que A hizo C.
    """
    if previous_state not in STATE_LABELS:
        raise ValueError(f"Estado inválido: {previous_state!r}")

    if player == "A":
        return previous_state[1]

    if player == "B":
        return previous_state[0]

    raise ValueError("player debe ser 'A' o 'B'.")


def tftq_response_theta(
    previous_state: str,
    player: str,
    configuration: TFTQConfiguration,
) -> float:
    """
    Regla de memoria uno de TFTQ.

    cooperación previa del oponente -> theta_1
    defección previa del oponente   -> theta_2
    """
    action = opponent_action(previous_state, player)

    if action == "C":
        return float(configuration.theta_cooperate)

    return float(configuration.theta_defect)


# =============================================================================
# 4. DISTRIBUCIÓN INICIAL Y MATRIZ DE TRANSICIÓN
# =============================================================================

def initial_distribution(
    phi_b: float,
    gamma: float = GAMMA_MAX,
) -> np.ndarray:
    """
    Calcula v_1.

    TFTQ, como TFT, comienza cooperando. Por tanto, en la primera ronda:

        A -> U(0, phi_A)
        B -> U(0, phi_B).

    En la Sección 7.7:
        phi_A = 0
        gamma = pi/2.
    """
    # En la primera ronda todavía no existe una acción previa del oponente.
    # Siguiendo la regla TFT, ambos jugadores comienzan cooperando. La única
    # diferencia es que B conserva la fase phi_B correspondiente al punto del
    # barrido que se está evaluando.
    return ewl_probabilities(
        theta_a=0.0,
        phi_a=PHI_A,
        theta_b=0.0,
        phi_b=phi_b,
        gamma=gamma,
    )


def transition_matrix(
    configuration: TFTQConfiguration,
    phi_b: float,
    gamma: float = GAMMA_MAX,
) -> np.ndarray:
    """
    Construye la matriz P de dimensión 4 x 4.

    La fila asociada al estado s representa:

        P(s_{t+1} | s_t=s).

    Cada jugador selecciona theta_1 o theta_2 dependiendo exclusivamente de
    la acción del oponente contenida en el estado anterior.

    IMPORTANTE:
        gamma se utiliza en TODAS las filas de P.

    Para reproducir la tesis:
        gamma = pi/2.
    """
    P = np.zeros((N_STATES, N_STATES), dtype=float)

    # Cada fila representa un estado observado en la ronda anterior.
    # La regla TFTQ consulta ese estado para identificar la acción previa
    # del oponente y seleccionar theta_1 o theta_2 para la ronda siguiente.
    for row, previous_state in enumerate(STATE_LABELS):
        theta_a = tftq_response_theta(
            previous_state,
            "A",
            configuration,
        )

        theta_b = tftq_response_theta(
            previous_state,
            "B",
            configuration,
        )

        P[row, :] = ewl_probabilities(
            theta_a=theta_a,
            phi_a=PHI_A,
            theta_b=theta_b,
            phi_b=phi_b,
            gamma=gamma,
        )

    # Verificación de matriz estocástica.
    if np.any(P < -1e-12):
        raise FloatingPointError("P contiene probabilidades negativas.")

    P = np.maximum(P, 0.0)
    P /= P.sum(axis=1, keepdims=True)

    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-12):
        raise FloatingPointError("Alguna fila de P no suma uno.")

    return P


# =============================================================================
# 5. CONDENA ACUMULADA ESPERADA
# =============================================================================

def round_distributions(
    v1: np.ndarray,
    P: np.ndarray,
    rounds: int,
) -> np.ndarray:
    """
    Devuelve las distribuciones de las rondas 1,...,T.

    La ronda 1 utiliza v_1.
    La ronda 2 utiliza v_1 P.
    ...
    La ronda T utiliza v_1 P^(T-1).
    """
    if rounds <= 0:
        raise ValueError("rounds debe ser positivo.")

    distributions = np.zeros((rounds, N_STATES), dtype=float)
    distributions[0] = v1

    for t in range(1, rounds):
        distributions[t] = distributions[t - 1] @ P

    return distributions


def cumulative_sentence(
    v1: np.ndarray,
    P: np.ndarray,
    sentence_vector: np.ndarray,
    rounds: int = DEFAULT_ROUNDS,
) -> float:
    """
    Evalúa

        C_i(T)
          = sum_{t=1}^T v_1^T P^(t-1) c_i.

    No equivale a multiplicar por T la condena de la última ronda.
    """
    distributions = round_distributions(v1, P, rounds)
    per_round = distributions @ sentence_vector

    return float(per_round.sum())


def evaluate_configuration(
    configuration: TFTQConfiguration,
    phi_b: float,
    rounds: int = DEFAULT_ROUNDS,
    gamma: float = GAMMA_MAX,
) -> Dict[str, object]:
    """
    Evalúa una combinación (theta_1, theta_2, phi_B).

    Retorna la distribución inicial, P y las condenas acumuladas de A y B.
    """
    v1 = initial_distribution(
        phi_b=phi_b,
        gamma=gamma,
    )

    P = transition_matrix(
        configuration=configuration,
        phi_b=phi_b,
        gamma=gamma,
    )

    # `distributions[t]` contiene la distribución de probabilidad sobre
    # (CC, CD, DC, DD) en la ronda t+1. Por ejemplo:
    #
    #   distributions[0] = v1
    #   distributions[1] = v1 P
    #   distributions[2] = v1 P^2
    #
    # Esto implementa directamente la evolución descrita en la tesis.
    distributions = round_distributions(
        v1=v1,
        P=P,
        rounds=rounds,
    )

    # La condena esperada de cada ronda se obtiene como el producto de la
    # distribución de estados por el vector de condenas correspondiente.
    #
    # Para A:
    #   E_A(t) = p_t · c_A
    #
    # Para B:
    #   E_B(t) = p_t · c_B
    sentence_per_round_A = distributions @ SENTENCE_A
    sentence_per_round_B = distributions @ SENTENCE_B

    return {
        "v1": v1,
        "P": P,
        "distributions": distributions,
        "sentence_per_round_A": sentence_per_round_A,
        "sentence_per_round_B": sentence_per_round_B,
        "cumulative_A": float(sentence_per_round_A.sum()),
        "cumulative_B": float(sentence_per_round_B.sum()),
    }


# =============================================================================
# 6. EXPERIMENTO DE LA SECCIÓN 7.7
# =============================================================================

def sanitize_label(configuration: TFTQConfiguration) -> str:
    """Nombre corto seguro para archivos."""
    t1 = configuration.theta_cooperate / np.pi
    t2 = configuration.theta_defect / np.pi
    return f"theta1_{t1:.5f}_theta2_{t2:.5f}".replace(".", "p")


def run_experiment(
    output: Path,
    rounds: int = DEFAULT_ROUNDS,
) -> None:
    """
    Ejecuta exactamente el barrido de la Sección 7.7.

    phi_B = k*pi/20, k=0,...,9
    gamma = pi/2 durante TODAS las rondas.
    """
    if rounds <= 0:
        raise ValueError("rounds debe ser positivo.")

    figures_dir = output / "figures"
    tables_dir = output / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Barrido usado en la Sección 7.7:
    #
    #   phi_B = k*pi/20,  k = 0,...,9
    #
    # Esto produce diez puntos entre 0 y 9pi/20.
    phi_values = np.arange(10, dtype=float) * np.pi / 20.0

    rows: List[dict] = []
    endpoints: List[dict] = []

    # Figura principal.
    fig, ax = plt.subplots(figsize=(10.5, 6.3))

    for config_index, config in enumerate(TFTQ_CONFIGURATIONS):
        cumulative_A: List[float] = []
        cumulative_B: List[float] = []

        for k, phi_b in enumerate(phi_values):
            # Todas las rondas se calculan con gamma = pi/2.
            # `evaluate_configuration()` utiliza este mismo gamma tanto para
            # v1 como para la matriz P.
            result = evaluate_configuration(
                configuration=config,
                phi_b=float(phi_b),
                rounds=rounds,
                gamma=GAMMA_MAX,
            )

            CA = float(result["cumulative_A"])
            CB = float(result["cumulative_B"])

            cumulative_A.append(CA)
            cumulative_B.append(CB)

            rows.append(
                {
                    "configuration": config.label,
                    "theta1": config.theta_cooperate,
                    "theta2": config.theta_defect,
                    "theta1_over_pi": config.theta_cooperate / np.pi,
                    "theta2_over_pi": config.theta_defect / np.pi,
                    "phiB_index_k": k,
                    "phiB": phi_b,
                    "phiB_over_pi": phi_b / np.pi,
                    "gamma": GAMMA_MAX,
                    "gamma_over_pi": GAMMA_MAX / np.pi,
                    "rounds": rounds,
                    "cumulative_sentence_A_years": CA,
                    "cumulative_sentence_B_years": CB,
                    "difference_A_minus_B_years": CA - CB,
                }
            )

            # Para facilitar auditoría, guardamos v1 y P en los dos extremos
            # del barrido: phi_B=0 y phi_B=9pi/20.
            if k in {0, 9}:
                tag = sanitize_label(config)
                phi_tag = f"phiB_{k}pi20"

                pd.DataFrame(
                    [result["v1"]],
                    columns=STATE_LABELS,
                ).to_csv(
                    tables_dir
                    / f"initial_distribution_{tag}_{phi_tag}.csv",
                    index=False,
                )

                pd.DataFrame(
                    result["P"],
                    index=STATE_LABELS,
                    columns=STATE_LABELS,
                ).to_csv(
                    tables_dir
                    / f"transition_matrix_{tag}_{phi_tag}.csv"
                )

        endpoints.append(
            {
                "configuration": config.label,
                "theta1_over_pi": config.theta_cooperate / np.pi,
                "theta2_over_pi": config.theta_defect / np.pi,
                "A_phiB_0": cumulative_A[0],
                "B_phiB_0": cumulative_B[0],
                "A_phiB_9pi20": cumulative_A[-1],
                "B_phiB_9pi20": cumulative_B[-1],
                "Delta_phiB_9pi20": cumulative_A[-1] - cumulative_B[-1],
            }
        )

        # Misma tonalidad para A y B de una configuración:
        # línea continua = jugador A
        # línea discontinua = jugador B.
        line_A, = ax.plot(
            phi_values / np.pi,
            cumulative_A,
            linestyle="-",
            label=f"A  \\Theta={config.label}",
        )

        ax.plot(
            phi_values / np.pi,
            cumulative_B,
            linestyle="--",
            color=line_A.get_color(),
            label=f"B  \\Theta={config.label}",
        )

    ax.set_xlabel(r"$\phi_B/\pi$")
    ax.set_ylabel(
        f"Condena acumulada esperada en {rounds} rondas (años)"
    )
    ax.set_title(
        r"Sensibilidad paramétrica de TFTQ con $\gamma=\pi/2$"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(
        figures_dir
        / "fig_7_7_tftq_condena_acumulada_10_rondas.png",
        dpi=240,
    )
    plt.close(fig)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(
        tables_dir / "tftq_accumulated_sentence_phiB.csv",
        index=False,
    )

    endpoints_df = pd.DataFrame(endpoints)
    endpoints_df.to_csv(
        tables_dir / "tftq_endpoint_summary.csv",
        index=False,
    )
    # Resumen textual de los resultados generados por el experimento.
    summary = (
        "Section 7.7 — TFTQ parametric sensitivity\n"
        "================================================\n"
        f"Rounds: {rounds}\n"
        f"phi_A: {PHI_A}\n"
        f"gamma: {GAMMA_MAX} = pi/2\n"
        "gamma is kept at pi/2 for v1 and every transition probability.\n\n"
        "Endpoint results:\n"
        + endpoints_df.to_string(index=False)
        + "\n"
    )

    (output / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)


# =============================================================================
# 7. INTERFAZ DE LÍNEA DE COMANDOS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce la sensibilidad paramétrica TFTQ de la Sección 7.7."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_tftq"),
        help="Directorio de salida.",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Número de rondas; la tesis utiliza 10.",
    )

    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds debe ser mayor que cero.")

    return args


if __name__ == "__main__":
    args = parse_args()

    run_experiment(
        output=args.output,
        rounds=args.rounds,
    )
