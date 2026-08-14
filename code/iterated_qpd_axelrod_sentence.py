#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torneo tipo Axelrod para el Dilema del Prisionero Cuántico Iterativo
=====================================================================

Este archivo es el complemento numérico del Capítulo 7 dedicado al torneo
round-robin clásico-cuántico. Implementa una población de 12 estrategias:

    Clásicas:
        TFT, ALLD, ALLC, Random, Grudger, TF2T

    Cuánticas:
        Q1_Tibio, Q2_Bondadoso, Q3_Traicionero,
        Q4_Reciproco, Q5_TibioReciproco, Q9_Switch

Cada par ordenado de estrategias (S_i, S_j) se enfrenta durante un número
finito de rondas. La estrategia situada en la fila se interpreta como el
jugador A (estrategia evaluada) y la situada en la columna como el jugador B
(oponente).

La dinámica de cada ronda utiliza el protocolo de
Eisert--Wilkens--Lewenstein (EWL):

    |00> -- J(gamma) -- (U_A x U_B) -- J(gamma)^† -- medición

La medición produce uno de los cuatro resultados:

    CC, CD, DC, DD

donde el primer símbolo corresponde al jugador A y el segundo al jugador B.

Convención de condenas
----------------------
En vez de maximizar recompensas, este programa minimiza años de condena.

Orden de estados: (CC, CD, DC, DD)

    c_A = (1, 5, 0, 3)
    c_B = (1, 0, 5, 3)

Por tanto, una condena menor representa un mejor desempeño estratégico.

Configuración principal
-----------------------
Por defecto:

    - 200 rondas por enfrentamiento
    - 5 repeticiones por par ordenado
    - gamma = 0, pi/4, pi/2 para los tres regímenes principales
    - barrido de gamma en 11 puntos entre 0 y pi/2
    - para el barrido exploratorio:
          min(rounds, 100) rondas
          min(repetitions, 3) repeticiones

Salidas
-------
El programa genera:

    results_tournament/
        figures/
            mapas de condena
            rankings
            comparación global
            curvas contra gamma
            frecuencias CC, CD, DC, DD
            comparación de grupos y clases de interacción

        tables/
            matrices de condena
            matrices de frecuencias
            rankings
            promedios por grupo
            barrido de gamma
            estrategia dominante por gamma

        summary.txt

Ejemplos
--------
    python iterated_qpd_axelrod_sentence_comentado_depurado.py

    python iterated_qpd_axelrod_sentence_comentado_depurado.py \
        --output results_tournament \
        --rounds 200 \
        --repetitions 5 \
        --seed 20260809

Notas de reproducibilidad
-------------------------
1. Cada enfrentamiento utiliza una semilla determinista derivada de:
       seed base + índice de A + índice de B + repetición + gamma.
2. Las estrategias se clonan y reinician antes de cada enfrentamiento.
3. Las probabilidades EWL se calculan exactamente con NumPy; el muestreo
   aparece únicamente al seleccionar el resultado de cada ronda.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. CONSTANTES DEL MODELO
# =============================================================================

# Orden canónico utilizado en TODO el programa.
#
# Es fundamental no cambiar este orden sin modificar también:
#   - SENTENCE_A
#   - SENTENCE_B
#   - las matrices/CSV
#   - la interpretación de las frecuencias.
STATE_LABELS: Tuple[str, ...] = ("CC", "CD", "DC", "DD")

# Número de estados observables del Dilema del Prisionero de dos jugadores.
N_STATES = len(STATE_LABELS)

# Convención de condenas, en años.
#
# Jugador A:
#   CC -> 1, CD -> 5, DC -> 0, DD -> 3
#
# Jugador B:
#   CC -> 1, CD -> 0, DC -> 5, DD -> 3
SENTENCE_A = np.array([1.0, 5.0, 0.0, 3.0], dtype=float)
SENTENCE_B = np.array([1.0, 0.0, 5.0, 3.0], dtype=float)

# Semilla utilizada por defecto para que el torneo sea reproducible.
DEFAULT_SEED = 20260809

# Tolerancia usada únicamente para limpieza numérica.
NUMERICAL_EPS = 1.0e-15


# =============================================================================
# 2. OBJETOS CUÁNTICOS BÁSICOS DEL ESQUEMA EWL
# =============================================================================

# Identidad de dimensión 4 correspondiente al sistema de dos qubits.
I4 = np.eye(4, dtype=complex)

# Operador D utilizado en el entrelazador EWL de esta implementación.
#
#     D = [[ 0,  1],
#          [-1,  0]]
#
# Cumple D^2 = -I y, por tanto,
#
#     (D x D)^2 = I_4.
#
# Esta propiedad permite escribir J(gamma) de forma analítica sin usar
# scipy.linalg.expm.
D_GATE = np.array(
    [
        [0.0, 1.0],
        [-1.0, 0.0],
    ],
    dtype=complex,
)

# Producto tensorial D x D.
K_DD = np.kron(D_GATE, D_GATE)

# Estado inicial |00>, en el orden de base:
#
#     |00>, |01>, |10>, |11>
#
# que se interpreta como:
#
#     CC, CD, DC, DD.
KET00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)


def U_ewl(theta: float, phi: float = 0.0) -> np.ndarray:
    """
    Construye la estrategia local restringida U(theta, phi) del esquema EWL.

    Parámetros
    ----------
    theta:
        Ángulo polar del espacio estratégico, normalmente en [0, pi].

        Casos clásicos:
            theta = 0   -> cooperación, si phi = 0
            theta = pi  -> defección

    phi:
        Fase estratégica, normalmente en [0, pi/2].

    Retorna
    -------
    numpy.ndarray
        Matriz unitaria compleja 2 x 2:

            [ exp(i phi) cos(theta/2)      sin(theta/2)          ]
            [ -sin(theta/2)                exp(-i phi) cos(theta/2) ]

    Notas
    -----
    La función no restringe automáticamente theta o phi para permitir
    experimentos controlados. Las estrategias definidas en este archivo sí
    utilizan parámetros dentro de los intervalos previstos.
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
    Construye el entrelazador EWL J(gamma).

    Se utiliza

        J(gamma) = exp[-i gamma/2 (D x D)].

    Como (D x D)^2 = I, la exponencial puede evaluarse exactamente como

        J(gamma)
        = cos(gamma/2) I
          - i sin(gamma/2) (D x D).

    Convención
    ----------
    gamma = 0       -> sin entrelazamiento
    gamma = pi/2    -> entrelazamiento máximo en esta parametrización
    """
    gamma = float(gamma)
    return (
        np.cos(gamma / 2.0) * I4
        - 1j * np.sin(gamma / 2.0) * K_DD
    )


# Caché de probabilidades EWL.
#
# El torneo evalúa repetidamente exactamente las mismas combinaciones de
# (theta_A, phi_A, theta_B, phi_B, gamma). Guardarlas evita repetir productos
# matriciales innecesarios.
_PROB_CACHE: Dict[Tuple[float, ...], np.ndarray] = {}


def ewl_probabilities(
    theta_a: float,
    phi_a: float,
    theta_b: float,
    phi_b: float,
    gamma: float,
) -> np.ndarray:
    """
    Calcula las probabilidades exactas del circuito EWL.

    El estado final es

        |psi_f>
        = J(gamma)^† [U_A x U_B] J(gamma) |00>.

    La regla de Born produce:

        p = (p_CC, p_CD, p_DC, p_DD).

    La salida se normaliza explícitamente para eliminar errores de redondeo.

    La caché utiliza parámetros redondeados a 12 decimales. Esta precisión es
    mucho mayor que la necesaria para los barridos usados aquí y evita que
    pequeñas diferencias binarias en flotantes impidan reutilizar resultados.
    """
    key = tuple(
        round(float(x), 12)
        for x in (theta_a, phi_a, theta_b, phi_b, gamma)
    )

    if key in _PROB_CACHE:
        # Se devuelve una copia para impedir que una modificación accidental
        # por parte del llamador altere el valor almacenado en la caché.
        return _PROB_CACHE[key].copy()

    J = J_ewl(gamma)

    # 1) Preparación entrelazada.
    psi = J @ KET00

    # 2) Operaciones locales de los jugadores.
    UAB = np.kron(
        U_ewl(theta_a, phi_a),
        U_ewl(theta_b, phi_b),
    )
    psi = UAB @ psi

    # 3) Desentrelazamiento.
    psi = J.conj().T @ psi

    # 4) Regla de Born.
    probabilities = np.abs(psi) ** 2

    # Limpieza de residuos numéricos extremadamente pequeños.
    probabilities[np.abs(probabilities) < NUMERICAL_EPS] = 0.0
    probabilities = np.maximum(probabilities, 0.0)

    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError(
            "La distribución EWL no puede normalizarse."
        )

    probabilities = probabilities / total

    # Verificación defensiva: debe ser una distribución de probabilidad.
    if not np.all(np.isfinite(probabilities)):
        raise FloatingPointError("EWL produjo probabilidades no finitas.")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        raise FloatingPointError("Las probabilidades EWL no suman uno.")

    _PROB_CACHE[key] = probabilities.copy()
    return probabilities


# =============================================================================
# 3. FUNCIONES AUXILIARES PARA INTERPRETAR LOS ESTADOS
# =============================================================================

def _validate_player(player: str) -> None:
    """Verifica que el identificador del jugador sea 'A' o 'B'."""
    if player not in {"A", "B"}:
        raise ValueError("player debe ser 'A' o 'B'.")


def _validate_state(state: str) -> None:
    """Verifica que el estado pertenezca a {CC, CD, DC, DD}."""
    if state not in STATE_LABELS:
        raise ValueError(
            f"Estado inválido {state!r}; se esperaba uno de {STATE_LABELS}."
        )


def opponent_action(state: str, player: str) -> str:
    """
    Extrae del estado conjunto la última acción del oponente.

    Ejemplo
    -------
    state = "CD"

    Para A:
        la acción del oponente B es "D".

    Para B:
        la acción del oponente A es "C".
    """
    _validate_player(player)
    _validate_state(state)
    return state[1] if player == "A" else state[0]


# =============================================================================
# 4. REPRESENTACIÓN DE LAS ESTRATEGIAS
# =============================================================================

@dataclass
class Agent:
    """
    Prototipo de una estrategia del torneo.

    El objeto contiene parámetros fijos (name, family, theta, phi, delta) y
    un estado interno que se reinicia al comienzo de cada enfrentamiento.

    Importante
    ----------
    `theta` es la condición angular inicial del prototipo.

    Para las estrategias adaptativas, la variable que realmente cambia durante
    el juego es:

        internal["theta"].

    Esto permite reutilizar el mismo prototipo en el torneo sin contaminar un
    enfrentamiento con el estado interno del anterior.
    """

    name: str
    family: str = "Classical"
    theta: float = 0.0
    phi: float = 0.0
    delta: float = np.pi / 10.0

    # `field(default=None, repr=False)` evita exponer un diccionario mutable
    # como argumento por defecto y deja claro que el estado interno se crea
    # mediante reset().
    internal: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Valida la definición estática de la estrategia."""
        if self.family not in {"Classical", "Quantum"}:
            raise ValueError(
                "family debe ser 'Classical' o 'Quantum'."
            )

        if not np.isfinite(self.theta):
            raise ValueError("theta debe ser finito.")
        if not np.isfinite(self.phi):
            raise ValueError("phi debe ser finito.")
        if self.delta < 0 or not np.isfinite(self.delta):
            raise ValueError("delta debe ser finito y no negativo.")

    def clone(self) -> "Agent":
        """
        Crea una copia limpia de la estrategia para un nuevo enfrentamiento.

        No copia el historial ni la memoria del encuentro anterior.
        """
        cloned = Agent(
            name=self.name,
            family=self.family,
            theta=self.theta,
            phi=self.phi,
            delta=self.delta,
        )
        cloned.reset()
        return cloned

    def reset(self) -> None:
        """
        Reinicia toda la memoria interna.

        `theta` vuelve siempre a la condición inicial almacenada en el
        prototipo; esto es esencial para Q2--Q5.
        """
        self.internal = {
            "grudge": False,
            "history": [],
            "theta": float(self.theta),
        }

    def action_params(
        self,
        player: str,
        rng: np.random.Generator,
    ) -> Tuple[float, float]:
        """
        Devuelve (theta, phi) para la próxima ronda.

        La función implementa la regla de decisión de cada estrategia.

        Las estrategias clásicas se representan dentro del espacio EWL por:
            C = U(0, 0)
            D = U(pi, 0)

        Las estrategias cuánticas pueden utilizar valores intermedios de theta,
        fases no nulas o variables internas adaptativas.
        """
        _validate_player(player)

        if self.internal is None:
            self.reset()

        history: List[str] = self.internal["history"]
        name = self.name

        # ---------------------------------------------------------------------
        # Estrategias clásicas
        # ---------------------------------------------------------------------

        if name == "ALLC":
            # Always Cooperate.
            return 0.0, 0.0

        if name == "ALLD":
            # Always Defect.
            return np.pi, 0.0

        if name == "Random":
            # Acción clásica aleatoria equiprobable.
            return (
                (0.0, 0.0)
                if rng.random() < 0.5
                else (np.pi, 0.0)
            )

        if name == "TFT":
            # Tit for Tat:
            #   primera ronda -> C
            #   rondas siguientes -> imita la última acción del oponente.
            if not history:
                return 0.0, 0.0

            last_opponent = opponent_action(history[-1], player)
            return (
                (0.0, 0.0)
                if last_opponent == "C"
                else (np.pi, 0.0)
            )

        if name == "Grudger":
            # Coopera hasta observar la primera defección del oponente.
            # A partir de entonces, delata para siempre.
            return (
                (np.pi, 0.0)
                if self.internal["grudge"]
                else (0.0, 0.0)
            )

        if name == "TF2T":
            # Tit for Two Tats:
            # únicamente responde con D si el oponente delató en las dos
            # rondas más recientes.
            if len(history) < 2:
                return 0.0, 0.0

            last_two = [
                opponent_action(s, player)
                for s in history[-2:]
            ]
            return (
                (np.pi, 0.0)
                if last_two == ["D", "D"]
                else (0.0, 0.0)
            )

        # ---------------------------------------------------------------------
        # Estrategias cuánticas
        # ---------------------------------------------------------------------

        if name == "Q1_Tibio":
            # Estrategia estacionaria.
            #
            # Por defecto:
            #   theta = pi/2
            #   phi   = 0
            return float(self.theta), float(self.phi)

        if name == "Q2_Bondadoso":
            # Estrategia adaptativa reforzada por cooperación.
            #
            # Comienza en theta = pi y cada cooperación observada reduce theta
            # en delta, desplazándola hacia C = U(0,0).
            return float(self.internal["theta"]), float(self.phi)

        if name == "Q3_Traicionero":
            # Estrategia adaptativa reforzada por defección.
            #
            # Comienza en theta = 0 y cada defección observada aumenta theta
            # en delta, desplazándola hacia D = U(pi,0).
            return float(self.internal["theta"]), float(self.phi)

        if name == "Q4_Reciproco":
            # Estrategia adaptativa bidireccional con phi = 0.
            return float(self.internal["theta"]), float(self.phi)

        if name == "Q5_TibioReciproco":
            # Misma adaptación bidireccional que Q4, pero con fase fija
            # no nula (pi/8 en strategy_set()).
            return float(self.internal["theta"]), float(self.phi)

        if name == "Q9_Switch":
            # Política de memoria uno con selección discreta.
            #
            # Primera ronda:
            #   theta = pi/4
            #
            # Después:
            #   cooperación del oponente -> pi/4
            #   defección del oponente   -> 3pi/4
            if not history:
                return np.pi / 4.0, float(self.phi)

            last_opponent = opponent_action(history[-1], player)
            theta = (
                np.pi / 4.0
                if last_opponent == "C"
                else 3.0 * np.pi / 4.0
            )
            return theta, float(self.phi)

        raise ValueError(f"Estrategia desconocida: {name!r}")

    def observe(self, state: str, player: str) -> None:
        """
        Actualiza la memoria de la estrategia después de cada ronda.

        Sólo se conservan los dos estados más recientes porque ninguna de las
        políticas implementadas necesita una historia observable más larga:

            TFT / Q9       -> memoria uno
            TF2T           -> memoria dos
            Grudger        -> memoria comprimida en un bit (`grudge`)
            Q2--Q5         -> memoria comprimida en `internal["theta"]`

        Obsérvese que Q2--Q5 pueden contener información acumulada de toda la
        trayectoria a través de theta aunque el historial explícito tenga
        longitud máxima dos.
        """
        _validate_player(player)
        _validate_state(state)

        if self.internal is None:
            self.reset()

        history: List[str] = self.internal["history"]
        history.append(state)

        # Conservamos únicamente los dos estados recientes.
        if len(history) > 2:
            del history[:-2]

        name = self.name
        opponent = opponent_action(state, player)

        if name == "Grudger" and opponent == "D":
            self.internal["grudge"] = True

        elif name == "Q2_Bondadoso":
            # Cooperación observada -> theta disminuye.
            if opponent == "C":
                self.internal["theta"] = max(
                    0.0,
                    self.internal["theta"] - self.delta,
                )

        elif name == "Q3_Traicionero":
            # Defección observada -> theta aumenta.
            if opponent == "D":
                self.internal["theta"] = min(
                    np.pi,
                    self.internal["theta"] + self.delta,
                )

        elif name in {"Q4_Reciproco", "Q5_TibioReciproco"}:
            # Regla bidireccional:
            #   cooperación -> theta -= delta
            #   defección   -> theta += delta
            direction = 1.0 if opponent == "D" else -1.0
            self.internal["theta"] += self.delta * direction
            self.internal["theta"] = float(
                np.clip(self.internal["theta"], 0.0, np.pi)
            )


def strategy_set() -> List[Agent]:
    """
    Construye la población completa del torneo.

    La población tiene 12 estrategias:
        6 clásicas + 6 cuánticas.

    Se devuelven prototipos; play_match() crea copias limpias antes de jugar.
    """
    return [
        # Estrategias clásicas.
        Agent("TFT", "Classical"),
        Agent("ALLD", "Classical"),
        Agent("ALLC", "Classical"),
        Agent("Random", "Classical"),
        Agent("Grudger", "Classical"),
        Agent("TF2T", "Classical"),

        # Estrategias cuánticas.
        Agent(
            "Q1_Tibio",
            "Quantum",
            theta=np.pi / 2.0,
            phi=0.0,
        ),
        Agent(
            "Q2_Bondadoso",
            "Quantum",
            theta=np.pi,
            phi=0.0,
        ),
        Agent(
            "Q3_Traicionero",
            "Quantum",
            theta=0.0,
            phi=0.0,
        ),
        Agent(
            "Q4_Reciproco",
            "Quantum",
            theta=np.pi / 2.0,
            phi=0.0,
        ),
        Agent(
            "Q5_TibioReciproco",
            "Quantum",
            theta=np.pi / 2.0,
            phi=np.pi / 8.0,
        ),
        Agent(
            "Q9_Switch",
            "Quantum",
            theta=0.0,
            phi=0.0,
        ),
    ]


# =============================================================================
# 5. SIMULACIÓN DE UN ENFRENTAMIENTO
# =============================================================================

def sample_state(
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    Muestrea un estado a partir de una distribución de longitud 4.

    Se usa rng.choice en vez de construir manualmente la CDF. Es más legible y
    evita correcciones ad hoc cuando la suma acumulada termina en
    0.999999999999... por redondeo.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.shape != (N_STATES,):
        raise ValueError(
            f"Se esperaba una distribución de forma ({N_STATES},)."
        )

    if np.any(probabilities < -1e-12):
        raise ValueError("La distribución contiene probabilidades negativas.")

    probabilities = np.maximum(probabilities, 0.0)
    total = float(probabilities.sum())

    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("La distribución no puede normalizarse.")

    probabilities = probabilities / total
    return int(rng.choice(N_STATES, p=probabilities))


def play_match(
    proto_a: Agent,
    proto_b: Agent,
    gamma: float,
    rounds: int,
    rng: np.random.Generator,
) -> Tuple[float, float, np.ndarray]:
    """
    Simula un enfrentamiento repetido entre dos estrategias.

    Parámetros
    ----------
    proto_a, proto_b:
        Prototipos de las estrategias. Se clonan para que cada enfrentamiento
        comience sin memoria previa.

    gamma:
        Grado de entrelazamiento EWL.

    rounds:
        Número de rondas del enfrentamiento.

    rng:
        Generador NumPy utilizado para todo el muestreo del encuentro.

    Retorna
    -------
    mean_sentence_A:
        Condena media por ronda de la estrategia situada como jugador A.

    mean_sentence_B:
        Condena media por ronda de la estrategia situada como jugador B.

    outcome_fractions:
        Vector:

            (f_CC, f_CD, f_DC, f_DD)

        de frecuencias empíricas durante el enfrentamiento.
    """
    if rounds <= 0:
        raise ValueError("rounds debe ser un entero positivo.")

    # Copias limpias: ningún enfrentamiento hereda el estado interno de otro.
    A = proto_a.clone()
    B = proto_b.clone()

    total_a = 0.0
    total_b = 0.0

    # Conteo de CC, CD, DC y DD.
    counts = np.zeros(N_STATES, dtype=int)

    for _ in range(rounds):
        # Cada estrategia selecciona sus parámetros locales utilizando
        # únicamente la información que su regla le permite consultar.
        theta_a, phi_a = A.action_params("A", rng)
        theta_b, phi_b = B.action_params("B", rng)

        # Probabilidades del circuito EWL para la ronda actual.
        p = ewl_probabilities(
            theta_a,
            phi_a,
            theta_b,
            phi_b,
            gamma,
        )

        # Medición / muestreo del resultado observable.
        idx = sample_state(p, rng)
        state = STATE_LABELS[idx]

        # Acumulación de condenas.
        total_a += SENTENCE_A[idx]
        total_b += SENTENCE_B[idx]

        # Conteo del resultado.
        counts[idx] += 1

        # Retroalimentación: cada jugador observa el mismo resultado conjunto,
        # pero interpreta la acción del oponente desde su propia posición.
        A.observe(state, "A")
        B.observe(state, "B")

    mean_a = total_a / rounds
    mean_b = total_b / rounds
    fractions = counts.astype(float) / rounds

    # Comprobación de consistencia.
    if not np.isclose(fractions.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Las frecuencias del enfrentamiento no suman uno.")

    return mean_a, mean_b, fractions


# =============================================================================
# 6. TORNEO ROUND-ROBIN
# =============================================================================

def match_seed(
    base_seed: int,
    i: int,
    j: int,
    repetition: int,
    gamma: float,
) -> int:
    """
    Construye una semilla determinista para un enfrentamiento concreto.

    El esquema conserva la fórmula usada en la implementación original para
    mantener reproducibilidad de los resultados existentes.
    """
    return int(
        base_seed
        + i * 100003
        + j * 1009
        + repetition * 17
        + int(float(gamma) * 1e6)
    )


def tournament(
    gamma: float,
    rounds: int = 200,
    repetitions: int = 5,
    seed: int = DEFAULT_SEED,
) -> Tuple[
    List[Agent],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Ejecuta el torneo completo para un valor fijo de gamma.

    Cada estrategia S_i se enfrenta con todas las estrategias S_j, incluida
    ella misma. El par es ORDENADO:

        fila i    -> jugador A / estrategia evaluada
        columna j -> jugador B / oponente

    Por ello C_ij no tiene por qué ser igual a C_ji.

    Retorna
    -------
    strategies:
        Lista de las 12 estrategias.

    sentence:
        Matriz C de tamaño 12 x 12.
        C[i,j] = condena media por ronda de S_i como A frente a S_j como B.

    opponent_sentence:
        Condena media del jugador B para los mismos enfrentamientos.

    global_sentence:
        Promedio de cada fila de sentence.
        Es el desempeño global de cada estrategia frente a toda la población.

    global_cc:
        Frecuencia media global de CC de cada estrategia.

    state_fractions:
        Tensor 12 x 12 x 4 con las frecuencias de CC, CD, DC y DD para cada
        enfrentamiento.
    """
    if rounds <= 0:
        raise ValueError("rounds debe ser positivo.")
    if repetitions <= 0:
        raise ValueError("repetitions debe ser positivo.")

    strategies = strategy_set()
    n = len(strategies)

    sentence = np.zeros((n, n), dtype=float)
    opponent_sentence = np.zeros((n, n), dtype=float)

    # Tensor:
    #   eje 0 -> estrategia evaluada A
    #   eje 1 -> oponente B
    #   eje 2 -> resultado CC/CD/DC/DD
    state_fractions = np.zeros((n, n, N_STATES), dtype=float)

    for i, strategy_a in enumerate(strategies):
        for j, strategy_b in enumerate(strategies):
            sentences_a: List[float] = []
            sentences_b: List[float] = []
            fractions: List[np.ndarray] = []

            for repetition in range(repetitions):
                local_seed = match_seed(
                    seed,
                    i,
                    j,
                    repetition,
                    gamma,
                )
                rng = np.random.default_rng(local_seed)

                mean_a, mean_b, f = play_match(
                    strategy_a,
                    strategy_b,
                    gamma,
                    rounds,
                    rng,
                )

                sentences_a.append(mean_a)
                sentences_b.append(mean_b)
                fractions.append(f)

            # Promedio sobre las repeticiones del mismo par ordenado.
            sentence[i, j] = float(np.mean(sentences_a))
            opponent_sentence[i, j] = float(np.mean(sentences_b))
            state_fractions[i, j, :] = np.mean(fractions, axis=0)

    # Desempeño global:
    # promedio de toda la fila, es decir, frente a los 12 oponentes.
    global_sentence = sentence.mean(axis=1)

    # Frecuencia global de CC:
    # promedio de f_CC frente a todos los oponentes.
    global_cc = state_fractions[:, :, 0].mean(axis=1)

    return (
        strategies,
        sentence,
        opponent_sentence,
        global_sentence,
        global_cc,
        state_fractions,
    )


# =============================================================================
# 7. CONSTRUCCIÓN DE RANKINGS
# =============================================================================

def ranking_df(
    strategies: List[Agent],
    global_sentence: np.ndarray,
    global_cc: np.ndarray,
    gamma: float,
    global_state_fracs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Construye la tabla de desempeño global de las estrategias.

    El ranking es ascendente porque:

        menor condena = mejor desempeño.

    Si se proporcionan las frecuencias de los cuatro estados, también se
    incorporan como columnas:
        mean_CC_fraction
        mean_CD_fraction
        mean_DC_fraction
        mean_DD_fraction.
    """
    data = {
        "strategy": [s.name for s in strategies],
        "family": [s.family for s in strategies],
        "mean_sentence_years": np.asarray(global_sentence, dtype=float),
        "mean_CC_fraction": np.asarray(global_cc, dtype=float),
    }

    if global_state_fracs is not None:
        global_state_fracs = np.asarray(global_state_fracs, dtype=float)

        expected_shape = (len(strategies), N_STATES)
        if global_state_fracs.shape != expected_shape:
            raise ValueError(
                "global_state_fracs debe tener forma "
                f"{expected_shape}, recibió {global_state_fracs.shape}."
            )

        for k, label in enumerate(STATE_LABELS):
            data[f"mean_{label}_fraction"] = global_state_fracs[:, k]

    df = pd.DataFrame(data)

    # rank = 1 -> menor condena.
    df["rank"] = (
        df["mean_sentence_years"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    df["gamma"] = float(gamma)

    return (
        df.sort_values(["rank", "strategy"])
        .reset_index(drop=True)
    )


# =============================================================================
# 8. FUNCIONES DE VISUALIZACIÓN
# =============================================================================

def save_heatmap(
    matrix: np.ndarray,
    names: List[str],
    title: str,
    path: Path,
    colorbar_label: str,
) -> None:
    """
    Guarda un mapa de calor.

    Para matrices de condena:
        tonos asociados a valores menores -> mejor desempeño individual.

    La dirección exacta claro/oscuro depende del mapa de color por defecto de
    Matplotlib; por eso la interpretación cuantitativa debe realizarse usando
    la barra de color y no únicamente el tono.
    """
    matrix = np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(
        range(len(names)),
        names,
        rotation=65,
        ha="right",
    )
    ax.set_yticks(range(len(names)), names)

    ax.set_xlabel("Oponente")
    ax.set_ylabel("Estrategia evaluada")
    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def strategy_linestyle(family: str) -> str:
    """
    Convención visual común de las curvas.

    Classical -> línea continua
    Quantum   -> línea discontinua
    """
    return "-" if family == "Classical" else "--"


# =============================================================================
# 9. EXPERIMENTO COMPLETO
# =============================================================================

def run(
    outdir: str | Path,
    rounds: int,
    repetitions: int,
    seed: int,
) -> None:
    """
    Ejecuta todos los experimentos del torneo y guarda figuras/tablas.

    Se distinguen dos niveles de simulación:

    1) Regímenes principales:
           gamma = 0, pi/4, pi/2
       usando `rounds` y `repetitions`.

    2) Barrido exploratorio:
           11 puntos de gamma
       usando como máximo 100 rondas y 3 repeticiones por par.
    """
    if rounds <= 0:
        raise ValueError("--rounds debe ser positivo.")
    if repetitions <= 0:
        raise ValueError("--repetitions debe ser positivo.")

    out = Path(outdir)
    figures_dir = out / "figures"
    tables_dir = out / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    regimes = [
        ("gamma_0", 0.0),
        ("gamma_pi4", np.pi / 4.0),
        ("gamma_pi2", np.pi / 2.0),
    ]

    regime_rankings: List[pd.DataFrame] = []
    summary_parts: List[str] = []

    # -------------------------------------------------------------------------
    # 9.1 Tres regímenes principales
    # -------------------------------------------------------------------------
    for tag, gamma in regimes:
        (
            strategies,
            sentence_matrix,
            opponent_sentence_matrix,
            global_sentence,
            global_cc,
            state_fractions,
        ) = tournament(
            gamma=gamma,
            rounds=rounds,
            repetitions=repetitions,
            seed=seed,
        )

        names = [s.name for s in strategies]

        # Matriz C_ij:
        # condena de la estrategia evaluada A (fila) frente a B (columna).
        pd.DataFrame(
            sentence_matrix,
            index=names,
            columns=names,
        ).to_csv(
            tables_dir / f"sentence_matrix_{tag}.csv"
        )

        # NUEVO EN LA VERSIÓN DEPURADA:
        # se conserva también la matriz de la condena del jugador B. El cálculo
        # ya existía en el código original, pero el resultado no se guardaba.
        pd.DataFrame(
            opponent_sentence_matrix,
            index=names,
            columns=names,
        ).to_csv(
            tables_dir / f"opponent_sentence_matrix_{tag}.csv"
        )

        # Guardar las cuatro matrices de frecuencias, no sólo CC.
        for state_index, state in enumerate(STATE_LABELS):
            pd.DataFrame(
                state_fractions[:, :, state_index],
                index=names,
                columns=names,
            ).to_csv(
                tables_dir / f"{state}_fraction_matrix_{tag}.csv"
            )

        # Promedio global de las frecuencias frente a todos los oponentes.
        global_state_fracs = state_fractions.mean(axis=1)

        ranking = ranking_df(
            strategies,
            global_sentence,
            global_cc,
            gamma,
            global_state_fracs,
        )
        ranking.to_csv(
            tables_dir / f"ranking_{tag}.csv",
            index=False,
        )
        regime_rankings.append(ranking)

        # Matriz de condena.
        save_heatmap(
            sentence_matrix,
            names,
            rf"Matriz de condena esperada del torneo, "
            rf"$\gamma={gamma / np.pi:.2g}\pi$",
            figures_dir / f"fig_7_7_sentence_matrix_{tag}.png",
            "Condena esperada por ronda (años)",
        )

        # Matriz CC.
        save_heatmap(
            state_fractions[:, :, 0],
            names,
            rf"Frecuencia de cooperación mutua, "
            rf"$\gamma={gamma / np.pi:.2g}\pi$",
            figures_dir / f"fig_7_7_CC_matrix_{tag}.png",
            "Fracción de resultados CC",
        )

        summary_parts.append(
            f"\nREGIME {tag} gamma={gamma:.8f}\n"
            + ranking.to_string(index=False)
        )

    # Tabla conjunta con los tres regímenes.
    all_regimes = pd.concat(
        regime_rankings,
        ignore_index=True,
    )
    all_regimes.to_csv(
        tables_dir / "ranking_three_regimes.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 9.2 Referencia exclusivamente clásica para gamma = 0
    # -------------------------------------------------------------------------
    #
    # Este cálculo responde a una pregunta diferente del ranking global:
    #
    # Ranking global:
    #   cada estrategia clásica se promedia frente a las 12 estrategias.
    #
    # Ranking clásico:
    #   cada estrategia clásica se promedia únicamente frente a las
    #   6 estrategias clásicas.
    #
    classical_names = [
        s.name
        for s in strategy_set()
        if s.family == "Classical"
    ]

    matrix_gamma_0 = pd.read_csv(
        tables_dir / "sentence_matrix_gamma_0.csv",
        index_col=0,
    )

    classical_block = matrix_gamma_0.loc[
        classical_names,
        classical_names,
    ]

    classical_block.to_csv(
        tables_dir / "classical_sentence_matrix_gamma_0.csv"
    )

    classical_ranking = (
        classical_block
        .mean(axis=1)
        .rename("mean_sentence_years_classical_only")
        .sort_values()
        .reset_index()
        .rename(columns={"index": "strategy"})
    )

    classical_ranking["rank_classical_only"] = np.arange(
        1,
        len(classical_ranking) + 1,
    )

    classical_ranking.to_csv(
        tables_dir / "classical_ranking_gamma_0.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        classical_ranking["strategy"],
        classical_ranking["mean_sentence_years_classical_only"],
    )
    ax.set_ylabel("Condena media por ronda (años)")
    ax.set_xlabel("Estrategia clásica evaluada")
    ax.set_title(
        r"Ranking clásico frente a oponentes clásicos, $\gamma=0$"
    )
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "fig_7_7_classical_only_ranking_gamma_0.png",
        dpi=220,
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 9.3 Comparación global compacta de los tres regímenes
    # -------------------------------------------------------------------------
    pivot = all_regimes.pivot(
        index="strategy",
        columns="gamma",
        values="mean_sentence_years",
    )

    # Ordenamos las estrategias según su desempeño en gamma = 0 para conservar
    # una referencia visual común en los tres conjuntos de barras.
    order = (
        all_regimes[np.isclose(all_regimes["gamma"], 0.0)]
        .sort_values("mean_sentence_years")["strategy"]
        .tolist()
    )
    pivot = pivot.loc[order]

    x = np.arange(len(pivot.index))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for offset_index, gamma in enumerate(
        [0.0, np.pi / 4.0, np.pi / 2.0]
    ):
        ax.bar(
            x + (offset_index - 1) * width,
            pivot[gamma].values,
            width,
            label=rf"$\gamma={gamma / np.pi:.2g}\pi$",
        )

    ax.set_xticks(
        x,
        pivot.index,
        rotation=55,
        ha="right",
    )
    ax.set_ylabel("Condena media global por ronda (años)")
    ax.set_xlabel("Estrategia")
    ax.set_title(
        "Desempeño global en los tres regímenes de entrelazamiento"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "fig_7_7_global_three_regimes.png",
        dpi=240,
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 9.4 Promedio de los grupos clásico y cuántico
    # -------------------------------------------------------------------------
    #
    # Importante:
    # este promedio NO dice que todas las estrategias de un grupo sean mejores
    # o peores que todas las del otro. Sólo resume las seis medias globales de
    # cada grupo.
    grouped = (
        all_regimes
        .groupby(["gamma", "family"])["mean_sentence_years"]
        .mean()
        .reset_index()
    )

    grouped.to_csv(
        tables_dir / "family_mean_sentence_three_regimes.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for family, family_df in grouped.groupby("family"):
        ax.plot(
            family_df["gamma"] / np.pi,
            family_df["mean_sentence_years"],
            marker="o",
            label=family,
        )

    ax.set_xlabel(r"$\gamma/\pi$")
    ax.set_ylabel("Condena media por ronda (años)")
    ax.set_title("Grupos de estrategias clásicas y cuánticas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "fig_7_7_classical_vs_quantum_families.png",
        dpi=220,
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 9.5 Ranking por régimen
    # -------------------------------------------------------------------------
    for tag, gamma in regimes:
        ranking = (
            all_regimes[
                np.isclose(all_regimes["gamma"], gamma)
            ]
            .sort_values(
                "mean_sentence_years",
                ascending=True,
            )
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            ranking["strategy"],
            ranking["mean_sentence_years"],
        )
        ax.set_ylabel("Condena esperada por ronda (años)")
        ax.set_title(
            rf"Ranking del torneo para "
            rf"$\gamma={gamma / np.pi:.2g}\pi$"
        )
        ax.tick_params(axis="x", rotation=65)
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"fig_7_7_ranking_{tag}.png",
            dpi=220,
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 9.6 Barrido exploratorio de gamma
    # -------------------------------------------------------------------------
    #
    # El barrido utiliza menos rondas/repeticiones para detectar tendencias y
    # cruces en la jerarquía sin multiplicar excesivamente el costo del torneo.
    gammas = np.linspace(0.0, np.pi / 2.0, 11)

    sweep_rows: List[pd.DataFrame] = []

    sweep_rounds = min(rounds, 100)
    sweep_repetitions = min(repetitions, 3)

    for k, gamma in enumerate(gammas):
        (
            strategies,
            _sentence_matrix,
            _opponent_sentence_matrix,
            global_sentence,
            global_cc,
            state_fractions,
        ) = tournament(
            gamma=gamma,
            rounds=sweep_rounds,
            repetitions=sweep_repetitions,
            seed=seed + 500000 + k * 1000,
        )

        ranking = ranking_df(
            strategies,
            global_sentence,
            global_cc,
            gamma,
            state_fractions.mean(axis=1),
        )

        sweep_rows.append(ranking)

    sweep = pd.concat(
        sweep_rows,
        ignore_index=True,
    )

    sweep.to_csv(
        tables_dir / "ranking_gamma_sweep.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 9.6.1 Condena media global vs gamma
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, strategy_df in sweep.groupby("strategy"):
        strategy_df = strategy_df.sort_values("gamma")
        family = strategy_df["family"].iloc[0]

        ax.plot(
            strategy_df["gamma"] / np.pi,
            strategy_df["mean_sentence_years"],
            linestyle=strategy_linestyle(family),
            label=name,
        )

    ax.set_xlabel(r"$\gamma/\pi$")
    ax.set_ylabel("Condena media por ronda (años)")
    ax.set_title("Condena esperada en función del entrelazamiento")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "fig_7_7_strategy_sentence_vs_gamma.png",
        dpi=240,
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 9.6.2 Posición en el ranking vs gamma
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, strategy_df in sweep.groupby("strategy"):
        strategy_df = strategy_df.sort_values("gamma")
        family = strategy_df["family"].iloc[0]

        ax.plot(
            strategy_df["gamma"] / np.pi,
            strategy_df["rank"],
            linestyle=strategy_linestyle(family),
            marker=".",
            label=name,
        )

    ax.set_xlabel(r"$\gamma/\pi$")
    ax.set_ylabel("Rango (1 = menor condena)")
    ax.invert_yaxis()
    ax.set_title("Reordenamiento de la jerarquía con el entrelazamiento")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "fig_7_7_strategy_rank_vs_gamma.png",
        dpi=240,
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 9.6.3 Frecuencias de CC, CD, DC y DD vs gamma
    # -------------------------------------------------------------------------
    #
    # Analizar sólo CC no basta para explicar la condena:
    #
    #   C_A = 1*f_CC + 5*f_CD + 0*f_DC + 3*f_DD.
    #
    # Por ello se generan cuatro curvas separadas.
    outcome_titles = {
        "CC": "Frecuencia de cooperación mutua",
        "CD": "Frecuencia del resultado CD",
        "DC": "Frecuencia del resultado DC",
        "DD": "Frecuencia de defección mutua",
    }

    outcome_files = {
        "CC": "fig_7_7_cooperation_vs_gamma.png",
        "CD": "fig_7_7_CD_vs_gamma.png",
        "DC": "fig_7_7_DC_vs_gamma.png",
        "DD": "fig_7_7_DD_vs_gamma.png",
    }

    for state in STATE_LABELS:
        fig, ax = plt.subplots(figsize=(11, 6))

        for name, strategy_df in sweep.groupby("strategy"):
            strategy_df = strategy_df.sort_values("gamma")
            family = strategy_df["family"].iloc[0]

            ax.plot(
                strategy_df["gamma"] / np.pi,
                strategy_df[f"mean_{state}_fraction"],
                linestyle=strategy_linestyle(family),
                label=name,
            )

        ax.set_xlabel(r"$\gamma/\pi$")
        ax.set_ylabel(f"Fracción media de resultados {state}")
        ax.set_title(outcome_titles[state])
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(
            figures_dir / outcome_files[state],
            dpi=240,
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 9.7 Clases de interacción: C-C, C-Q y Q-Q
    # -------------------------------------------------------------------------
    #
    # C-Q agrupa ambas orientaciones:
    #   clásica como A vs cuántica como B
    #   cuántica como A vs clásica como B
    #
    # Por ello debe interpretarse como una media de enfrentamientos mixtos,
    # no como una matriz dirigida de una sola orientación.
    population = strategy_set()
    names = [s.name for s in population]
    family_by_name = {
        s.name: s.family
        for s in population
    }

    interaction_rows: List[dict] = []

    for tag, gamma in regimes:
        matrix = pd.read_csv(
            tables_dir / f"sentence_matrix_{tag}.csv",
            index_col=0,
        )

        values = {
            "C-C": [],
            "C-Q": [],
            "Q-Q": [],
        }

        for strategy_a in names:
            for strategy_b in names:
                family_a = family_by_name[strategy_a]
                family_b = family_by_name[strategy_b]

                if family_a == "Classical" and family_b == "Classical":
                    key = "C-C"
                elif family_a == "Quantum" and family_b == "Quantum":
                    key = "Q-Q"
                else:
                    key = "C-Q"

                values[key].append(
                    float(matrix.loc[strategy_a, strategy_b])
                )

        for key, group_values in values.items():
            interaction_rows.append(
                {
                    "gamma": gamma,
                    "interaction_class": key,
                    "mean_sentence_years": float(
                        np.mean(group_values)
                    ),
                }
            )

    interaction_df = pd.DataFrame(interaction_rows)

    interaction_df.to_csv(
        tables_dir / "interaction_class_means.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for key, class_df in interaction_df.groupby(
        "interaction_class"
    ):
        ax.plot(
            class_df["gamma"] / np.pi,
            class_df["mean_sentence_years"],
            marker="o",
            label=key,
        )

    ax.set_xlabel(r"$\gamma/\pi$")
    ax.set_ylabel(
        "Condena media de la estrategia evaluada por ronda (años)"
    )
    ax.set_title(
        "Encuentros clásico--clásico, "
        "clásico--cuántico y cuántico--cuántico"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "fig_7_7_interaction_classes.png",
        dpi=220,
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 9.8 Estrategia que ocupa el primer lugar para cada gamma
    # -------------------------------------------------------------------------
    top_strategies = (
        sweep
        .sort_values(["gamma", "rank"])
        .groupby("gamma", as_index=False)
        .first()[
            [
                "gamma",
                "strategy",
                "mean_sentence_years",
            ]
        ]
    )

    top_strategies.to_csv(
        tables_dir / "top_strategy_by_gamma.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 9.9 Resumen textual
    # -------------------------------------------------------------------------
    summary_text = (
        f"Axelrod-style tournament\n"
        f"Rounds={rounds}, repetitions={repetitions}\n"
        + "\n".join(summary_parts)
        + "\n\nTOP STRATEGY BY GAMMA\n"
        + top_strategies.to_string(index=False)
        + "\n\nINTERACTION CLASS MEANS\n"
        + interaction_df.to_string(index=False)
    )

    (out / "summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    # Salida compacta en terminal.
    print(
        all_regimes
        .sort_values(["gamma", "rank"])
        .to_string(index=False)
    )
    print("\nTop-strategy changes:")
    print(top_strategies.to_string(index=False))


# =============================================================================
# 10. INTERFAZ DE LÍNEA DE COMANDOS
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Construye y valida los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Torneo round-robin clásico-cuántico del "
            "Dilema del Prisionero Iterativo EWL."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_tournament"),
        help="Directorio de salida.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="Número de rondas por enfrentamiento principal.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Número de repeticiones por par ordenado.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla base para reproducibilidad.",
    )

    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds debe ser mayor que cero.")
    if args.repetitions <= 0:
        parser.error("--repetitions debe ser mayor que cero.")

    return args


if __name__ == "__main__":
    arguments = parse_args()

    run(
        outdir=arguments.output,
        rounds=arguments.rounds,
        repetitions=arguments.repetitions,
        seed=arguments.seed,
    )
