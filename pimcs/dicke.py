from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from .qoperator import QLeafExpr, QExpr


class Dicke:
    def __init__(
            self,
            dims: tuple[int, int],
            hamiltonian: Optional[QExpr] = None,
            photon_loss_rate: float = 0.0,
            local_dephasing_rate: float = 0.0,
            local_emission_rate: float = 0.0,
            local_pumping_rate: float = 0.0,
            collective_dephasing_rate: float = 0.0,
            collective_emission_rate: float = 0.0,
            collective_pumping_rate: float = 0.0,
            cavity_absorption_rate: float = 0.0,
            cavity_emission_rate: float = 0.0):
        if hamiltonian is None:
            hamiltonian = QLeafExpr(1)  # identity

        if not isinstance(hamiltonian, QExpr):
            raise ValueError(f"Hamiltonian must be a QExpr, got {type(hamiltonian)!r}")

        self.number_of_emitters = int(dims[0])
        self.photon_truncation = int(dims[1])
        self.hamiltonian = hamiltonian

        self.photon_loss_rate = float(photon_loss_rate)
        self.local_dephasing_rate = float(local_dephasing_rate)
        self.local_emission_rate = float(local_emission_rate)
        self.local_pumping_rate = float(local_pumping_rate)
        self.collective_dephasing_rate = float(collective_dephasing_rate)
        self.collective_emission_rate = float(collective_emission_rate)
        self.collective_pumping_rate = float(collective_pumping_rate)
        self.cavity_absorption_rate = float(cavity_absorption_rate)
        self.cavity_emission_rate = float(cavity_emission_rate)


# state is defined as tensor_prod_j exp(-i sigma_x * angle / 2) |up_j>
class DickeState:
    def __init__(self, angle):
        self.angle = float(angle)

def ground() -> DickeState:
    return DickeState(np.pi)

def excited() -> DickeState:
    return DickeState(0)

def rotated_qubits(angle: float) -> DickeState:
    return DickeState(angle)

# maximally coherent spin state
def coherent_state() -> DickeState:
    return rotated_qubits(np.pi / 2)
