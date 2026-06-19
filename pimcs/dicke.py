from pimcs.operators import Leaf
from scipy.special import gammaln
import numpy as np


class Dicke:
    def __init__(
        self,
        N: int,
        hamiltonian = None,
        emission: float = 0.0,
        dephasing: float = 0.0,
        pumping: float = 0.0,
        collective_emission: float = 0.0,
        collective_dephasing: float = 0.0,
        collective_pumping: float = 0.0,
        cavity_absorption: float = 0.0,
        cavity_emission: float = 0.0,
        cavity_loss: float = 0.0,
    ):
        if hamiltonian is None:
            hamiltonian = Leaf(1) # identity

        self.N = N
        self.hamiltonian = hamiltonian
        self.emission = float(emission)
        self.dephasing = float(dephasing)
        self.pumping = float(pumping)
        self.collective_emission  = float(collective_emission)
        self.collective_dephasing = float(collective_dephasing)
        self.collective_pumping = float(collective_pumping)
        self.cavity_absorption = float(cavity_absorption)
        self.cavity_emission = float(cavity_emission)
        self.cavity_loss = float(cavity_loss)



class DickeState:
    def __init__(self, j: float, m: float = None):
        if j % 0.5 != 0:
            raise ValueError(f"J sector must be a half-integer value, got {j}")

        self.j = j
        self.coeffs = np.zeros(int(2*j + 1), dtype = np.complex128)

        if m is not None:
            if m % 0.5 != j % 0.5:
                raise ValueError(f"M must be a half-integer of the same kind as J, got {m}") 

            index = int(j - m)
            self.coeffs[index] = 1

    def __add__(self, other: DickeState) -> DickeState:
        if self.j != other.j:
            raise ValueError(f"Cannot add Dicke states of different J sectors: {self.j} != {other.j}")

        result = Dicke(self.j) 
        result.coeffs = self.coeffs + other.coeffs
        return result

    def __mul__(self, other: complex) -> DickeState:
        try:
            other = np.complex128(other)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot convert {value!r} to a complex number") from e

        result = Dicke(self.j)
        result.coeffs = self.coeffs * other
        return result

    def norm(self):
        return np.sum((self.coeffs * self.coeffs.conj()).real)



# helper constructors

def dicke(N: int, j: float, m: float):
    assert 0 <= j <= N/2, "Invalid value for total spin"
    assert np.abs(m) <= j, "Invalid m for value of j"
    return DickeState(j,m)


def ground(N: int) -> DickeState:
    return DickeState(N/2, -N/2)


def excited(N: int) -> DickeState:
    return DickeState(N/2, N/2)


def rotated_qubits(N: int, angle: float) -> DickeState:
    if angle == 0:
        return exicted(N)

    # TODO: handle entire angle range properly (including poles in log-space)
    try:
        log_cos = np.log(np.cos(angle / 2))
        log_sin = np.log(np.sin(angle / 2))
    except:
        raise ValueError(f"Invalid rotation angle, got {angle}")

    j = N / 2
    m = j - np.arange(0, N + 1)
    u = np.int64(m + j) # number of spin up atoms

    log_probability = 0.5 * (gammaln(N + 1) - gammaln(u + 1) - gammaln(N - u + 1)) + u * log_cos + (N - u) * log_sin

    probability = np.exp(log_probability)
    phase = -1j ** ((N - u) % 4)

    result = DickeState(j)
    result.coeffs = probability * phase
    return result

