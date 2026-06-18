from pimcs.dicke import Dicke, DickeState
from pimcs.operators import validate_dimension
import pimcs.c_gen as c
import numpy as np
import ctypes, math


class MCSolveResult:
    def __init__(self, expect):
        self.expect = expect


def mcsolve(system: Dicke, psi0: DickeState, tlist: list[float], e_ops = [], ntraj: int = 0, ncpu: int = 0,
            jtol: float = 0.01) -> MCSolveResult:
    assert psi0.j == system.N/2, "Only states in the maximal J-sector are currently supported"

    if not math.isclose(psi0.norm(), 1):
        raise ValueError(f"Initial wavefunction is not normalized, got norm of {psi0.norm()}")

    tlist = np.array(tlist)
    assert len(tlist) > 1 and tlist[0] == 0

    dt = tlist[1]
    assert np.isclose(tlist[1:] - tlist[:-1], dt).all()

    spin_dim, boson_dim = validate_dimension(system.hamiltonian)

    if spin_dim is None:
        raise ValueError(f"Hamiltonian contains no spin operators")

    if boson_dim is None:
        boson_dim = 1 # must have at least one, even just for free spins

    boson_energy, padding, code = c.generate_backend_code(system.hamiltonian, e_ops, displace = False)
    config = c.generate_config(system, boson_dim, tlist, len(e_ops), ntraj, ncpu, boson_energy, jtol, padding, True, len(tlist))

    with open("pimcs/c_backend/tmp.h", 'w') as handle:
        handle.write(code)

    with open("pimcs/c_backend/tmpconfig.h", 'w') as handle:
        handle.write(config)

    print("Building optimized executable...")
    lib, hash_id = c.build_executable()

    print("Running trajectories...")
    coeffs = np.ascontiguousarray(psi0.coeffs, dtype = np.complex64)
    lib.run_trajectories(ctypes.c_uint64(hash_id), coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    expect = np.zeros((len(e_ops), len(tlist)), dtype = np.complex128)
 
    for t in range(ntraj):
        t, *data = np.loadtxt(f"trajectory-{hash_id:x}-{t+1}.txt").T

        for i in range(len(e_ops)):
            complex_data = data[2*i] + data[2*i+1] * 1j
            expect[i] += np.interp(tlist, t, complex_data)

    return MCSolveResult(expect / ntraj)

