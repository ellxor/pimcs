import numpy as np
import ctypes, math

from multiprocessing import Process

from .dicke import Dicke, DickeState
from .operators import validate_dimension
from . import c_gen as c


class MCSolveResult:
    def __init__(self, expect, boson_density, spin_density):
        self.expect = expect
        self.boson_density = boson_density
        self.spin_density = spin_density


class MCSolver:
    def __init__(self, libpath, id, psi0):
        self.libpath = libpath
        self.psi0 = psi0
        self.id = id

    def __call__(self):
        coeffs = np.ascontiguousarray(self.psi0.coeffs, dtype = np.complex64)
        lib = ctypes.CDLL(self.libpath)

        lib.run_trajectories(
            ctypes.c_float(self.psi0.j),
            ctypes.c_uint64(self.id),
            coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ) 


def running_in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def mcsolve(system: Dicke, psi0: DickeState, tlist: list[float], e_ops = [], ntraj: int = 0, ncpu: int = 0,
            jtol: float = 0.01, stol: float = 1e-20, rkpoly: int = 4) -> MCSolveResult:

    if psi0.j > system.N/2:
        raise ValueError(f"J spin length is larger than N/2, where N = {system.N}")

    if not math.isclose(psi0.norm(), 1):
        raise ValueError(f"Initial wavefunction is not normalized, got norm of {psi0.norm()}")

    tlist = np.array(tlist)
    assert len(tlist) > 1 and tlist[0] == 0

    dt = tlist[1]
    assert np.isclose(tlist[1:] - tlist[:-1], dt).all(), "tlist must be linearly spaced"

    spin_dim, boson_dim = validate_dimension(system.hamiltonian)

    if spin_dim is None:
        raise ValueError(f"Hamiltonian contains no spin operators")

    if boson_dim is None:
        boson_dim = 1 # must have at least one, even just for free spins

    code, spin_width, boson_width = c.generate_backend_code(system.hamiltonian, e_ops, displace = False)
    config = c.generate_config(system, boson_dim, tlist, len(e_ops), ntraj, ncpu, jtol, stol, spin_width, boson_width, len(tlist), rkpoly)

    with open("pimcs/c_backend/tmp.h", 'w') as handle:
        handle.write(code)

    with open("pimcs/c_backend/tmpconfig.h", 'w') as handle:
        handle.write(config)

    print("Building optimized executable...")
    libpath, hash_id = c.build_executable()
    solver = MCSolver(libpath, hash_id, psi0)

    print("Running trajectories...")
    if running_in_notebook():
        thread = Process(target = solver)
        thread.start()
        thread.join()
    else:
        solver()

    expect = np.zeros((len(e_ops), len(tlist)), dtype = np.complex64)
    boson_density = np.zeros((boson_dim, len(tlist)))
    spin_density = np.zeros((spin_dim + 1, len(tlist)))
 
    for t in range(ntraj):
        t, *data = np.loadtxt(f"trajectory-{hash_id:x}-{t+1}.txt").T

        for i in range(len(e_ops)):
            complex_data = data[2*i] + data[2*i+1] * 1j
            expect[i] += np.interp(tlist, t, complex_data)

        i = 2 * (i + 1)

        for k in range(boson_dim):
            boson_density[k] += np.interp(tlist, t, data[i+k])

        i += boson_dim

        for k in range(spin_dim + 1):
            spin_density[k] += np.interp(tlist, t, data[i+k])

    expect /= ntraj
    boson_density /= ntraj
    spin_density /= ntraj

    expect = [ e.real if op.is_herm() else e for e, op in zip(expect, e_ops) ]
    return MCSolveResult(expect, boson_density, spin_density)

