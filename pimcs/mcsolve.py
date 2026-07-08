import numpy as np
import ctypes, math

from .dicke import Dicke, DickeState
from .operators import validate_dimension
from . import c_gen as c


class MCSolveResult:
    def __init__(self, expect):
        self.expect = expect

class MCSolver:
    def __init__(self, libpath, id, psi0, tfuncs, correlation_time: float, output_shape: tuple[int,int,int]):
        self.libpath = libpath
        self.psi0 = psi0
        self.id = id
        self.tfuncs = np.concatenate(tfuncs) if len(tfuncs) > 0 else np.zeros(0)
        self.correlation_time = correlation_time if correlation_time is not None else -1
        self.output = np.zeros(output_shape, dtype = np.complex128)

    def __call__(self):
        coeffs = np.ascontiguousarray(self.psi0.coeffs, dtype = np.complex128)
        tfuncs = np.ascontiguousarray(self.tfuncs, dtype = np.complex128)
        lib = ctypes.CDLL(self.libpath)

        lib.run_trajectories(
            ctypes.c_double(self.psi0.j),
            ctypes.c_uint64(self.id),
            coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tfuncs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
	    ctypes.c_double(self.correlation_time),
	    self.output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ) 



def mcsolve(system: Dicke, psi0: DickeState, tlist: list[float], e_ops = [], ntraj: int = 0, ncpu: int = 0, correlation_time: int = None,
            jtol: float = 0.05, stol: float = 1e-20, rkpoly: int = 4,
	    enable_displacement: bool = False) -> MCSolveResult:

    if psi0.j > system.N/2:
        raise ValueError(f"J spin length is larger than N/2, where N = {system.N}")

    if not math.isclose(psi0.norm(), 1):
        raise ValueError(f"Initial wavefunction is not normalized, got norm of {psi0.norm()}")

    tlist = np.array(tlist)
    assert len(tlist) > 1

    dt = tlist[1] - tlist[0]
    assert np.isclose(tlist[1:] - tlist[:-1], dt).all() and dt > 0, "tlist must be monotonically increasing and linearly spaced"

    spin_dim, boson_dim = validate_dimension(system.hamiltonian)

    if spin_dim is None:
        raise ValueError(f"Hamiltonian contains no spin operators")

    if boson_dim is None:
        boson_dim = 1 # must have at least one, even just for free spins

    if enable_displacement:
        if boson_dim > 20:
            raise ValueError(f"Displacement should use a small photon truncation!")
        if system.cavity_absorption != 0 or system.cavity_emission != 0:
            raise ValueError(f"Displacement does not yet support these decay channels")

    tfunc_dt = 0.05 / system.N
    tfunc_points = int((tlist[-1] - tlist[0]) / tfunc_dt)
    tfunc_tlist = np.linspace(tlist[0], tlist[-1], tfunc_points)

    code, spin_width, boson_width, tfuncs = c.generate_backend_code(system.hamiltonian, e_ops, tfunc_tlist, displace = enable_displacement, two_time_correlation = correlation_time is not None)
    config = c.generate_config(system, boson_dim, tlist, len(e_ops), ntraj, ncpu, jtol, stol, spin_width, boson_width, len(tlist), rkpoly, tfunc_points, enable_displacement)

    with open("pimcs/c_backend/tmp.h", 'w') as handle:
        handle.write(code)

    with open("pimcs/c_backend/tmpconfig.h", 'w') as handle:
        handle.write(config)

    if correlation_time is not None:
        ntraj *= 4

    print("Building optimized executable...")
    libpath, hash_id = c.build_executable()
    solver = MCSolver(libpath, hash_id, psi0, tfuncs, correlation_time, ( ntraj, len(e_ops), len(tlist) ))
    solver()

    expect = np.sum(solver.output, axis = 0) / ntraj
    expect = [ (e.real if op.is_herm() and correlation_time is None else e) for e, op in zip(expect, e_ops) ]

    return MCSolveResult(expect)

