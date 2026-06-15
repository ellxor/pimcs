# Permutation Invariant Monte Carlo Solver (pimcs)

A small library to do quantum trajectories for permutational invariant spin-1/2 ensembles,
along with an associated bosonic mode, e.g. Dicke model.

> [!WARNING]
> The library is still in beta-testing phase and many parts are not fully implemented,
> and it currently creates debug files. It requires a UNIX based OS (macOS or Linux)
> and C11 compliant C compiler (gcc 5+ or clang 4+)

The interface is designed to be a mesh of QuTiP `piqs` and `mcsolve` submodules.

Check out the [example notebook](example.ipynb) as a quick reference.

**TODO:**
- [x] Relax C backend requirements to C11 (atomics) for better compatibility
- [x] Add unified interface for single `import pimcs`
- [ ] Add more graceful way for c backend to be killed from Python frontend - currently requires killing by PID
- [ ] Add support for initial states not in maximal J sector
- [x] Add support for non-displaced trajectories to Python frontend
- [ ] Add support for two-time correlations to Python frontend
- [ ] Support fully quadratic Hamiltonians in the bosonic mode: (a)^2 and (a†)^2 terms
- [ ] Add Cython backend and code generation
- [ ] Promote operators to be Qobj operators for better QuTiP integration (using opaque data field)
- [ ] Auto-detection of Hermitian observables: give real arrays instead of always complex valued
- [ ] Add support for multiple spin spaces
- [ ] Add support for multiple boson modes
- [ ] Explore idea for displaced Holstein-Primakoff
