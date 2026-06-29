# Permutation Invariant Monte Carlo Solver (pimcs)

A small library to do quantum trajectories for permutational invariant spin-1/2 ensembles,
along with an associated bosonic mode, e.g. Dicke model.

The interface is designed to be a mesh of QuTiP `piqs` and `mcsolve` submodules.

> [!WARNING]
> The library is still in beta-testing phase and currently produces debug files. Displacement
> operator transform is currently being added.

**Requirements**
- POSIX compliant system (macOS or Linux)
- C11 compliant C compiler (gcc 5+ or clang 4+)
- Python `numpy` and `scipy` libraries

Check out the [example notebook](example_notebook.ipynb) as a quick reference.

**TODO:**
- [x] Relax C backend requirements to C11 (atomics) for better compatibility
- [x] Add unified interface for single `import pimcs`
- [x] Allow C code to be stopped from Python / Jupyter notebook
- [x] Add support for initial states not in maximal J sector
- [x] Support arbitrary Hamiltonians
- [x] Add support for time-dependent Hamiltonians
- [x] Auto-detection of Hermitian observables: give real arrays instead of always complex valued
- [x] Add displaced trajectory code (derive EOMs for arbitrary H)
- [ ] Add support for two-time correlations to Python frontend
- [ ] Implement displaced trajectories for higher-than-quadratic Hamiltonians
- [ ] Add Cython backend and code generation
- [ ] Promote operators to be Qobj operators for better QuTiP integration (using opaque data field)
- [ ] Add support for multiple spin spaces
- [ ] Add support for multiple boson modes
- [ ] Explore idea for displaced Holstein-Primakoff
