# Permutation Invariant Monte Carlo Solver (pimcs)

A small library to do quantum trajectories for permutational invariant spin-1/2 ensembles,
along with an associated bosonic mode, e.g. Dicke model.

The interface is designed to be a mesh of QuTiP `piqs` and `mcsolve` submodules.

**Requirements**
- POSIX compliant system (macOS and Linux tested)
- C11 compliant C compiler (gcc 5+ or clang 4+)
- Python `numpy` and `scipy` libraries

Check out the [example notebook](example_notebook.ipynb) as a quick reference.

**TODO:**
- Upload code for two-time correlations
- Add proper documentation/wiki page
