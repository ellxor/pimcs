# Permutation Invariant Monte Carlo Solver (pimcs)

A small library to do quantum trajectories for permutational invariant spin-1/2 ensembles,
along with an associated bosonic mode, e.g. Dicke model.

This library is based on our work [Lloyd, Ziolkowska and Keeling, 2026](https://doi.org/10.48550/arXiv.2605.11103)
which contains results for N ~ 10⁴ emitters obtained using server resources.

The interface is designed to be a mesh of QuTiP `piqs` and `mcsolve` submodules. Check out
the [example notebook](example_notebook.ipynb) as a quick reference, this includes results
one can quickly obtain on a laptop.


**Requirements**
- POSIX compliant system (macOS and Linux tested)
- C11 compliant C compiler (gcc 5+ or clang 4+)
- Python `numpy` library

Check out the [example notebook](example_notebook.ipynb) as a quick reference.

**TODO:**
- Upload code for two-time correlations
- Add proper documentation/wiki page
