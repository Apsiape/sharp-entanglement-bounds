# The price of precision: usable entanglement in nonlocal games and the geometry of groups

Seth Douglas and Nidhal Mghirbi

Version 1.0.0, October 2026. DOI: [10.5281/zenodo.23697346](https://doi.org/10.5281/zenodo.23697346).

The paper is split from its companion paper, *An explicit nonlocal game of quantum value one that needs exponential
entanglement* (Nidhal Mghirbi and Seth Douglas, 2026, doi:[10.5281/zenodo.23697255](https://doi.org/10.5281/zenodo.23697255)), released together with it; each theorem is proved in exactly
one of the two. The companion proves the qubit count in an arbitrary state, the state-weighted transfer and return for linear
system games, the CHSH(n) and Slofstra–Vidick rates, and the explicit configuration-lamp game; this paper states what it uses
from it, with pointers.

## Abstract

In a nonlocal game of quantum value one that no finite-dimensional strategy wins, near-perfect play needs growing dimension, which separable states can have. We ask how much usable entanglement it needs: the demand at loss $\varepsilon$ is the least one-way distillable entanglement of the original state of a strategy of loss at most $\varepsilon$. For synchronous games with tested diagonals and for constraint-variable consistency games, demand diverges exactly when no finite-dimensional strategy is perfect; then, for any number of Bell pairs and any fidelity below one, every sufficiently good mixed strategy yields that many at that fidelity by local isometries. For rational synchronous or binary constraint games, a loss guaranteeing this is computable under a promise, but boundedness of demand is undecidable among games of value one. Demand is at least logarithmic if near-winners have a sign relation and an even dilation mirrored by both players, as in a game from Abels' group. For agreement games the directions agree up to a factor three in loss, optimal among Bob-fixed deterministic local symmetrizations. The geometry of the group behind a game, not its presentation, sets the rates. Fixed games demand $\alpha\log(1/\varepsilon)$ for every $\alpha=\frac12\sum_i\log_{m_i}q_i$ with $q_i\ge2$ a power of two dividing $m_i$, a dense set containing all positive rationals and some transcendentals, such as $\frac12\log_62$; a configuration-lamp game demands $\varepsilon^{-1+o(1)}$; Pauli lamps on the points of Houghton's three rays give demand between $\varepsilon^{-0.0690}$ and $\varepsilon^{-1/4}$. If group relations hold exactly and Alice matches Bob to error $e$, demand counts qubits addressable by short words: $\Theta(1/e)$, $\Theta(e^{-k})$ for integers $k\ge1$, or $2^{\Theta(1/e)}$. Conditional on our attainment and tail bounds, the $I_{3322}$ optimum needs bounded usable entanglement.

## Contents

| Path | Contents |
|---|---|
| `paper/` | the paper (`main.pdf`), its LaTeX source, and `abstract-arxiv.txt`, the plain-text abstract for submission metadata |
| `verify/` | numerical and exact checks of the inequalities and finite constructions, with their recorded output in `verify/logs/` |
| `verify/artifacts/` | the larger exported objects (the point-Pauli Boolean game, the L4M8 and two-copy linear games) with their exact checkers, the selector and exact-source checks, and `SHA256SUMS` |

## Building and checking

The paper builds with `cd paper && latexmk -pdf main`, or with
`cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && pdflatex main`.

The checks need Python 3 with NumPy and SciPy. Each runner executes its scripts one at a time, single-threaded:

- `cd verify && python -I run_all.py` runs the checks of the inequalities and of the tree-of-faces construction, in about three minutes;
- `cd verify && python -I run_artifacts.py` verifies the SHA-256 hashes of the exported objects and replays their checks, in about eighty seconds.

Finite checks support the proofs in the paper; they do not replace them.

## License

The paper and documentation are licensed CC BY 4.0, and the scripts MIT or Apache 2.0, at your option; see
[RIGHTS.md](RIGHTS.md) and [NOTICE](NOTICE). To cite this work, see [CITATION.cff](CITATION.cff).
