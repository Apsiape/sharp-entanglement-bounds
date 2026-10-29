# The price of precision: usable entanglement in nonlocal games and the geometry of groups

Seth Douglas and Nidhal Mghirbi

Version 1.1.0, October 2026. Concept DOI: [10.5281/zenodo.23697345](https://doi.org/10.5281/zenodo.23697345).

The companion paper, *An explicit nonlocal game of quantum value one that needs exponential entanglement* (Nidhal Mghirbi and Seth Douglas, 2026, [10.5281/zenodo.23697255](https://doi.org/10.5281/zenodo.23697255)), proves the qubit count in an arbitrary state, state-weighted transfer and return for linear system games, the CHSH(n) and Slofstra-Vidick rates, and the explicit configuration-lamp game. This paper identifies every imported result at its use. Coefficient realization is supplied by Theorem 1.1 of [The uncomputable price of precision](https://doi.org/10.5281/zenodo.23736188); Appendix D maps the additional uses of that paper.

## Abstract

In a nonlocal game of value one, entanglement demand at loss $\varepsilon$ is the infimum of the one-way distillable entanglement of an original strategy state losing at most $\varepsilon$. For synchronous games with tested diagonals and constraint-variable consistency games, this demand diverges exactly when no finite-dimensional strategy is perfect. Every good enough mixed strategy yields any fixed number of Bell pairs at any fixed fidelity below one by local isometries. Under a promise, a sufficient loss is computable for rational synchronous or binary constraint games, but bounded demand is undecidable among games of value one. Agreement games have the same demand in both directions up to a factor three in loss. A sign relation and an even dilation, mirrored by both players, force a logarithmic lower bound. Theorem 1.1 of our companion "The uncomputable price of precision" (10.5281/zenodo.23736188) realizes every positive $\Delta^0_2$ logarithmic coefficient with a fixed game on the original mixed state. We classify the positive pure original-state entropy-coefficient spectrum as the positive $\Delta^0_2$ reals; the full usable spectrum remains open. The least usable entanglement compatible with $M$ approximate Pauli pairs at pairwise error $\kappa\le1/100$ is of order $\min\{M,\kappa^{-2}\}$. Group geometry gives logarithmic laws, an almost inverse-loss law, and a point-Pauli game with demand between $\varepsilon^{-\beta}$ and $\varepsilon^{-1/4}$, where $\beta=0.0734595360384268649\ldots$. Its lower bound uses explicit far-commutator fillings; a magnetic-cycle estimate obstructs improving the fixed-holonomy subdivision alone. Under exact group relations, demand counts qubits addressable by short words, with polynomial and exponential laws. Conditional on attainment and tail bounds, the $I_{3322}$ optimum needs bounded usable entanglement.

## Contents

| Path | Contents |
|---|---|
| `paper/` | the paper (`main.pdf`), its LaTeX source, and `abstract-arxiv.txt`, the plain-text abstract for submission metadata |
| `verify/` | numerical and exact checks of the inequalities and finite constructions, with their recorded output in `verify/logs/` |
| `verify/window/` | two free-word verifiers of the filling certificate set and their holonomy checks |
| `verify/recorded/` | recorded spectrum, floor, leakage and pairwise-budget diagnostics |
| `verify/artifacts/` | the larger exported objects (the point-Pauli Boolean game, the L4M8 and two-copy linear games) with their exact checkers, the selector and exact-source checks, and `SHA256SUMS` |

## Building and checking

Run `python paper/build.py`. It uses four plain pdflatex passes and BibTeX, with shell escape disabled, and writes intermediate outputs into a temporary build directory outside the repository and copies the final PDF and matching BBL into `paper/`.

The checks need Python 3 with NumPy and SciPy. Each runner executes its scripts one at a time, single-threaded:

- `cd verify && python -I run_all.py` runs the checks of the inequalities and of the tree-of-faces construction, including the spectrum, floor, budget and independent filling checks, in about four minutes;
- `cd verify && python -I run_artifacts.py` verifies the SHA-256 hashes of the exported objects and replays their checks, in about eighty seconds.

The standalone upload sources are in `paper/arxiv/`; they build with `pdflatex main` using the supplied `main.bbl`. The root bibliography is regenerated from `paper/refs.bib`.

The checks intentionally refresh recorded data: `check_window.py` rewrites `verify/window/certificates.json` and `engine_a_results.json` and `engine_b_results.json`; `check_refinements.py` rewrites `verify/recorded/refinements-results.json`; the runner refreshes `verify/logs/`. Engine A produces one certificate set, and two independently implemented verifiers check it; engine B verifies that same set. These are not two independent certificate-generation procedures.

Finite checks support the proofs in the paper; they do not replace them.

## License

The paper and documentation are licensed CC BY 4.0, and the scripts MIT or Apache 2.0, at your option; see
[RIGHTS.md](RIGHTS.md) and [NOTICE](NOTICE). To cite this work, see [CITATION.cff](CITATION.cff).
