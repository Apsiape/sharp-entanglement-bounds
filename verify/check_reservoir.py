"""Checks of the tapered reservoir (Appendix A): the matched overlap is at least cos(pi/(L+1)), the best overlap over
local unitaries (sorted Schmidt spectra), and the resulting failure bound for the game."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math

import numpy as np

from common import Report


def shells(L):
    p = np.array([2 / (L + 1) * math.sin(math.pi * (m + 1) / (L + 1)) ** 2 for m in range(L)])
    return p


def spectra(L):
    p = shells(L)
    gamma = np.concatenate([np.full(2 ** m, p[m] / 2 ** m) for m in range(L)])  # squared Schmidt coefficients
    source = np.concatenate([gamma / 2, gamma / 2])  # EPR (x) Gamma
    target = gamma  # |11> (x) Gamma
    return p, gamma, source, target


def main():
    rep = Report("Tapered reservoir")
    ok_sum = ok_rank = ok_match = ok_best = ok_fail = True
    for L in (1, 2, 3, 5, 8, 12, 16):
        p, gamma, source, target = spectra(L)
        ok_sum &= abs(p.sum() - 1) < 1e-12 and abs(gamma.sum() - 1) < 1e-12
        ok_rank &= len(gamma) == 2 ** L - 1
        matched = sum(2 ** (m + 1) * math.sqrt(p[m] / 2 ** (m + 1) * p[m + 1] / 2 ** (m + 1)) for m in range(L - 1))
        ok_match &= matched >= math.cos(math.pi / (L + 1)) - 1e-12
        # best overlap over local unitaries: sum of sqrt products of the sorted spectra (von Neumann)
        s = np.sort(source)[::-1]
        t = np.sort(np.concatenate([target, np.zeros(len(source) - len(target))]))[::-1]
        best = float(np.sqrt(s * t).sum())
        ok_best &= best >= matched - 1e-12
        fail = 0.25 * 0.5 * (1 - ((1 + matched) / 2) ** 2)
        ok_fail &= fail <= math.pi ** 2 / (16 * (L + 1) ** 2) + 1e-15
        rep.info(f"L={L:2d}: matched overlap {matched:.6f} >= cos(pi/(L+1)) = {math.cos(math.pi / (L + 1)):.6f}; "
                 f"best {best:.6f}; failure {fail:.2e} <= {math.pi ** 2 / (16 * (L + 1) ** 2):.2e}")
    rep.check("shell weights sum to one", ok_sum, True, ok_sum)
    rep.check("Schmidt rank 2^L - 1", ok_rank, True, ok_rank)
    rep.check("matched overlap >= cos(pi/(L+1))", ok_match, True, ok_match)
    rep.check("best local-unitary overlap >= matched overlap", ok_best, True, ok_best)
    rep.check("game failure <= pi^2 / (16 (L+1)^2)", ok_fail, True, ok_fail)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
