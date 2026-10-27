"""Exact checks of the numerical constants used in the proofs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
from fractions import Fraction

from common import Report, h2

SQ2 = math.sqrt(2)
LN2 = math.log(2)


def main():
    rep = Report("Constants in the proofs")
    c = rep.check
    # CHSH(n)
    c("one pair: 4(1+sqrt2)^2 sqrt2 < 33", f"{4 * (1 + SQ2) ** 2 * SQ2:.4f}", 33, 4 * (1 + SQ2) ** 2 * SQ2 < 33)
    c("CHSH: 6(33 + 2 sqrt2) <= 215", f"{6 * (33 + 2 * SQ2):.4f}", 215, 6 * (33 + 2 * SQ2) <= 215)
    a_coef = 6 * 297 + Fraction(996 * 215, 3)
    b_coef = 6 * 297 + Fraction(1388 * 215, 3)
    c("CHSH: 6*297 + 996*215/3 <= 73162", f"{float(a_coef):.2f}", 73162, a_coef <= 73162)
    c("CHSH: 6*297 + 1388*215/3 <= 101256", f"{float(b_coef):.2f}", 101256, b_coef <= 101256)
    c("CHSH: 2*14^2 = 392 (anticommutator 12 plus state commutator 2)", 2 * 14 ** 2, 392, 2 * 14 ** 2 == 392)
    cb = 3 * 101256 * 2 * SQ2 / LN2
    ca = 73162 * 3 * SQ2
    c("CHSH: 3*101256*2sqrt2/ln2 <= 1.24e6", f"{cb:.4e}", "1.24e6", cb <= 1.24e6)
    c("CHSH: 73162*3sqrt2 <= 3.11e5", f"{ca:.4e}", "3.11e5", ca <= 3.11e5)
    br = 1 - 0.124 - h2(0.0311)
    c("CHSH: 1 - 0.124 - h(0.0311) > 0.67", f"{br:.5f}", 0.67, br > 0.67)
    # the uniform corollary of the count
    c("count: (sqrt(10M)+1)^2 <= 16M for M = 2..10^5", "checked", "", all((math.sqrt(10 * m) + 1) ** 2 <= 16 * m
                                                                      for m in range(2, 100001)))
    b2 = 1 - 0.1 / (2 * LN2) - h2(0.025)
    c("count: 1 - 0.1/(2 ln2) - h(0.025) > 3/4", f"{b2:.5f}", 0.75, b2 > 0.75)
    # the configuration-lamp sector theorem; l = 21 N n >= 63, Lambda >= 14000 N^2 n
    ok = True
    for N in range(1, 400):
        n = max(3, N)
        l = 21 * N * n
        lam = 14000 * N * N * n
        ok &= 3 * (2 * l + 1) <= 7 * l
        ok &= 61 + 18 * l <= lam
        ok &= 4 * l * (1 + 2 * (2 * l + 8 + 30 * (4 * l + 1))) <= 984 * l * l
        ok &= 34 * l + 984 * l * l + 9 <= 1000 * l * l
        ok &= 666 * l <= lam
        gamma, gamma0 = 7 * l, 1000 * l * l  # in units of e
        ok &= 61 * lam ** 2 + 28 * gamma + 12 <= 63 * lam ** 2
        ok &= 5 * gamma0 + 15 * gamma + 6 <= 63 * lam ** 2
        ok &= 2 * gamma0 + 5 * gamma + 18 <= 63 * lam ** 2
        ok &= 30 <= gamma  # 30 e <= gamma, used for the anticommutator
    c("lamp sector theorem: the step inequalities for N = 1..399", ok, True, ok)
    c("lamp sector theorem: 16*252^2/(1.02e7) < 1/10", f"{16 * 252 ** 2 / 1.02e7:.4f}", 0.1, 16 * 252 ** 2 / 1.02e7 < 0.1)
    c("lamp sector theorem: 160*63504 <= 1.02e7", 160 * 63504, "1.02e7", 160 * 63504 <= 1.02e7)
    c("lamp sector theorem: g(1) = 1 + 4*23^2 = 2117", 1 + 4 * 23 ** 2, 2117, 1 + 4 * 23 ** 2 == 2117)
    q = 2 * 1071 - math.log2(1.02e7)
    c("lamp sector theorem: 2*1071 - log2(1.02e7) > 2117", f"{q:.3f}", 2117, q > 2117)
    c("lamp sector theorem: 4*1.02e7 <= 4.1e7", 4 * 1.02e7, "4.1e7", 4 * 1.02e7 <= 4.1e7)
    # the profile of K and the game
    c("profile of K: 5 pi + 19 pi/2 = 29 pi/2", f"{5 * math.pi + 9.5 * math.pi:.6f}", f"{14.5 * math.pi:.6f}", True)
    c("SV game: 4(9+3+3) = 60, 4(1+6) = 28, 28+12 = 40, 29*60 = 1740", (4 * 15, 4 * 7, 28 + 12, 29 * 60),
      (60, 28, 40, 1740), (4 * 15, 4 * 7, 28 + 12, 29 * 60) == (60, 28, 40, 1740))
    c("K profile upper bound: log2(sqrt2 pi + 1) < 3", f"{math.log2(SQ2 * math.pi + 1):.4f}", 3,
      math.log2(SQ2 * math.pi + 1) < 3)
    # entropy from transport
    g_half = math.log2(1 / math.tanh(0.25)) + 0.5 * 1.25 / LN2
    c("entropy: coth(1/4) <= 5, log2 5 < 7/3, (5/8)/ln2 < 15/16", f"{1 / math.tanh(0.25):.4f}", 5,
      1 / math.tanh(0.25) <= 5 and math.log2(5) < 7 / 3 and 0.625 / LN2 < 15 / 16)
    c("entropy: G_{1/2}(5/4) < 157/48", f"{g_half:.4f}", f"{157 / 48:.4f}", g_half < 157 / 48)
    c("entropy: 1 + 2*157/48 < 8", f"{1 + 2 * 157 / 48:.4f}", 8, 1 + 2 * 157 / 48 < 8)
    g_ninth = math.log2(1 / math.tanh(1 / 18)) + (1 / 9) * (33 / 4) / LN2
    c("K entropy: coth(1/18) <= 19, 19^4 < 2^17, (11/12)/ln2 < 11/8", f"{1 / math.tanh(1 / 18):.4f}", 19,
      1 / math.tanh(1 / 18) <= 19 and 19 ** 4 < 2 ** 17 and (11 / 12) / LN2 < 11 / 8)
    c("K entropy: G_{1/9}(33/4) < 17/4 + 11/8", f"{g_ninth:.4f}", f"{17 / 4 + 11 / 8:.4f}", g_ninth < 17 / 4 + 11 / 8)
    tot = Fraction(1) + Fraction(17, 4) + Fraction(11, 8) + Fraction(7, 3) + Fraction(15, 16)
    c("K entropy: 1 + 17/4 + 11/8 + 7/3 + 15/16 = 475/48 < 10", tot, "475/48", tot == Fraction(475, 48) and tot < 10)
    # the return theorem
    beta = [3 * (1 + (l - 1) * (l - 2) / 2) + 13 * (l - 1) / SQ2 for l in (1, 2, 3)]
    c3 = 0.25 * (beta[2] ** 2 + (11 + sum(beta)) ** 2)
    c("return theorem: C_3", f"{c3:.3f}", "about 788.2", abs(c3 - 788.167) < 1e-2)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
