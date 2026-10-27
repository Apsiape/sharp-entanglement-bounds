"""Attacks on the qubit count in an arbitrary state (Section 3; proved in the companion paper, Theorem C): parts (i), (ii), (iii),
the uniform corollary, and the example showing that the state commutators are needed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from common import (Report, acomm, comm, entropy_bits, h2, herm, hs, paulis, perturbed_involution, random_state, rng,
                    sqrtm_psd)


def products(zs):
    m, d = len(zs), zs[0].shape[0]
    out = {}
    for g in itertools.product((0, 1), repeat=m):
        f = np.eye(d, dtype=complex)
        for gi, z in zip(g, zs):
            if gi:
                f = f @ z
        out[g] = f
    return out


def lhs_rhs(xs, zs, rho):
    """Returns per-pair (a^2, rhs_i, b^2, rhs_ii) and (S, bracket sum)."""
    m = len(zs)
    xi = sqrtm_psd(rho)
    fs = products(zs)
    cz = [hs(comm(z, xi)) for z in zs]
    cx = [hs(comm(x, xi)) for x in xs]
    rows, brackets = [], 0.0
    for i in range(m):
        a2 = b2 = 0.0
        for g, f in fs.items():
            ge = list(g)
            ge[i] ^= 1
            a2 += hs((fs[tuple(ge)] - f @ zs[i]) @ xi) ** 2
            sgn = -1.0 if g[i] else 1.0
            b2 += hs((sgn * xs[i] @ f - f @ xs[i]) @ xi) ** 2
        a2 /= len(fs)
        b2 /= len(fs)
        rhs1 = sum(2 * (hs(comm(zs[i], zs[j]) @ xi) + cz[j]) ** 2 + 2 * cz[j] ** 2 for j in range(i + 1, m))
        rhs2 = 2 * (hs(acomm(xs[i], zs[i]) @ xi) + cz[i]) ** 2
        rhs2 += sum(2 * (hs(comm(xs[i], zs[j]) @ xi) + cz[j]) ** 2 for j in range(m) if j != i)
        rhs2 += sum(2 * c ** 2 for c in cz)
        a, b = math.sqrt(a2), math.sqrt(b2)
        brackets += 1 - (b + cx[i]) ** 2 / (2 * math.log(2)) - h2(min((a + cz[i]) ** 2 / 4, 0.5))
        rows.append((a2, rhs1, b2, rhs2))
    return rows, entropy_bits(rho), brackets


def random_trials(rep, g):
    worst1 = worst2 = 0.0
    min_gap = float("inf")
    n = 0
    kinds = ["random", "geometric", "lowrank", "nearflat", "pure", "diag"]
    for m in (1, 2, 3):
        for extra in (1, 2):
            x0, z0 = paulis(m, extra)
            d = x0[0].shape[0]
            for trial in range(18):
                eta = [0.0, 0.05, 0.2, 0.5, 0.8, 1.5][trial % 6]
                xs = [perturbed_involution(x, eta, g) for x in x0]
                zs = [perturbed_involution(z, eta, g) for z in z0]
                rho = random_state(d, g, kinds[trial % len(kinds)])
                rows, s, br = lhs_rhs(xs, zs, rho)
                for a2, r1, b2, r2 in rows:
                    if r1 > 1e-12:
                        worst1 = max(worst1, a2 / r1)
                    elif a2 > 1e-10:
                        worst1 = float("inf")
                    if r2 > 1e-12:
                        worst2 = max(worst2, b2 / r2)
                    elif b2 > 1e-10:
                        worst2 = float("inf")
                min_gap = min(min_gap, s - br)
                n += 1
    rep.check(f"part (i), {n} random instances, max a_i^2 / bound", f"{worst1:.4f}", 1, worst1 <= 1 + 1e-9)
    rep.check(f"part (ii), {n} random instances, max b_i^2 / bound", f"{worst2:.4f}", 1, worst2 <= 1 + 1e-9)
    rep.check(f"part (iii), {n} random instances, min S(rho) - sum of brackets", f"{min_gap:.3e}", 0, min_gap >= -1e-9)


def unpack(p, m, d, x0, z0):
    k = 0
    xs, zs = [], []
    for base, out in ((x0, xs), (z0, zs)):
        for b in base:
            h = p[k:k + d * d].reshape(d, d)
            k += d * d
            hh = herm(h + 1j * h.T)
            v = expm(1j * hh)
            out.append(v @ b @ v.conj().T)
    a = p[k:k + d * d].reshape(d, d) + 1j * p[k + d * d:k + 2 * d * d].reshape(d, d)
    rho = a @ a.conj().T
    return xs, zs, rho / np.trace(rho).real


def optimizer_attacks(rep, g):
    m, extra = 2, 1
    x0, z0 = paulis(m, extra)
    d = x0[0].shape[0]
    npar = 2 * m * d * d + 2 * d * d
    worst = {"i": 0.0, "ii": 0.0}
    min_gap = float("inf")
    for start in range(4):
        p0 = 0.05 * g.normal(size=npar)
        # start near the exact Paulis with a non-flat state
        p0[2 * m * d * d:2 * m * d * d + d * d] += np.diag(np.linspace(1, 0.3, d)).ravel()

        def gap(p):
            xs, zs, rho = unpack(p, m, d, x0, z0)
            _, s, br = lhs_rhs(xs, zs, rho)
            return s - br

        res = minimize(gap, p0, method="L-BFGS-B", options={"maxiter": 60})
        min_gap = min(min_gap, res.fun)
        for part, idx in (("i", (0, 1)), ("ii", (2, 3))):
            def neg_ratio(p, idx=idx):
                xs, zs, rho = unpack(p, m, d, x0, z0)
                rows, _, _ = lhs_rhs(xs, zs, rho)
                return -max(r[idx[0]] / max(r[idx[1]], 1e-12) for r in rows)

            res = minimize(neg_ratio, p0 + 0.2 * g.normal(size=npar), method="L-BFGS-B", options={"maxiter": 60})
            worst[part] = max(worst[part], -res.fun)
    rep.check("part (i), optimizer (4 starts), max ratio", f"{worst['i']:.4f}", 1, worst["i"] <= 1 + 1e-9)
    rep.check("part (ii), optimizer (4 starts), max ratio", f"{worst['ii']:.4f}", 1, worst["ii"] <= 1 + 1e-9)
    rep.check("part (iii), optimizer (4 starts), min gap", f"{min_gap:.3e}", 0, min_gap >= -1e-7)


def necessity(rep):
    e = np.eye(4, dtype=complex)
    xi = np.outer(e[0], e[0])
    z1 = np.diag([1, -1, 1, -1]).astype(complex)
    z2 = np.zeros((4, 4), complex)
    z2[0, 0], z2[1, 1], z2[2, 3], z2[3, 2] = 1, -1, 1, 1
    z3 = np.zeros((4, 4), complex)
    z3[0, 2] = z3[2, 0] = z3[1, 3] = z3[3, 1] = 1
    zs = [z1, z2, z3]
    fs = products(zs)
    a2 = np.mean([hs((fs[(1 - g[0],) + g[1:]] - f @ z1) @ xi) ** 2 for g, f in fs.items()])
    rel = hs(comm(z1, z2) @ xi) + hs(comm(z1, z3) @ xi)
    rep.check("example: state-weighted relations of Z_1 vanish", f"{rel:.2e}", 0, rel < 1e-12)
    rep.check("example: a_1^2 equals 1", f"{a2:.6f}", 1, abs(a2 - 1) < 1e-12)
    rep.check("example: state commutator of Z_3 is sqrt 2", f"{hs(comm(z3, xi)):.6f}", f"{math.sqrt(2):.6f}",
              abs(hs(comm(z3, xi)) - math.sqrt(2)) < 1e-12)


def exact_pauli(rep):
    xs, zs = paulis(3)
    rows, s, br = lhs_rhs(xs, zs, np.eye(8) / 8)
    rep.check("exact Paulis on the flat state: S = sum of brackets = 3", f"{s:.6f}, {br:.6f}", 3,
              abs(s - 3) < 1e-9 and abs(br - 3) < 1e-9)


def main():
    rep = Report("Qubit count in an arbitrary state")
    g = rng(20261024)
    exact_pauli(rep)
    necessity(rep)
    random_trials(rep, g)
    optimizer_attacks(rep, g)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
