"""Attacks on the return direction (Section 4.1; proved in the companion paper, Theorem D): the sequential-measurement lemma, parity repair, the loss bound
of the return theorem on perturbed state-weighted models of the magic square game, and the negative control."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import math

import numpy as np
import scipy.linalg as sl

from common import (PX, PY, PZ, I2, Report, comm, herm, hs, perturbed_unitary, purification, random_state, rng,
                    round_unitary, sgn_inv, sqrtm_psd, unitary)

SQ2 = math.sqrt(2)


def beta(l):
    return 3 * (1 + (l - 1) * (l - 2) / 2) + 13 * (l - 1) / SQ2


def c_s(s):
    return 0.25 * (beta(s) ** 2 + (s * s + 2 + sum(beta(l) for l in range(1, s + 1))) ** 2)


def sequential(bs, rho, sign):
    """Exact simulation of Alice's sequential measurement of B_j^T on the canonical purification.
    Returns d_j (exact), disagreement probabilities before and after repair, and the bad-parity probability."""
    d, k = len(rho), len(bs)
    psi = purification(rho)  # psi[a, b]; (Q x I) psi <-> Q psi, (I x Y) psi <-> psi Y^T
    branches = [((), psi)]
    for b in bs:
        q = b.T
        new = []
        for rec, vec in branches:
            for s in (1, -1):
                new.append((rec + (s,), ((np.eye(d) + s * q) / 2) @ vec))
        branches = new
    dj2 = np.zeros(k)
    dis = np.zeros(k)
    dis_rep = np.zeros(k)
    bad = 0.0
    for rec, vec in branches:
        wrong = math.prod(rec) != sign
        if wrong:
            bad += np.linalg.norm(vec) ** 2
        rep_rec = list(rec)
        if wrong:
            rep_rec[-1] *= -1
        for j, b in enumerate(bs):
            dj2[j] += np.linalg.norm(rec[j] * vec - vec @ b.T) ** 2
            # probability that Bob's outcome of B_j differs from the recorded sign
            dis[j] += np.linalg.norm(vec @ ((np.eye(d) - rec[j] * b) / 2).T) ** 2
            dis_rep[j] += np.linalg.norm(vec @ ((np.eye(d) - rep_rec[j] * b) / 2).T) ** 2
    return np.sqrt(dj2), dis, dis_rep, bad


def lemma_checks(rep, g):
    worst_a = worst_p = worst_b = 0.0
    kinds = ["random", "geometric", "lowrank", "nearflat", "pure", "diag"]
    for t in range(240):
        d, k = int(g.choice([2, 3, 4, 6])), int(g.integers(1, 5))
        rho = random_state(d, g, kinds[t % 6])
        xi = sqrtm_psd(rho)
        if t % 2:  # nearly commuting observables that nearly commute with the state
            base = np.diag(g.choice([-1.0, 1.0], size=d)).astype(complex)
            rho = np.diag(g.dirichlet(np.ones(d))).astype(complex)
            xi = sqrtm_psd(rho)
            bs = [sgn_inv(np.diag(g.choice([-1.0, 1.0], size=d)) + [1e-3, 0.05, 0.3][t % 3] *
                          herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d)))) for _ in range(k)]
        else:
            bs = [sgn_inv(herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d)))) for _ in range(k)]
        sign = int(g.choice([-1, 1]))
        dj, dis, _, bad = sequential(bs, rho, sign)
        cs = [hs(comm(b, xi)) for b in bs]
        for j in range(k):
            bound = cs[j] + sum(hs(comm(bs[j], bs[l]) @ xi) / SQ2 + sum(cs[:l]) for l in range(j))
            worst_a = max(worst_a, dj[j] - bound)
            worst_p = max(worst_p, abs(dis[j] - dj[j] ** 2 / 4))
        r = hs((np.linalg.multi_dot(bs + [np.eye(d)]) - sign * np.eye(d)) @ xi)
        worst_b = max(worst_b, bad - (r + dj.sum()) ** 2 / 4)
    rep.check("sequential measurements (a): max d_j - bound, 240 instances", f"{worst_a:.2e}", 0, worst_a <= 1e-10)
    rep.check("sequential measurements (a): |Pr[a_j != b_j] - d_j^2/4|", f"{worst_p:.2e}", 0, worst_p <= 1e-10)
    rep.check("sequential measurements (b): max bad parity - bound", f"{worst_b:.2e}", 0, worst_b <= 1e-10)


def magic_square():
    xx = lambda a, b: np.kron(a, b)
    ops = [xx(PX, I2), xx(I2, PX), xx(PX, PX),
           xx(I2, PZ), xx(PZ, I2), xx(PZ, PZ),
           -xx(PX, PZ), -xx(PZ, PX), xx(PY, PY)]
    eqs = [([0, 1, 2], 0), ([3, 4, 5], 0), ([6, 7, 8], 0), ([0, 3, 6], 1), ([1, 4, 7], 1), ([2, 5, 8], 1)]
    return ops, eqs


def model_error(us, uj, eqs, rho):
    """The precision e at which (phi, rho) satisfies the three conditions for the solution group presentation."""
    d = len(rho)
    xi = sqrtm_psd(rho)
    eye = np.eye(d)
    errs = []
    for u in us:
        errs.append(hs((u @ u - eye) @ xi))
        errs.append(hs((u @ uj @ u.conj().T @ uj.conj().T - eye) @ xi))
        errs.append(hs(comm(u, xi)))
    errs.append(hs((uj @ uj - eye) @ xi))
    errs.append(hs(comm(uj, xi)))
    errs.append(hs((uj + eye) @ xi))
    for vs, bi in eqs:
        for j, l in itertools.combinations(vs, 2):
            errs.append(hs((us[j] @ us[l] @ us[j].conj().T @ us[l].conj().T - eye) @ xi))
        prod = np.linalg.matrix_power(uj, bi) @ np.linalg.multi_dot([us[j] for j in vs])
        errs.append(hs((prod - eye) @ xi))
    return max(errs)


def return_theorem(rep, g):
    ops, eqs = magic_square()
    worst = 0.0
    n = 0
    for t in range(48):
        k = [1, 2, 3][t % 3]
        sigma = random_state(k, g, ["random", "geometric", "diag"][t % 3]) if k > 1 else np.eye(1, dtype=complex)
        rho = np.kron(np.eye(4) / 4, sigma)
        eta = [0.0, 1e-4, 1e-3, 1e-2, 0.03, 0.1][t % 6]
        us = [perturbed_unitary(np.kron(o, np.eye(k)), eta, g) for o in ops]
        uj = perturbed_unitary(-np.eye(4 * k, dtype=complex), eta, g)
        if t % 4 == 3:  # rotate the state slightly, so that it no longer commutes exactly
            u = sl.expm(1j * eta * herm(g.normal(size=(4 * k, 4 * k)) + 1j * g.normal(size=(4 * k, 4 * k))))
            rho = u @ rho @ u.conj().T
        e = model_error(us, uj, eqs, rho)
        bs = [round_unitary(u) for u in us]
        loss = 0.0
        for vs, bi in eqs:
            _, _, dis_rep, _ = sequential([bs[j] for j in vs], rho, (-1) ** bi)
            loss = max(loss, dis_rep.max())
        if e > 0:
            worst = max(worst, loss / (c_s(3) * e ** 2))
        else:
            worst = max(worst, 0.0 if loss < 1e-12 else float("inf"))
        n += 1
    rep.info(f"C_3 = {c_s(3):.3f}")
    rep.check(f"return theorem, {n} perturbed magic-square models: max pair loss / (C_3 e^2)", f"{worst:.2e}", 1,
              worst <= 1)


def negative_control(rep):
    z = PZ.copy()
    plus = np.array([1, 1], complex) / SQ2
    rho = np.outer(plus, plus.conj())
    _, _, dis_rep, bad = sequential([z, z], rho, 1)
    rep.check("negative control: state commutators sqrt 2", f"{hs(comm(z, rho)):.6f}", f"{SQ2:.6f}",
              abs(hs(comm(z, rho)) - SQ2) < 1e-12)
    rep.check("negative control: the construction loses with probability 1/2", f"{dis_rep.max():.6f}", 0.5,
              abs(dis_rep.max() - 0.5) < 1e-12 and bad < 1e-12)


def main():
    rep = Report("Return direction")
    g = rng(11)
    lemma_checks(rep, g)
    return_theorem(rep, g)
    negative_control(rep)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
