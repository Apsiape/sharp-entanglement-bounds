"""Attacks on the auxiliary lemmas: standard form, consistency gives state commutation, rounding, rotating the
state, the replacement estimate, sector states, the entropy lemma and the entropy of a sector."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math

import numpy as np
from scipy.optimize import minimize

from common import (Report, comm, entropy_bits, h2, herm, hs, perturbed_unitary, random_state, rel_entropy_bits,
                    rng, round_unitary, sgn_inv, sqrtm_psd, unitary)

KINDS = ["random", "geometric", "lowrank", "nearflat", "pure", "diag"]


def random_pure(da, db, g):
    v = g.normal(size=(da, db)) + 1j * g.normal(size=(da, db))
    return v / np.linalg.norm(v)


def standard_form(rep, g):
    worst_a = worst_b = 0.0
    for t in range(200):
        da, db = int(g.integers(1, 5)), int(g.integers(1, 5))
        psi = random_pure(da, db, g)  # psi[a, b]
        if t % 3 == 0:  # reduced Schmidt rank
            psi = psi[:, :1] @ psi[:1, :] if min(da, db) > 1 else psi
            psi = psi / np.linalg.norm(psi)
        rho_b = psi.T @ psi.conj()
        xi = sqrtm_psd(rho_b)
        y = g.normal(size=(db, db)) + 1j * g.normal(size=(db, db))
        lhs = np.linalg.norm(psi @ y.T)  # (I (x) Y) psi as an array
        worst_a = max(worst_a, abs(lhs - hs(y @ xi)))
        a = herm(g.normal(size=(da, da)) + 1j * g.normal(size=(da, da)))
        # the operator A~ of the proof, built from a Schmidt decomposition
        u, s, vh = np.linalg.svd(psi)
        r = int((s > 1e-12).sum())
        e, f = u[:, :r], vh[:r, :].T  # psi = sum s_k e_k (x) f_k
        alk = e.conj().T @ a @ e  # A_{lk} = <e_l, A e_k>
        at = np.zeros((db, db), complex)
        for k in range(r):
            for l in range(r):
                at += alk[l, k] * np.outer(f[:, k], f[:, l].conj())
        lhs2 = hs(y @ xi - xi @ at)
        rhs2 = np.linalg.norm(a @ psi - psi @ y.T)
        worst_b = max(worst_b, lhs2 - rhs2)
        worst_b = max(worst_b, hs(at - at.conj().T))
    rep.check("standard form (a): | ||(I x Y)psi|| - ||Y xi|| |, 200 instances", f"{worst_a:.2e}", 0, worst_a < 1e-10)
    rep.check("standard form (b): max of ||Y xi - xi A~|| - ||(A x I - I x Y)psi||", f"{worst_b:.2e}", 0,
              worst_b < 1e-10)


def consistency(rep, g):
    worst = 0.0
    for t in range(400):
        d = int(g.integers(2, 6))
        rho = random_state(d, g, KINDS[t % len(KINDS)])
        xi = sqrtm_psd(rho)
        b = sgn_inv(herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))
        scale = [0.1, 1.0, 3.0][t % 3]  # T of any norm, including norm above 1
        tt = scale * herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d)))
        if t % 4 == 0:
            tt = b + 0.1 * tt  # near-consistent instances
        theta = hs(b @ xi - xi @ tt)
        worst = max(worst, hs(comm(b, xi)) / (math.sqrt(2) * theta))
    # optimizer: maximize the ratio over T for fixed B, rho
    d = 3
    best = 0.0
    for start in range(4):
        rho = random_state(d, g, "geometric")
        xi = sqrtm_psd(rho)
        b = sgn_inv(herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))

        def neg(p):
            tt = herm(p[:9].reshape(3, 3) + 1j * p[9:].reshape(3, 3))
            return -hs(comm(b, xi)) / (math.sqrt(2) * max(hs(b @ xi - xi @ tt), 1e-12))

        res = minimize(neg, g.normal(size=18), method="Nelder-Mead", options={"maxiter": 4000})
        best = max(best, -res.fun)
    rep.check("consistency: max ||[B,xi]|| / (sqrt2 ||B xi - xi T||), 400 random", f"{worst:.4f}", 1, worst <= 1 + 1e-9)
    rep.check("consistency: optimizer over T, max ratio", f"{best:.4f}", 1, best <= 1 + 1e-9)
    xi = np.diag([1.0, 0.0]).astype(complex)
    bx = np.array([[0, 1], [1, 0]], complex)
    rep.check("consistency: the sharp example has ratio 1", f"{hs(comm(bx, xi)) / (math.sqrt(2) * hs(bx @ xi)):.6f}",
              1, abs(hs(comm(bx, xi)) / (math.sqrt(2) * hs(bx @ xi)) - 1) < 1e-12)


def rounding_rotation(rep, g):
    w1 = w2 = w3 = 0.0
    for t in range(300):
        d = int(g.integers(2, 7))
        rho = random_state(d, g, KINDS[t % len(KINDS)])
        xi = sqrtm_psd(rho)
        base = np.diag(g.choice([-1.0, 1.0], size=d)).astype(complex)
        v = unitary(d, g)
        u = perturbed_unitary(v @ base @ v.conj().T, [0.0, 0.1, 0.5, 2.0][t % 4], g)
        r = round_unitary(u)
        sq = hs((u @ u - np.eye(d)) @ xi)
        if sq > 1e-12:
            w1 = max(w1, hs((r - u) @ xi) / sq)
        w2 = max(w2, hs(comm(r, xi)) - hs(comm(u, xi)) - 2 * sq)
        lhs = np.abs(np.linalg.eigvalsh(herm(u.conj().T @ rho @ u - rho))).sum()
        c = hs(comm(u, xi))
        if c > 1e-12:
            w3 = max(w3, lhs / (2 * c))
    rep.check("rounding: max ||(r(U)-U)xi|| / ||(U^2-I)xi||", f"{w1:.4f}", 1, w1 <= 1 + 1e-9)
    rep.check("rounding: max ||[r(U),xi]|| - ||[U,xi]|| - 2||(U^2-I)xi||", f"{w2:.2e}", 0, w2 <= 1e-10)
    rep.check("rotating the state: max ||U*rho U - rho||_1 / (2||[U,xi]||)", f"{w3:.4f}", 1, w3 <= 1 + 1e-9)


def replacement(rep, g):
    worst = 0.0
    for t in range(200):
        d, m = int(g.integers(2, 5)), int(g.integers(1, 5))
        rho = random_state(d, g, KINDS[t % len(KINDS)])
        xi = sqrtm_psd(rho)
        yp = [unitary(d, g) for _ in range(m)]
        y = []
        for u in yp:
            a = u + 0.3 * (g.normal(size=(d, d)) + 1j * g.normal(size=(d, d)))
            y.append(a / max(1.0, np.linalg.norm(a, 2)))  # contractions
        lhs = hs((np.linalg.multi_dot(y + [np.eye(d)]) - np.linalg.multi_dot(yp + [np.eye(d)])) @ xi)
        rhs = sum(hs((y[l] - yp[l]) @ xi) + 2 * sum(hs(comm(yp[k], xi)) for k in range(l + 1, m)) for l in range(m))
        worst = max(worst, lhs / rhs)
    rep.check("replacement estimate: max ratio, 200 instances", f"{worst:.4f}", 1, worst <= 1 + 1e-9)


def sector(rep, g):
    wa = wb1 = wb2 = wc1 = wc2 = 0.0
    wd = -1.0
    for t in range(300):
        d = int(g.integers(2, 7))
        rho = random_state(d, g, KINDS[t % len(KINDS)])
        xi = sqrtm_psd(rho)
        v = unitary(d, g)
        jp = v @ np.diag(g.choice([-1.0, 1.0], size=d)).astype(complex) @ v.conj().T
        if t % 2 == 0:  # nearly commuting with the state
            jp = sgn_inv(jp + 3 * rho)
        pm = (np.eye(d) - jp) / 2
        pp = (np.eye(d) + jp) / 2
        p = np.trace(pm @ rho).real
        if p < 1e-6:
            continue
        rm = pm @ rho @ pm / p
        xm = sqrtm_psd(rm)
        eta = pm @ xi @ pm / math.sqrt(p)
        off = hs(pp @ xi @ pm)
        cj = hs(comm(jp, xi))
        wa = max(wa, hs(xm - eta) - off / math.sqrt(p), off - cj / 2)
        r = g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))
        on = np.linalg.norm(r, 2)
        wb1 = max(wb1, hs(r @ xm) - (hs(r @ pm @ xi) + on * off) / math.sqrt(p))
        wb2 = max(wb2, hs(r @ xm) - (hs(r @ xi) + on * cj) / math.sqrt(p))
        y = sgn_inv(herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))  # an involution
        wc1 = max(wc1, hs(comm(y, xm)) - (2 * hs(comm(y, pm) @ xi) + hs(comm(y, xi)) + 2 * off) / math.sqrt(p))
        u = unitary(d, g)
        wc2 = max(wc2, hs(comm(u, xm)) - (hs(comm(u, jp) @ xi) + 3 * hs(comm(u, xi)) + cj) / math.sqrt(p))
        lhs = entropy_bits(rho) - (p * entropy_bits(rm) - h2(min(cj ** 2 / 4, 0.5)))
        wd = max(wd, -lhs)
    rep.check("sector states (a): max excess", f"{wa:.2e}", 0, wa <= 1e-9)
    rep.check("sector states (b), first form: max excess", f"{wb1:.2e}", 0, wb1 <= 1e-9)
    rep.check("sector states (b), second form: max excess", f"{wb2:.2e}", 0, wb2 <= 1e-9)
    rep.check("sector states (c), involutions: max excess", f"{wc1:.2e}", 0, wc1 <= 1e-9)
    rep.check("sector states (c), unitaries: max excess", f"{wc2:.2e}", 0, wc2 <= 1e-9)
    rep.check("entropy of a sector: max excess of the right side", f"{wd:.2e}", 0, wd <= 1e-9)


def js_bound(rep, g):
    worst = -1.0
    for t in range(1500):
        d, k = int(g.integers(1, 5)), int(g.integers(1, 5))
        l1 = g.normal(size=(k, d)) + 1j * g.normal(size=(k, d))
        l1 /= np.linalg.norm(l1)
        if t % 3 == 0:
            l2 = l1 + [0.01, 0.1, 0.5][t % 9 // 3] * (g.normal(size=(k, d)) + 1j * g.normal(size=(k, d)))
        else:
            l2 = g.normal(size=(k, d)) + 1j * g.normal(size=(k, d))
        l2 /= np.linalg.norm(l2)
        rho, sig = l1 @ l1.conj().T, l2 @ l2.conj().T
        lhs = rel_entropy_bits(rho, (rho + sig) / 2)
        rhs = h2(min(np.linalg.norm(l1 - l2) ** 2 / 4, 0.5))
        worst = max(worst, lhs - rhs)
    best = -1.0
    for start in range(6):
        d, k = 2, 3

        def neg(p):
            a = p[:k * d].reshape(k, d) + 1j * p[k * d:2 * k * d].reshape(k, d)
            b = p[2 * k * d:3 * k * d].reshape(k, d) + 1j * p[3 * k * d:].reshape(k, d)
            a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
            r1, r2 = a @ a.conj().T, b @ b.conj().T
            return -(rel_entropy_bits(r1, (r1 + r2) / 2) - h2(min(np.linalg.norm(a - b) ** 2 / 4, 0.5)))

        res = minimize(neg, g.normal(size=4 * 3 * 2), method="Nelder-Mead", options={"maxiter": 6000})
        best = max(best, -res.fun)
    rep.check("entropy lemma: max D(rho||(rho+sigma)/2) - bound, 1500 random", f"{worst:.2e}", 0, worst <= 1e-9)
    rep.check("entropy lemma: optimizer (6 starts), max excess", f"{best:.2e}", 0, best <= 1e-7)
    # the termwise bound in an unequal-norm basis fails: one column with b = (1 - sqrt x) a
    x = 1e-4
    col = math.log2(2 / (1 + (1 - math.sqrt(x)) ** 2))
    rep.check("unequal-norm column: termwise entropy exceeds h(x/4)", f"{col:.4e} > {h2(x / 4):.4e}", "fails",
              col > h2(x / 4))


def main():
    rep = Report("Auxiliary lemmas")
    g = rng(7)
    standard_form(rep, g)
    consistency(rep, g)
    rounding_rotation(rep, g)
    replacement(rep, g)
    sector(rep, g)
    js_bound(rep, g)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
