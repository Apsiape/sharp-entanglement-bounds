"""Attacks on the spectral transport argument (Section 9; its dimension part is proved in the companion paper): the Wasserstein-1 facts on the circle, the spectral
coupling, the profile of K, entropy from transport, and the two examples."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math

import numpy as np
import scipy.linalg as sl
from scipy.optimize import linprog

from common import Report, comm, entropy_bits, herm, hs, perturbed_unitary, random_state, rng, sqrtm_psd

TWO_PI = 2 * math.pi


def geo(a, b):
    t = np.mod(np.abs(a - b), TWO_PI)
    return np.minimum(t, TWO_PI - t)


def w1(pa, wa, pb, wb):
    """Exact circle W1 of two discrete measures: min over c of the L1 norm of F_a - F_b - c (weighted median)."""
    pa, pb = np.mod(pa, TWO_PI), np.mod(pb, TWO_PI)
    pts = np.concatenate([pa, pb])
    wts = np.concatenate([wa, -wb])
    order = np.argsort(pts, kind="stable")
    pts, wts = pts[order], wts[order]
    cum = np.cumsum(wts)
    lengths = np.diff(np.concatenate([pts, [pts[0] + TWO_PI]]))
    o = np.argsort(cum)
    c_sorted, l_sorted = cum[o], lengths[o]
    half = l_sorted.sum() / 2
    c = c_sorted[np.searchsorted(np.cumsum(l_sorted), half)]
    return float((lengths * np.abs(cum - c)).sum())


def w1_lp(pa, wa, pb, wb):
    n, m = len(pa), len(pb)
    cost = geo(pa[:, None], pb[None, :]).ravel()
    a_eq, b_eq = [], []
    for i in range(n):
        row = np.zeros(n * m)
        row[i * m:(i + 1) * m] = 1
        a_eq.append(row)
        b_eq.append(wa[i])
    for j in range(m):
        row = np.zeros(n * m)
        row[j::m] = 1
        a_eq.append(row)
        b_eq.append(wb[j])
    res = linprog(cost, A_eq=np.array(a_eq), b_eq=np.array(b_eq), bounds=(0, None), method="highs")
    return float(res.fun)


def w1_haar(p, w):
    """Exact circle W1 between a discrete measure and Haar measure (convex minimization over the shift c)."""
    p = np.mod(p, TWO_PI)
    o = np.argsort(p)
    p, w = p[o], w[o]
    starts = np.concatenate([[0.0], p])
    ends = np.concatenate([p, [TWO_PI]])
    levels = np.concatenate([[0.0], np.cumsum(w)])

    def integral(c):
        tot = 0.0
        for s, e, f in zip(starts, ends, levels):
            if e <= s:
                continue
            # integrate |f - t/(2 pi) - c| for t in [s, e]; the integrand is linear in t
            g0, g1 = f - s / TWO_PI - c, f - e / TWO_PI - c
            if g0 * g1 >= 0:
                tot += (e - s) * abs(g0 + g1) / 2
            else:
                tz = s + (e - s) * g0 / (g0 - g1)
                tot += (tz - s) * abs(g0) / 2 + (e - tz) * abs(g1) / 2
        return tot

    lo, hi = -1.0, 1.0
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if integral(m1) <= integral(m2):
            hi = m2
        else:
            lo = m1
    return integral((lo + hi) / 2)


def lift(p, w):
    """The inverse-branch operator: each point goes to its two square roots with half the weight."""
    return np.concatenate([p / 2, p / 2 + math.pi]), np.concatenate([w, w]) / 2


def spectral(y, rho):
    t, z = sl.schur(y, output="complex")
    lam = np.diag(t)
    wts = np.real(np.diag(z.conj().T @ rho @ z))
    return np.mod(np.angle(lam), TWO_PI), np.clip(wts, 0, None)


def random_measure(g, kind):
    k = int(g.integers(1, 9))
    if kind == "random":
        p = g.uniform(0, TWO_PI, k)
    elif kind == "cluster":
        p = g.uniform(0, TWO_PI) + g.normal(0, 0.3, k)
    else:  # jittered lattice
        p = TWO_PI * np.arange(k) / k + g.normal(0, 0.3 / k, k)
    w = g.dirichlet(np.ones(k) * g.choice([0.2, 1.0, 5.0]))
    return p, w


def circle_facts(rep, g):
    e_lp = e_lift = e_ident = e_c = e_d = 0.0
    kinds = ["random", "cluster", "lattice"]
    for t in range(240):
        pa, wa = random_measure(g, kinds[t % 3])
        pb, wb = random_measure(g, kinds[(t + 1) % 3])
        if len(pa) * len(pb) <= 36:
            e_lp = max(e_lp, abs(w1(pa, wa, pb, wb) - w1_lp(pa, wa, pb, wb)))
        la, lwa = lift(pa, wa)
        lb, lwb = lift(pb, wb)
        e_lift = max(e_lift, w1(la, lwa, lb, lwb) - w1(pa, wa, pb, wb) / 2)
        dl, dlw = lift(2 * pa, wa)  # P(D_* mu)
        mix_p, mix_w = np.concatenate([pa, pa + math.pi]), np.concatenate([wa, wa]) / 2
        e_ident = max(e_ident, w1(dl, dlw, mix_p, mix_w))
        wh = w1_haar(pa, wa)
        e_c = max(e_c, wh - (w1(2 * pa, wa, pa, wa) + w1(pa + math.pi, wa, pa, wa)))
        e_d = max(e_d, math.pi / (2 * len(pa)) - wh)
    rep.check("circle W1: exact formula agrees with linear programming", f"{e_lp:.2e}", "1e-6 (solver tolerance)",
              e_lp < 1e-6)
    rep.check("inverse branches contract W1 by 1/2: max excess", f"{e_lift:.2e}", 0, e_lift < 1e-12)
    rep.check("P(D_* mu) = (mu + S_* mu)/2: max W1 distance", f"{e_ident:.2e}", 0, e_ident < 1e-12)
    rep.check("W1(mu,h) <= W1(D_* mu,mu) + W1(S_* mu,mu): max excess", f"{e_c:.2e}", 0, e_c < 1e-9)
    rep.check("d atoms are at W1 distance >= pi/(2d) from Haar: max excess", f"{e_d:.2e}", 0, e_d < 1e-9)


def couplings(rep, g):
    e_a = e_b = e_m = 0.0
    kinds = ["random", "geometric", "lowrank", "nearflat", "pure", "diag"]
    for t in range(240):
        d = int(g.integers(2, 7))
        rho = random_state(d, g, kinds[t % 6])
        xi = sqrtm_psd(rho)
        y = sl.expm(1j * herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))
        q = perturbed_unitary(y, [0.0, 0.05, 0.3, 1.0][t % 4], g)
        pa, wa = spectral(y, rho)
        pb, wb = spectral(q, rho)
        e_a = max(e_a, w1(pa, wa, pb, wb) - math.pi / 2 * hs(y @ xi - xi @ q))
        u = sl.expm(1j * [0.05, 0.3, 1.0][t % 3] * herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))
        pr, wr = spectral(y, u.conj().T @ rho @ u)
        e_b = max(e_b, w1(pr, wr, pa, wa) - math.pi * hs(comm(u, xi)))
        # the coupling identity: second moment of the joint spectral measure equals ||P xi - xi Q||^2
        tp, zp = sl.schur(y, output="complex")
        tq, zq = sl.schur(q, output="complex")
        mom = sum(abs(tp[i, i] - tq[j, j]) ** 2 * hs(np.outer(zp[:, i], zp[:, i].conj()) @ xi
                                                       @ np.outer(zq[:, j], zq[:, j].conj())) ** 2
                  for i in range(d) for j in range(d))
        e_m = max(e_m, abs(mom - hs(y @ xi - xi @ q) ** 2))
    rep.check("spectral coupling: second moment = ||P xi - xi Q||^2", f"{e_m:.2e}", 0, e_m < 1e-10)
    rep.check("spectral coupling: W1 <= (pi/2)||P xi - xi Q||, max excess", f"{e_a:.2e}", 0, e_a < 1e-10)
    rep.check("rotating the state: W1 <= pi ||[U,xi]||, max excess", f"{e_b:.2e}", 0, e_b < 1e-10)
    # W2 cannot replace W1 in the rotation step
    for delta in (1e-2, 1e-4):
        rho = np.diag([0.5 + delta, 0.5 - delta]).astype(complex)
        xi = sqrtm_psd(rho)
        ux = np.array([[0, 1], [1, 0]], complex)
        c = hs(comm(ux, xi))
        w2 = math.sqrt(2 * delta * 4)
        rep.check(f"W2 rotation counterexample, delta={delta:g}: W2 / ||[U,xi]||", f"{w2 / c:.1f}", "unbounded",
                  w2 / c > 10)


K_RELATORS = ["xyXYY", "xcXC", "yaYB", "ybYA", "Cab", "aa", "bb", "cc"]


def word(w, gens):
    out = np.eye(gens["a"].shape[0], dtype=complex)
    for ch in w:
        m = gens[ch.lower()]
        out = out @ (m.conj().T if ch.isupper() else m)
    return out


def k_model(m):
    d = 2 * m
    z = np.exp(2j * math.pi * np.arange(d) / d)
    y = np.diag(z)
    a = np.zeros((d, d), complex)
    for j in range(d):
        a[j, (j + m) % d] = 1
    x = np.zeros((d, d), complex)
    for j in range(d):
        x[j, 2 * (j % m) + j // m] = 1
    return {"a": a, "b": -a, "c": -np.eye(d, dtype=complex), "x": x, "y": y}


def k_errors(gens, rho):
    xi = sqrtm_psd(rho)
    rel = max(hs((word(r, gens) - np.eye(len(rho))) @ xi) for r in K_RELATORS)
    com = max(hs(comm(gens[s], xi)) for s in "abcxy")
    vis = hs((gens["c"] + np.eye(len(rho))) @ xi)
    return max(rel, com, vis), xi


def k_profile(rep, g):
    worst = 0.0
    for m in (1, 2, 3, 5, 10):
        gens = k_model(m)
        d = 2 * m
        rho = np.eye(d) / d
        xi = sqrtm_psd(rho)
        defects = [hs((word(r, gens) - np.eye(d)) @ xi) for r in K_RELATORS]
        target = math.sqrt(2) * math.sin(math.pi / d)
        worst = max(worst, abs(defects[0] - target), max(defects[1:]))
    rep.check("explicit K model: one defect sqrt2 sin(pi/d), the others 0", f"{worst:.2e}", 0, worst < 1e-12)
    e_dd = e_ds = e_dim = 0.0
    n = 0
    kinds = ["nearflat", "geometric", "random", "diag"]
    for t in range(120):
        m = int(g.choice([1, 2, 3, 4, 6]))
        d = 2 * m
        gens = {k: perturbed_unitary(v, [0.0, 0.02, 0.1, 0.4][t % 4], g) for k, v in k_model(m).items()}
        rho = random_state(d, g, kinds[t % 4]) if t % 5 else np.eye(d, dtype=complex) / d
        e, xi = k_errors(gens, rho)
        if e < 1e-12:
            continue
        p, w = spectral(gens["y"], rho)
        dd = w1(2 * p, w, p, w)
        ds = w1(p + math.pi, w, p, w)
        e_dd = max(e_dd, dd / (5 * math.pi * e))
        e_ds = max(e_ds, ds / (9.5 * math.pi * e))
        e_dim = max(e_dim, 1 / (29 * e * d))
        n += 1
    rep.check(f"profile of K, {n} perturbed models: max W1(D_* mu, mu) / (5 pi e)", f"{e_dd:.4f}", 1, e_dd <= 1)
    rep.check(f"profile of K, {n} perturbed models: max W1(S_* mu, mu) / (9.5 pi e)", f"{e_ds:.4f}", 1, e_ds <= 1)
    rep.check(f"profile of K, {n} perturbed models: max 1/(29 e d)", f"{e_dim:.4f}", 1, e_dim <= 1)


def gibbs_g(a, t):
    return math.log2(1 / math.tanh(a / 2)) + a * t / math.log(2)


def readout(y, rho, m):
    """Joint law of the binned two-sided readout on the canonical purification, from ||P_j xi P_k||^2."""
    xi = sqrtm_psd(rho)
    t, z = sl.schur(y, output="complex")
    ang = np.mod(np.angle(np.diag(t)), TWO_PI)
    bins = np.minimum((ang * m / TWO_PI).astype(int), m - 1)
    proj = []
    for k in range(m):
        cols = z[:, bins == k]
        proj.append(cols @ cols.conj().T)
    joint = np.array([[hs(proj[j] @ xi @ proj[k]) ** 2 for k in range(m)] for j in range(m)])  # [B, A]
    return joint


def shannon(p):
    p = np.asarray(p).ravel()
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def entropy_transport(rep, g):
    worst_holevo = worst_main = worst_simple = -1e9
    n = 0
    kinds = ["random", "geometric", "nearflat", "diag", "lowrank"]
    for t in range(160):
        d = int(g.choice([2, 4, 8, 12, 16]))
        if t % 2:
            y = np.diag(np.exp(2j * math.pi * (np.arange(d) + g.normal(0, 0.2, d)) / d))
        else:
            y = sl.expm(1j * herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))
        rho = random_state(d, g, kinds[t % 5])
        if t % 7 == 0:  # commuting with Y, close to flat: entropy and spread together
            rho = np.diag(g.dirichlet(np.ones(d) * 20)).astype(complex)
            y = np.diag(np.exp(2j * math.pi * np.arange(d) / d))
        xi = sqrtm_psd(rho)
        p, w = spectral(y, rho)
        delta = w1_haar(p, w)
        c = hs(comm(y, xi))
        s = entropy_bits(rho)
        for m in (2, 3, 4, 6, 8, 16):
            joint = readout(y, rho, m)
            mi = shannon(joint.sum(1)) + shannon(joint.sum(0)) - shannon(joint)
            worst_holevo = max(worst_holevo, mi - s)
            for a in (0.1, 0.5, 1.0):
                for b in (0.1, 0.5, 1.0):
                    bound = math.log2(m) - gibbs_g(a, 1 + m * delta / TWO_PI) - gibbs_g(b, 1 + m * c / 4)
                    worst_main = max(worst_main, bound - s)
        worst_simple = max(worst_simple, math.log2(1 / (delta + c)) - 8 - s)
        n += 1
    rep.check(f"Holevo: max I(A:B) - S(rho), {n} cases", f"{worst_holevo:.2e}", 0, worst_holevo <= 1e-9)
    rep.check("entropy from transport, finite-bin form: max (bound - S)", f"{worst_main:.3f}", 0, worst_main <= 1e-9)
    rep.check("entropy from transport, S >= log(1/(delta+c)) - 8: max (bound - S)", f"{worst_simple:.3f}", 0,
              worst_simple <= 1e-9)
    # nonvacuous control: flat clocks, where both the bound and the entropy grow like log d
    for d in (64, 256):
        y = np.diag(np.exp(2j * math.pi * np.arange(d) / d))
        rho = np.eye(d, dtype=complex) / d
        p, w = spectral(y, rho)
        delta = w1_haar(p, w)
        best = max(math.log2(m) - gibbs_g(a, 1 + m * delta / TWO_PI) - gibbs_g(1.0, 1.0)
                   for m in (d // 4, d // 2, d) for a in (0.1, 0.3, 1.0))
        rep.check(f"flat clock d={d}: finite-bin bound is nonvacuous and below S = {math.log2(d):.0f}",
                  f"{best:.2f}", f"{math.log2(d):.0f}", 0 < best <= math.log2(d))


def k_entropy(rep, g):
    worst = -1e9
    n = 0
    for t in range(60):
        m = int(g.choice([4, 8, 16]))
        d = 2 * m
        gens = {k: perturbed_unitary(v, [0.0, 0.002, 0.01][t % 3], g) for k, v in k_model(m).items()}
        rho = np.diag(1 + 0.05 * g.uniform(-1, 1, d)).astype(complex)
        rho /= np.trace(rho).real
        if t % 2:
            u = sl.expm(1j * 0.01 * herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))
            rho = u @ rho @ u.conj().T
        e, xi = k_errors(gens, rho)
        if e > 0.5:
            continue
        worst = max(worst, math.log2(1 / e) - 10 - entropy_bits(rho))
        n += 1
    rep.check(f"entropy profile of K, {n} models with e <= 1/2: max (log(1/e) - 10 - S)", f"{worst:.3f}", 0,
              worst <= 1e-9)


def clock(rep):
    d = 16
    y = np.diag(np.exp(2j * math.pi * np.arange(d) / d))
    v = np.ones(d, dtype=complex) / math.sqrt(d)
    rho = np.outer(v, v.conj())
    p, w = spectral(y, rho)
    delta = w1_haar(p, w)
    c = hs(comm(y, rho))
    rep.check("pure clock state: W1 to Haar equals pi/(2d)", f"{delta:.6f}", f"{math.pi / (2 * d):.6f}",
              abs(delta - math.pi / (2 * d)) < 1e-6)
    rep.check("pure clock state: entropy 0, state commutator sqrt 2", f"{entropy_bits(rho):.2e}, {c:.6f}",
              "0, 1.414214", entropy_bits(rho) < 1e-9 and abs(c - math.sqrt(2)) < 1e-9)


def main():
    rep = Report("Spectral transport")
    g = rng(31)
    circle_facts(rep, g)
    couplings(rep, g)
    k_profile(rep, g)
    entropy_transport(rep, g)
    k_entropy(rep, g)
    clock(rep)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
