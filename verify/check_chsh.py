"""Attacks on the CHSH(n) argument, whose theorem is proved in the companion paper (Theorem E there): the sum-of-squares identity, the one-pair bounds, the two algebraic
identities, every inequality of the triple construction on perturbed near-optimal strategies with non-flat states,
and the exact strategy with floor(n/2) ebits."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import math

import numpy as np

from common import (PX, PY, PZ, I2, Report, acomm, comm, entropy_bits, h2, hs, kron_list, perturbed_involution, rng,
                    round_unitary, sgn_inv, herm, sqrtm_psd, unitary)

SQ2 = math.sqrt(2)


def jordan_wigner(m):
    ops = []
    for k in range(m):
        pre = [PZ] * k
        post = [I2] * (m - k - 1)
        ops.append(kron_list(pre + [PX] + post))
        ops.append(kron_list(pre + [PY] + post))
    ops.append(kron_list([PZ] * m))
    return ops


def expect(psi, a, b):
    """<psi, (A x B) psi> with psi stored as a matrix psi[alice, bob]."""
    return complex(np.vdot(psi, a @ psi @ b.T)).real


def strategy(n, lam, eta, g):
    """Clifford strategy on Phi_{2^m} (x) (a non-flat ancilla state with Schmidt weights lam), perturbed by eta."""
    m = n // 2
    cl = jordan_wigner(m)[:n]
    k = len(lam)
    da = 2 ** m * k
    phi = np.eye(2 ** m) / math.sqrt(2 ** m)
    anc = np.diag(np.sqrt(lam))
    psi = np.kron(phi, anc)  # psi[a, b] for Phi (x) ancilla, both Alice-first
    A = [np.kron(c, np.eye(k)) for c in cl]
    B = {}
    for i, j in itertools.permutations(range(n), 2):
        lo, hi = min(i, j), max(i, j)
        sign = 1 if (i, j) == (lo, hi) else -1
        B[(i, j)] = np.kron(((cl[lo] + sign * cl[hi]) / SQ2).T, np.eye(k))
    A = [perturbed_involution(a, eta, g) for a in A]
    B = {key: perturbed_involution(b, eta, g) for key, b in B.items()}
    return psi.astype(complex), A, B


def deficits(psi, A, B, n):
    delta = {}
    for i, j in itertools.combinations(range(n), 2):
        c = expect(psi, A[i], B[(i, j)] + B[(j, i)]) + expect(psi, A[j], B[(i, j)] - B[(j, i)])
        delta[(i, j)] = 2 * SQ2 - c
    return delta


def sos_identity(rep, g):
    worst = 0.0
    worst_pair = {"eta": 0.0, "c": 0.0}
    for t in range(300):
        da, db = int(g.integers(2, 5)), int(g.integers(2, 5))
        psi = g.normal(size=(da, db)) + 1j * g.normal(size=(da, db))
        if t % 2:  # near-optimal instances on a qubit pair with a random local ancilla
            base = np.array([[1, 0], [0, 1]]) / SQ2
            anc = g.dirichlet(np.ones(2))
            psi = np.kron(base, np.diag(np.sqrt(anc)))
            da = db = 4
            a, a2 = np.kron(PZ, I2), np.kron(PX, I2)
            b, b2 = np.kron(((PZ + PX) / SQ2).T, I2), np.kron(((PZ - PX) / SQ2).T, I2)
            eta = [1e-4, 1e-2, 0.1, 0.5][t % 4]
            a, a2, b, b2 = (perturbed_involution(x, eta, g) for x in (a, a2, b, b2))
        else:
            a, a2 = (sgn_inv(herm(g.normal(size=(da, da)) + 1j * g.normal(size=(da, da)))) for _ in range(2))
            b, b2 = (sgn_inv(herm(g.normal(size=(db, db)) + 1j * g.normal(size=(db, db)))) for _ in range(2))
        psi = psi / np.linalg.norm(psi)
        t0, t1 = (b + b2) / SQ2, (b - b2) / SQ2
        c = expect(psi, a, b + b2) + expect(psi, a2, b - b2)
        delta = 2 * SQ2 - c
        n0 = np.linalg.norm(a @ psi - psi @ t0.T)
        n1 = np.linalg.norm(a2 @ psi - psi @ t1.T)
        worst = max(worst, abs(delta - (n0 ** 2 + n1 ** 2) / SQ2))
        xi = sqrtm_psd(psi @ psi.conj().T)  # Alice's reduced state
        if delta > 1e-12:
            worst_pair["eta"] = max(worst_pair["eta"], hs(acomm(a, a2) @ xi) ** 2 / (33 * delta))
            worst_pair["c"] = max(worst_pair["c"], hs(comm(a, xi)) ** 2 / (2 * SQ2 * delta),
                                  hs(comm(a2, xi)) ** 2 / (2 * SQ2 * delta))
    rep.check("sum-of-squares identity, 300 instances: max error", f"{worst:.2e}", 0, worst < 1e-10)
    rep.check("one pair: max ||{A,A'}xi||^2 / (33 delta)", f"{worst_pair['eta']:.4f}", 1, worst_pair["eta"] <= 1)
    rep.check("one pair: max ||[A,xi]||^2 / (2 sqrt2 delta)", f"{worst_pair['c']:.4f}", 1, worst_pair["c"] <= 1)


def identities(rep, g):
    worst = 0.0
    for t in range(100):
        d = int(g.integers(2, 6))
        ab, ac, abp, acp, aa = (sgn_inv(herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d)))) for _ in range(5))
        lhs = ab @ ac @ abp @ acp - abp @ acp @ ab @ ac
        rhs = ab @ acomm(ac, abp) @ acp - acomm(ab, abp) @ ac @ acp + abp @ ab @ acomm(ac, acp) - abp @ acomm(ab, acp) @ ac
        u, w = 1j * aa @ ab, 1j * ab @ ac
        lhs2 = acomm(u, w)
        rhs2 = -ab @ acomm(aa, ac) @ ab + acomm(aa, ab) @ ac @ ab - aa @ ab @ acomm(ab, ac)
        worst = max(worst, hs(lhs - rhs), hs(lhs2 - rhs2))
    rep.check("the two algebraic identities on random involutions: max error", f"{worst:.2e}", 0, worst < 1e-10)


def partitions_into_triples(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for pair in itertools.combinations(rest, 2):
        remaining = [x for x in rest if x not in pair]
        for p in partitions_into_triples(remaining):
            yield [(first,) + pair] + p


def best_partition(n, delta):
    """A partition into triples whose internal deficit sum is at most n * (average deficit), as in Step 0."""
    def internal(p):
        return sum(delta[(min(x, y), max(x, y))] for t in p for x, y in itertools.combinations(t, 2))
    return min(partitions_into_triples(list(range(n))), key=internal)


def triple_chain(rep, g):
    names = ["rounding", "raw commutators", "raw anticommutator", "rounded commutators", "rounded anticommutator",
             "rounded state commutators", "entropy bound from the count", "aggregate A", "aggregate B", "eta", "c"]
    worst = {k: 0.0 for k in names}
    worst["entropy bound from the count"] = -1e9
    final_checked = 0
    final_ok = True
    for t in range(40):
        n = [3, 6][t % 2]
        lam = [g.dirichlet(np.ones(2) * 0.5), np.array([1.0]), np.array([0.7, 0.2, 0.1])][t % 3]
        eta = [1e-7, 1e-5, 1e-3, 0.02, 0.1][t % 5]
        psi, A, B = strategy(n, lam, eta, g)
        delta = deficits(psi, A, B, n)
        xi = sqrtm_psd(psi @ psi.conj().T)
        eta_xy = {(x, y): hs(acomm(A[x], A[y]) @ xi) for x, y in itertools.combinations(range(n), 2)}
        eta_xy.update({(y, x): v for (x, y), v in list(eta_xy.items())})
        c = [hs(comm(a, xi)) for a in A]
        for (x, y), dl in delta.items():
            if dl > 1e-13:
                worst["eta"] = max(worst["eta"], eta_xy[(x, y)] ** 2 / (33 * dl))
                worst["c"] = max(worst["c"], max(c[x], c[y]) ** 2 / (2 * SQ2 * dl))
        triples = best_partition(n, delta)
        M = len(triples)
        U, W, X, Z, lamk = [], [], [], [], []
        for (a, b, cc) in triples:
            u, w = 1j * A[a] @ A[b], 1j * A[b] @ A[cc]
            xk, zk = round_unitary(u), round_unitary(w)
            U.append(u); W.append(w); X.append(xk); Z.append(zk)
            lamk.append(eta_xy[(a, b)] + eta_xy[(b, cc)] + eta_xy[(a, cc)] + c[a] + c[b] + c[cc])
            for r_, base, e_, ca, cb in ((xk, u, eta_xy[(a, b)], c[a], c[b]), (zk, w, eta_xy[(b, cc)], c[b], c[cc])):
                worst["rounding"] = max(worst["rounding"], hs((r_ - base) @ xi) - e_,
                                        hs(comm(r_, xi)) - (ca + cb + 2 * e_))
        mu = {}
        for k, l in itertools.permutations(range(M), 2):
            mu[(k, l)] = sum(eta_xy[(x, y)] for x in triples[k] for y in triples[l])
            worst["raw commutators"] = max(worst["raw commutators"],
                                           hs(comm(W[k], W[l]) @ xi) - (mu[(k, l)] + 4 * lamk[k] + 4 * lamk[l]),
                                           hs(comm(U[k], W[l]) @ xi) - (mu[(k, l)] + 4 * lamk[k] + 4 * lamk[l]))
            worst["rounded commutators"] = max(worst["rounded commutators"],
                                               hs(comm(Z[k], Z[l]) @ xi) - (mu[(k, l)] + 8 * lamk[k] + 8 * lamk[l]),
                                               hs(comm(X[k], Z[l]) @ xi) - (mu[(k, l)] + 8 * lamk[k] + 8 * lamk[l]))
        for k in range(M):
            worst["raw anticommutator"] = max(worst["raw anticommutator"], hs(acomm(U[k], W[k]) @ xi) - 4 * lamk[k])
            worst["rounded anticommutator"] = max(worst["rounded anticommutator"],
                                                  hs(acomm(X[k], Z[k]) @ xi) - 12 * lamk[k])
            worst["rounded state commutators"] = max(worst["rounded state commutators"],
                                                     hs(comm(X[k], xi)) - 2 * lamk[k], hs(comm(Z[k], xi)) - 2 * lamk[k])
        # the qubit count with the actual a_k, b_k, and the aggregate bounds of Step 3
        fs = {}
        for gg in itertools.product((0, 1), repeat=M):
            f = np.eye(len(xi), dtype=complex)
            for gi, zz in zip(gg, Z):
                if gi:
                    f = f @ zz
            fs[gg] = f
        A_sum = B_sum = 0.0
        for k in range(M):
            a2 = b2 = 0.0
            for gg, f in fs.items():
                ge = list(gg); ge[k] ^= 1
                a2 += hs((fs[tuple(ge)] - f @ Z[k]) @ xi) ** 2
                b2 += hs(((-1.0 if gg[k] else 1.0) * X[k] @ f - f @ X[k]) @ xi) ** 2
            A_sum += a2 / len(fs) + 4 * lamk[k] ** 2
            B_sum += b2 / len(fs) + 4 * lamk[k] ** 2
        s = entropy_bits(xi @ xi)
        bound = M - B_sum / math.log(2) - M * h2(min(A_sum / (2 * M), 0.5))
        worst["entropy bound from the count"] = max(worst["entropy bound from the count"], bound - s)
        dbar = sum(delta.values()) / len(delta)
        npr = 3 * M
        worst["aggregate A"] = max(worst["aggregate A"], A_sum - 73162 * npr ** 2 * dbar)
        worst["aggregate B"] = max(worst["aggregate B"], B_sum - 101256 * npr ** 2 * dbar)
        eps = dbar / (2 * SQ2)  # the smallest eps for which this strategy is eps-optimal (N = [n])
        if npr * eps <= 1e-7:
            final_checked += 1
            final_ok &= s >= M / 2
    for k in ("eta", "c"):
        rep.check(f"triple chain, pair bound for {k}: max ratio", f"{worst[k]:.4f}", 1, worst[k] <= 1)
    for k in names[:9]:
        rep.check(f"triple chain, {k}: max excess", f"{worst[k]:.2e}", 0, worst[k] <= 1e-9)
    rep.check(f"final bound S(rho_A) >= M/2 where n eps <= 1e-7 ({final_checked} strategies)", final_ok, True, final_ok)


def exact_strategy(rep):
    ok = True
    for n in range(3, 8):
        m = n // 2
        cl = jordan_wigner(m)[:n]
        anti = max(hs(acomm(cl[i], cl[j])) for i, j in itertools.combinations(range(n), 2))
        psi = np.eye(2 ** m, dtype=complex) / math.sqrt(2 ** m)
        B = {}
        for i, j in itertools.combinations(range(n), 2):
            B[(i, j)] = ((cl[i] + cl[j]) / SQ2).T
            B[(j, i)] = ((cl[i] - cl[j]) / SQ2).T
        beta = sum(expect(psi, cl[i], B[(i, j)] + B[(j, i)]) + expect(psi, cl[j], B[(i, j)] - B[(j, i)])
                   for i, j in itertools.combinations(range(n), 2)) / (2 * n * (n - 1))
        s = entropy_bits(psi @ psi.conj().T)
        ok &= anti < 1e-12 and abs(beta - SQ2 / 2) < 1e-12 and abs(s - m) < 1e-9
        rep.info(f"n={n}: anticommutators {anti:.1e}, bias {beta:.12f}, entropy {s:.6f} = floor(n/2) = {m}")
    rep.check("exact strategy attains sqrt2/2 with floor(n/2) ebits, n = 3..7", ok, True, ok)


def main():
    rep = Report("CHSH(n)")
    g = rng(5)
    sos_identity(rep, g)
    identities(rep, g)
    triple_chain(rep, g)
    exact_strategy(rep)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
