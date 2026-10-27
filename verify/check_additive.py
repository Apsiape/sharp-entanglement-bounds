"""Controls for the two lemmas behind additivity (random spectral cuts; weighted histories), on complex states of
full and deficient rank. Every random-cut expectation is integrated exactly over the offset, by splitting [0, 1/q) at
the spectral breakpoints. Bounds are checked with the explicit constants of the proofs:
  (cut)      E sum_a ||P_a(R) xi - xi P_a(S)||^2 <= (q/2) ||R xi - xi S||;
  (control)  E sum_a ||[P_a(Z), V] xi||^2 <= (3q/2) ||[Z,V] xi|| + 3q c(Z) + (3q + 24) c(V);
  (covariant) the same with ||(V Z V* - zeta Z) xi|| and the shifted labels;
  (two cuts) E sum_{a,b} ||[P_a(Z), P_b(B)] xi||^2 <= (1/4)[3 q sqrt(q') Y^(1/2) + (6q + 48) sqrt(q') (q' c(B)/2)^(1/2)
             + 3 q q' c(Z)], with Y = (3q'/2) ||[B,Z] xi|| + (3q' + 24) c(Z) + 3 q' c(B);
  (history)  sum_h K_h* K_h = sum_h K_h K_h* = I; (sum_h ||[K_h, xi]||^2)^(1/2) <= sum_t c_t;
             (sum_h ||E K_h xi||^2)^(1/2) <= ||E xi|| + ||E|| sum_t c_t, and its failure for a channel that is trace
             preserving but not unital;
  (covariance) the prefix law of a sequential record and its cyclic shift differ in total variation by at most the
             aggregate covariance error;
  (prediction) on the canonical purification, sqrt(2 tau_t) <= 2 H_t + c_t, with tau_t the mismatch probability of
             Alice's and Bob's t-th records and H_t the aggregate history commutator, itself bounded by the telescope."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import math

import numpy as np
import scipy.linalg as sl

from common import Report, comm, hs, rng, sqrtm_psd, unitary


def state(d, rank, g):
    V = unitary(d, g)
    w = np.zeros(d)
    w[:rank] = g.uniform(0.1, 1.0, size=rank)
    w /= w.sum()
    return (V * np.sqrt(w)) @ V.conj().T


def spectral(U):
    T, Zs = sl.schur(U, output="complex")
    return (np.angle(np.diag(T)) / (2 * math.pi)) % 1.0, Zs


def cut_projections(angles, vecs, theta, q):
    labels = np.floor(((angles - theta) % 1.0) * q).astype(int) % q
    return [vecs[:, labels == a] @ vecs[:, labels == a].conj().T for a in range(q)]


def offset_pieces(q, *angle_lists):
    pts = sorted(set([0.0, 1.0 / q] + [float(x % (1.0 / q)) for al in angle_lists for x in al]))
    return [((lo + hi) / 2, q * (hi - lo)) for lo, hi in zip(pts, pts[1:]) if hi > lo]


def near_unitary(U0, eps, g):
    d = U0.shape[0]
    H = g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))
    return sl.expm(1j * eps * (H + H.conj().T) / 2) @ U0


def cut_checks(rep, g):
    worst = {"cut": 0.0, "control": 0.0, "covariant": 0.0, "two": 0.0}
    n = 0
    for d in (4, 6):
        for rank in (1, d - 1, d):
            for eps in (0.1, 1e-2, 1e-4, 1e-6):
                xi = state(d, rank, g)
                q = 2 if d == 4 else int(g.choice([2, 3]))
                q2 = int(g.choice([2, 3]))
                # (cut) R and a perturbation S of it
                R = unitary(d, g)
                S = near_unitary(R, eps, g)
                aR, vR = spectral(R)
                aS, vS = spectral(S)
                lhs = sum(wt * sum(hs(P @ xi - xi @ Q) ** 2 for P, Q in
                                   zip(cut_projections(aR, vR, th, q), cut_projections(aS, vS, th, q)))
                          for th, wt in offset_pieces(q, aR, aS))
                worst["cut"] = max(worst["cut"], lhs / (q / 2 * hs(R @ xi - xi @ S)))
                # (control) Z with a unitary V that nearly commutes with it
                Z, aZ, vZ = R, aR, vR
                V = near_unitary(vZ @ np.diag(np.exp(2j * np.pi * g.random(d))) @ vZ.conj().T, eps, g)
                cZ, cV = hs(comm(Z, xi)), hs(comm(V, xi))
                lhs = sum(wt * sum(hs(comm(P, V) @ xi) ** 2 for P in cut_projections(aZ, vZ, th, q))
                          for th, wt in offset_pieces(q, aZ))
                rhs = 1.5 * q * hs(comm(Z, V) @ xi) + 3 * q * cZ + (3 * q + 24) * cV
                worst["control"] = max(worst["control"], lhs / rhs)
                # (covariant) Z with spectrum invariant under rotation by 1/q, V close to the shift carrying Z to zeta Z
                base = unitary(d, g)
                u = g.random(d // q)
                phases = np.concatenate([u + a / q for a in range(q)]) % 1.0
                Zc0 = base @ np.diag(np.exp(2j * np.pi * phases)) @ base.conj().T
                shift = np.zeros((d, d))
                for a in range(q):
                    for k in range(d // q):
                        shift[((a + 1) % q) * (d // q) + k, a * (d // q) + k] = 1
                V0 = base @ shift @ base.conj().T
                Zc = near_unitary(Zc0, eps, g)
                Vc = near_unitary(V0, eps, g)
                aC, vC = spectral(Zc)
                zeta = np.exp(2j * np.pi / q)
                lhs = 0.0
                for th, wt in offset_pieces(q, aC):
                    Ps = cut_projections(aC, vC, th, q)
                    lhs += wt * sum(hs((Ps[a] @ Vc - Vc @ Ps[(a + 1) % q]) @ xi) ** 2 for a in range(q))
                rhs = 1.5 * q * hs((Vc @ Zc @ Vc.conj().T - zeta * Zc) @ xi) + 3 * q * hs(comm(Zc, xi)) +                     (3 * q + 24) * hs(comm(Vc, xi))
                worst["covariant"] = max(worst["covariant"], lhs / rhs)
                # (two cuts) Z and B, a perturbation of a unitary commuting with Z
                B = near_unitary(vZ @ np.diag(np.exp(2j * np.pi * g.random(d))) @ vZ.conj().T, eps, g)
                aB, vB = spectral(B)
                cB = hs(comm(B, xi))
                lhs = 0.0
                for thB, wB in offset_pieces(q2, aB):
                    Qs = cut_projections(aB, vB, thB, q2)
                    for thZ, wZ in offset_pieces(q, aZ):
                        Ps = cut_projections(aZ, vZ, thZ, q)
                        lhs += wB * wZ * sum(hs(comm(P, Qb) @ xi) ** 2 for P in Ps for Qb in Qs)
                Yv = 1.5 * q2 * hs(comm(B, Z) @ xi) + (3 * q2 + 24) * cZ + 3 * q2 * cB
                rhs = 0.25 * (3 * q * math.sqrt(q2) * math.sqrt(Yv) + (6 * q + 48) * math.sqrt(q2) *
                              math.sqrt(q2 * cB / 2) + 3 * q * q2 * cZ)
                worst["two"] = max(worst["two"], lhs / rhs)
                n += 1
    rep.check(f"random cuts, {n} cases: E sum ||P(R) xi - xi P(S)||^2 / ((q/2)||R xi - xi S||)",
              f"{worst['cut']:.3f}", 1, worst["cut"] <= 1 + 1e-9)
    rep.check("random cuts with a unitary control: ratio to the explicit bound", f"{worst['control']:.3f}", 1,
              worst["control"] <= 1 + 1e-9)
    rep.check("random cuts, covariant form: ratio to the explicit bound", f"{worst['covariant']:.3f}", 1,
              worst["covariant"] <= 1 + 1e-9)
    rep.check("two spectral partitions: ratio to the explicit square-root bound", f"{worst['two']:.3f}", 1,
              worst["two"] <= 1 + 1e-9)


def random_partition(d, g, k):
    V = unitary(d, g)
    labels = g.integers(0, k, size=d)
    return [V[:, labels == a] @ V[:, labels == a].conj().T for a in range(k)]


def history_checks(rep, g):
    worst_comp = worst_cent = worst_E = worst_tv = worst_pred = 0.0
    cases = 0
    for d in (3, 4, 5):
        for rank in (1, d - 1, d):
            xi = state(d, rank, g)
            I = np.eye(d, dtype=complex)
            # partitions close to a common frame, so that the errors are small but nonzero
            frame = unitary(d, g)
            parts = []
            for t in range(4):
                k = int(g.choice([2, 3]))
                W = near_unitary(frame, 10 ** g.uniform(-3, -0.5), g)
                labels = g.integers(0, k, size=d)
                parts.append([W[:, labels == a] @ W[:, labels == a].conj().T for a in range(k)])
            cs = [math.sqrt(sum(hs(comm(P, xi)) ** 2 for P in part)) for part in parts]
            hist = [((), I)]
            for t, part in enumerate(parts):
                hist = [(h + (a,), P @ K) for h, K in hist for a, P in enumerate(part)]
                tp = sum(K.conj().T @ K for _, K in hist)
                un = sum(K @ K.conj().T for _, K in hist)
                worst_comp = max(worst_comp, hs(tp - I), hs(un - I))
                agg = math.sqrt(sum(hs(comm(K, xi)) ** 2 for _, K in hist))
                worst_cent = max(worst_cent, agg - sum(cs[:t + 1]))
                E = g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))
                E /= np.linalg.norm(E, 2)
                lhs = math.sqrt(sum(hs(E @ K @ xi) ** 2 for _, K in hist))
                worst_E = max(worst_E, lhs - hs(E @ xi) - sum(cs[:t + 1]))
            # covariance: a unitary V and the cyclic shift of the last label
            T = len(parts)
            last = parts[-1]
            k = len(last)
            V = near_unitary(np.eye(d), 0.3, g)
            hist = [((), I)]
            for part in parts:
                hist = [(h + (a,), P @ K) for h, K in hist for a, P in enumerate(part)]
            prob = {h: hs(K @ xi) ** 2 for h, K in hist}
            shift = lambda h: h[:-1] + ((h[-1] + 1) % k,)
            tv = 0.5 * sum(abs(prob[h] - prob[shift(h)]) for h in prob)
            Kd = dict(hist)
            agg = math.sqrt(sum(hs(Kd[h] @ xi @ V - V @ Kd[shift(h)] @ xi) ** 2 for h in prob))
            worst_tv = max(worst_tv, tv - agg)
            # prediction: Bob measures parts in order, Alice the transposes in the same order (canonical purification)
            for t in range(1, T):
                prefix = [((), I)]
                for part in parts[:t]:
                    prefix = [(h + (a,), P @ K) for h, K in prefix for a, P in enumerate(part)]
                Pt = parts[t]
                tau = 0.0
                for _, Kh in prefix:
                    for _, Kg in prefix:
                        sig = Kh @ xi @ Kg.conj().T
                        tau += sum(hs(Pa @ sig @ Pb) ** 2 for i, Pa in enumerate(Pt) for j, Pb in enumerate(Pt)
                                   if i != j)
                Ht = math.sqrt(sum(hs(comm(Pa, Kh) @ xi) ** 2 for Pa in Pt for _, Kh in prefix))
                bound_H = 0.0
                for s in range(t):
                    dts = math.sqrt(sum(hs(comm(Pa, Pb) @ xi) ** 2 for Pa in Pt for Pb in parts[s]))
                    opn = math.sqrt(sum(np.linalg.norm(comm(Pa, Pb), 2) ** 2 for Pa in Pt for Pb in parts[s]))
                    bound_H += dts + opn * sum(cs[:s])
                worst_pred = max(worst_pred, math.sqrt(2 * tau) - (2 * Ht + cs[t]), Ht - bound_H)
            cases += 1
    rep.check(f"histories, {cases} sequences: trace preserving and unital (max error)", f"{worst_comp:.1e}", 0,
              worst_comp < 1e-12)
    rep.check("histories: aggregate centrality minus sum of partition centralities", f"{worst_cent:.1e}", "<= 0",
              worst_cent <= 1e-12)
    rep.check("histories: aggregate ||E K_h xi|| minus (||E xi|| + ||E|| sum c_t)", f"{worst_E:.1e}", "<= 0",
              worst_E <= 1e-12)
    rep.check("covariance: total variation minus aggregate covariance error", f"{worst_tv:.1e}", "<= 0",
              worst_tv <= 1e-12)
    rep.check("prediction: sqrt(2 tau) - (2H + c) and H - telescope bound", f"{worst_pred:.1e}", "<= 0",
              worst_pred <= 1e-12)
    # a channel that is trace preserving but not unital breaks the history bound
    xi = np.eye(2) / math.sqrt(2)
    E = np.diag([1.0, 0.0])
    reset = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 1.0], [0.0, 0.0]])]
    lhs = sum(hs(E @ K @ xi) ** 2 for K in reset)
    c = math.sqrt(sum(hs(comm(K, xi)) ** 2 for K in reset))
    rhs = (hs(E @ xi) + c) ** 2
    ok = abs(lhs - 1) < 1e-12 and abs(rhs - 0.5) < 1e-12
    rep.check("negative control: the reset channel (trace preserving, not unital) violates the history bound",
              f"lhs {lhs:.3f} > rhs {rhs:.3f}", "violation", ok)


def main():
    rep = Report("Random spectral cuts and weighted histories")
    g = rng(41)
    cut_checks(rep, g)
    history_checks(rep, g)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
