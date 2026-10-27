"""Checks for the coefficient family: cyclic triangular extraction, cuts across the parties with q arcs, Fourier-lamp
models with non-dyadic dilations, the depth arithmetic of Theorem rational-ED, and the coefficient set.

1. One stage of the cyclic extraction (Lemma cyclic-triangular): for q = 2, 4, 8, with controls U of order other than q
   (a phase on the wrap-around), Alice's isometries W_A^t = sum_b |b> (U^A)^(t-b) P_b and Bob's W_B^t = sum_c |c> (U^B)^(c-t) P_c
   on the maximally entangled state, with Bob holding the transposes, give exactly the ideal branches, and every W^t is an isometry.
1b. The one-stage inequality (eq. cyc-stage) without orthogonality of unrecorded vectors. Negative control: for q = 2, t = 0,
   U = X, P_b = |b><b| and v = |+>, the vectors U^(t-b) P_b v are both |0>/sqrt 2, with inner product 1/2, so they are not
   orthogonal in b. The identity used instead, sum_b ||U^(t-b) P_b v||^2 = ||v||^2, holds on random partitions, unitaries and
   vectors; and ||(A - I) eta||, computed branch by branch, is at most the right side of eq. cyc-stage with C_Q = 1 on perturbed
   exact instances and on unrelated random partitions, controls and vectors with an environment.
2. Several stages on a mixed state: Alice's partitions and controls are conjugated by a different random unitary at each stage
   (which breaks the mirrors and the commutation of different stages), on Phi_D tensored with an Alice ancilla entangled with an
   environment that is never touched. The extraction error is zero at zero perturbation and scales linearly with the largest
   hypothesis residual delta. Negative control: Bob's controls in Alice's orientation give an output far from Phi.
3. Cuts across the parties with q arcs (proof of Theorem rational-ED): for commuting unitaries R_A, S_B of two parties and an
   offset uniform modulo 1/q, E sum_a ||(P_a(R_A) - P_a(S_B)) Psi||^2 <= (q/2) ||(R_A - S_B) Psi||, computed exactly from the joint
   spectral measure and compared with a Monte Carlo average over offsets.
4. Fourier-lamp models of L_{q,m} for non-dyadic m (q = 2, 4, 8 dividing m = 6, 10, 12, 24): the row, the squares and the dilation
   are exact, and the lamp conjugacies obey the operator bound of Example fourier-lamps.
5. Depth arithmetic: with L_i = floor((x - 32 log(2 + x)) / log m_i), L_i >= 1 for large x, e m_i^{L_i} <= (2 + x)^{-32} with
   e = 2^{-x}, and log D = sum r_i L_i >= 2 alpha x - 64 alpha log(2 + x) - sum r_i.
6. The coefficient set: rational multiples a/(2 log m^b) = (a/b)/(2 log m), every positive rational a/b realized with m = 2^b, the
   dense set a/(2 b log 6), and the one-copy family r/(2(b + log 3)) approximating targets in (0, 1/2).
7. The balanced construction of Proposition extended. The l literal lifts of Slofstra's Proposition 4.8 (involutive generators
   rho(x) + I, f_j the exchange, v_j = [[0, rho(y_j)^*], [rho(y_j), 0]], later y_i as rho(y_i) + rho(y_i)) are built on random
   d-dimensional inputs with w = -I, for l = 1, 2, 3. Negative control (the rank obstruction): for l >= 2 the marked involution is
   W = (-I_d) + I_{(2^l - 1) d}, and for every unitary T, ||T W T^* + W||_f >= 2 (1 - 2^{1-l}) (trace bound) and
   >= 2 (1 - 2^{1-l})^{1/2} (Hoffman-Wielandt, sqrt 2 at l = 2); the last exchange attains the second bound, and random unitaries and
   block permutations never beat it. For l = 1 the exchange gives T W T^* = -W exactly. Positive check: the conjugates W_i of w by the
   swap words g_i are -I on block i only, commute with each other and with every conjugated source generator, and the parity
   p = W_0 ... W_{k-1} (k = 2^{l-1}) has equal multiplicities with f_l p f_l = -p, all exactly. Then the Slofstra-Vidick group K with the
   companion's flat models (Example 11.11 there; X a permutation matrix): an involutive permutation tau with tau X tau = X^{-1},
   f' = tau + tau and v' = tau X + tau X are involutions with f' v' = X + X, c lifts to (-I_d) + I_d with t c t = -c for the exchange t,
   and every relator of K keeps, in the lift, at most the residual it has in the model. This replays Steps 1 to 3 of the balanced lift
   of the companion's Lemma 11.13, at the level of the relators of K, with code written independently of the companion's program
   anc/sv_balanced_lift.py, which also builds the solution group (Step 4).
Exact or floating-point finite checks; single-threaded; small matrices only."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import math
from fractions import Fraction

import numpy as np
import scipy.linalg as sl

from common import Report, herm, rng, unitary


# ---------------------------------------------------------------- helpers
def shift(q, phase):
    """Cyclic shift |a> -> |a+1> with a phase on the wrap-around, so that U^q = e^{i phase} I."""
    u = np.zeros((q, q), dtype=complex)
    for a in range(q):
        u[(a + 1) % q, a] = np.exp(1j * phase) if a == q - 1 else 1.0
    return u


def digit_ops(qs, j):
    """Projections onto digit j of a mixed-radix register and the cyclic shift of that digit."""
    dims = list(qs)
    left = int(np.prod(dims[:j])) if j else 1
    right = int(np.prod(dims[j + 1:])) if j + 1 < len(dims) else 1
    q = dims[j]
    projs = [np.kron(np.kron(np.eye(left), np.diag(np.eye(q)[a]).astype(complex)), np.eye(right)) for a in range(q)]
    u = np.kron(np.kron(np.eye(left), shift(q, 0.731 + 0.2 * j)), np.eye(right))
    return projs, u


def stage_isometry(projs, u, t, orientation):
    """W^t = sum_b |b> (x) U^{s(b)} P_b with s(b) = t - b (Alice) or b - t (Bob); exponents in (-q, q)."""
    q = len(projs)
    d = projs[0].shape[0]
    w = np.zeros((q * d, d), dtype=complex)
    for b in range(q):
        k = (t - b) if orientation == "A" else (b - t)
        w[b * d:(b + 1) * d, :] = np.linalg.matrix_power(u, k) @ projs[b] if k >= 0 else \
            np.linalg.matrix_power(u.conj().T, -k) @ projs[b]
    return w


def compose(stages_A, t):
    """Composite local isometry: stage 0 first; registers ordered (reg_{M-1}, ..., reg_0, system)."""
    d = stages_A[0][0][0].shape[0]
    total = np.eye(d, dtype=complex)
    for j, (projs, u, orient) in enumerate(stages_A):
        w = stage_isometry(projs, u, t[j], orient)
        q = len(projs)
        # apply w to the system factor of the current output (registers so far tensor system)
        r = total.shape[0] // d
        total = np.kron(np.eye(r), w) @ total  # registers so far (r) tensor (q tensor system)
        # reorder so the new register comes first: (r, q, d) -> (q, r, d)
        total = total.reshape(r, q, d, -1).transpose(1, 0, 2, 3).reshape(q * r * d, -1)
    return total


def distance_to_phi(v, regdim, dA, dB, dE):
    """min over unit chi of ||v - Phi_regdim (x) chi||, v on (regA, A, regB, B, E) ordering given below."""
    v = v.reshape(regdim, dA, regdim, dB, dE)
    w = np.einsum("iajbe,ij->abe", v, np.eye(regdim)) / math.sqrt(regdim)
    f = np.linalg.norm(w)
    return math.sqrt(max(0.0, 2 - 2 * f))


# ---------------------------------------------------------------- 1. one stage
def one_stage(rep):
    worst_branch, worst_iso = 0.0, 0.0
    for q in (2, 4, 8):
        ua = shift(q, 0.731)
        ub = ua.T
        phi = np.eye(q, dtype=complex).reshape(-1) / math.sqrt(q)
        projs = [np.diag(np.eye(q)[a]).astype(complex) for a in range(q)]
        for t in range(q):
            for orient, u in (("A", ua), ("B", ub)):
                w = stage_isometry(projs, u, t, orient)
                worst_iso = max(worst_iso, float(np.linalg.norm(w.conj().T @ w - np.eye(q))))
            for b, c in itertools.product(range(q), repeat=2):
                wa = (np.linalg.matrix_power(ua, t - b) if t >= b else np.linalg.matrix_power(ua.conj().T, b - t)) @ projs[b]
                wb = (np.linalg.matrix_power(ub, c - t) if c >= t else np.linalg.matrix_power(ub.conj().T, t - c)) @ projs[c]
                actual = np.kron(wa, wb) @ phi
                ideal = np.kron(projs[t], np.eye(q)) @ phi if b == c else np.zeros_like(phi)
                worst_branch = max(worst_branch, float(np.linalg.norm(actual - ideal)))
        assert np.linalg.norm(np.linalg.matrix_power(ua, q) - np.exp(1j * 0.731) * np.eye(q)) < 1e-9  # U^q is not I
    rep.check("one stage, q = 2, 4, 8, controls of order other than q: largest branch error", f"{worst_branch:.1e}", 0,
              worst_branch < 1e-12)
    rep.check("one stage: every W^t is an isometry (largest deviation)", f"{worst_iso:.1e}", 0, worst_iso < 1e-12)


# ---------------------------------------------------------------- 1b. the one-stage inequality without orthogonality
def mpow(u, k):
    return np.linalg.matrix_power(u, k) if k >= 0 else np.linalg.matrix_power(u.conj().T, -k)


def random_partition(d, q, g):
    """A projective partition of C^d into q nonzero blocks (d >= q), in a random basis."""
    labels = np.concatenate([np.arange(q), g.integers(0, q, size=d - q)])
    v = unitary(d, g)
    return [v @ np.diag((labels == a).astype(complex)) @ v.conj().T for a in range(q)]


def stage_error_and_bound(PA, UA, PB, UB, eta):
    """||(A - I) eta|| for one stage of the cyclic lemma, computed branch by branch, and the right side of the stage
    inequality (eq. cyc-stage) with C_Q = 1. eta has shape (dA, dB, dE); Alice acts on axis 0, Bob on axis 1."""
    q = len(PA)

    def act_a(op, v):
        return np.einsum("xa,abe->xbe", op, v)

    def act_b(op, v):
        return np.einsum("yb,abe->aye", op, v)

    lhs2 = 0.0
    for t in range(q):
        for b in range(q):
            wa = mpow(UA, t - b) @ PA[b]
            for c in range(q):
                actual = act_a(wa, act_b(mpow(UB, c - t) @ PB[c], eta))
                ideal = act_a(PA[t], eta) if b == c else 0.0
                lhs2 += float(np.linalg.norm(actual - ideal)) ** 2
    lhs = math.sqrt(lhs2 / q)
    proj = math.sqrt(sum(float(np.linalg.norm(act_a(PA[c], eta) - act_b(PB[c], eta))) ** 2 for c in range(q)))
    unit = sum(float(np.linalg.norm(act_a(mpow(UA, n), eta) - act_b(mpow(UB, n), eta))) for n in range(-q + 1, q))
    cov = sum(math.sqrt(sum(float(np.linalg.norm(act_a(mpow(UA, n) @ PA[b] @ mpow(UA, -n) - PA[(b + n) % q], eta))) ** 2
                            for b in range(q))) for n in range(-q + 1, q))
    return lhs, proj + unit + cov


def stage_inequality(rep, g):
    # negative control: the unrecorded vectors (U^A)^(t-b) P_b v are not orthogonal in b (q = 2, t = 0, U = X, v = |+>)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    P = [np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)]
    v = np.ones(2, dtype=complex) / math.sqrt(2)
    u0, u1 = mpow(X, 0) @ P[0] @ v, mpow(X, -1) @ P[1] @ v
    inner = float(abs(np.vdot(u0, u1)))
    rep.check("negative control: unrecorded vectors U^(t-b) P_b v for q = 2, t = 0, U = X, v = |+>, |inner product|",
              f"{inner:.3f}", "= 1/2 (not orthogonal)", abs(inner - 0.5) < 1e-12)
    # the identity used instead: sum_b ||U^(t-b) P_b v||^2 = ||v||^2, for random partitions, unitaries and vectors
    worst = abs(float(np.linalg.norm(u0)) ** 2 + float(np.linalg.norm(u1)) ** 2 - 1.0)
    cases = 0
    for q in (2, 3, 4, 5, 8):
        for d in (q, 2 * q + 1, 3 * q):
            Ps, U = random_partition(d, q, g), unitary(d, g)
            w = g.normal(size=d) + 1j * g.normal(size=d)
            for t in range(q):
                total = sum(float(np.linalg.norm(mpow(U, t - b) @ Ps[b] @ w)) ** 2 for b in range(q))
                worst = max(worst, abs(total - float(np.linalg.norm(w)) ** 2) / float(np.linalg.norm(w)) ** 2)
                cases += 1
    rep.check(f"identity used instead: sum_b ||U^(t-b) P_b v||^2 = ||v||^2, {cases} random (q, d, t) cases, largest relative "
              "deviation", f"{worst:.1e}", 0, worst < 1e-12)
    # the stage inequality with constant one, on perturbed exact instances (Bob holding transposes on Phi_d (x) env) and on
    # unrelated random partitions, controls and vectors
    worst_ratio, n = 0.0, 0
    for q in (2, 3, 4, 6):
        k, dE = 2, 2
        d = q * k
        PA0 = [np.kron(np.diag(np.eye(q)[a]).astype(complex), np.eye(k)) for a in range(q)]
        UA0 = np.kron(shift(q, 0.731), np.eye(k))
        phi = np.einsum("ab,e->abe", np.eye(d, dtype=complex) / math.sqrt(d), np.eye(dE)[0])
        for tau in (1e-1, 1e-2, 1e-3):
            VA = sl.expm(1j * tau * herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))
            VB = sl.expm(1j * tau * herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))
            PA = [VA @ P0 @ VA.conj().T for P0 in PA0]
            PB = [VB @ P0.T @ VB.conj().T for P0 in PA0]
            UA = VA @ UA0 @ VA.conj().T @ sl.expm(1j * tau * herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))
            UB = VB @ UA0.T @ VB.conj().T
            noise = g.normal(size=phi.shape) + 1j * g.normal(size=phi.shape)
            eta = phi + tau * noise / np.linalg.norm(noise)
            eta = eta / np.linalg.norm(eta)
            lhs, rhs = stage_error_and_bound(PA, UA, PB, UB, eta)
            worst_ratio = max(worst_ratio, lhs / rhs)
            n += 1
        for _ in range(3):
            PA, PB = random_partition(d, q, g), random_partition(d, q, g)
            UA, UB = unitary(d, g), unitary(d, g)
            eta = g.normal(size=(d, d, dE)) + 1j * g.normal(size=(d, d, dE))
            eta = eta / np.linalg.norm(eta)
            lhs, rhs = stage_error_and_bound(PA, UA, PB, UB, eta)
            worst_ratio = max(worst_ratio, lhs / rhs)
            n += 1
    rep.check(f"one-stage inequality (eq. cyc-stage) with C_Q = 1, {n} perturbed and random instances, q = 2, 3, 4, 6: "
              "largest ||(A - I) eta|| / right side", f"{worst_ratio:.3f}", "<= 1", worst_ratio <= 1 + 1e-12)


# ---------------------------------------------------------------- 2. several stages on a mixed state
def hypotheses(PA, UA, PB, UB, psi_ops, Q):
    """Largest root aggregate of the hypotheses (i)-(iii) of the cyclic lemma, on the vector Psi."""
    apply_A, apply_B = psi_ops
    M = len(PA)
    worst = 0.0
    for j in range(M):
        q = len(PA[j])
        worst = max(worst, math.sqrt(sum(np.linalg.norm(apply_A(PA[j][a]) - apply_B(PB[j][a])) ** 2 for a in range(q))))
        for h in range(-Q, Q + 1):
            uah = np.linalg.matrix_power(UA[j], h) if h >= 0 else np.linalg.matrix_power(UA[j].conj().T, -h)
            ubh = np.linalg.matrix_power(UB[j], h) if h >= 0 else np.linalg.matrix_power(UB[j].conj().T, -h)
            worst = max(worst, float(np.linalg.norm(apply_A(uah) - apply_B(ubh))))
            cov = math.sqrt(sum(np.linalg.norm(apply_A(uah @ PA[j][a] @ uah.conj().T - PA[j][(a + h) % q])) ** 2
                                for a in range(q)))
            worst = max(worst, cov)
            for t in range(j):
                worst = max(worst, math.sqrt(sum(np.linalg.norm(apply_A(uah @ P - P @ uah)) ** 2 for P in PA[t])))
        for t in range(j):
            worst = max(worst, math.sqrt(sum(np.linalg.norm(apply_A(P @ R - R @ P)) ** 2 for P in PA[j] for R in PA[t])))
    return worst


def several_stages(rep, g):
    qs = (4, 2, 2)
    M, D, Q = len(qs), int(np.prod(qs)), max(qs)
    dAp, dE = 2, 2  # Alice's ancilla A' is entangled with the environment E
    dA = D * dAp
    phi = np.eye(D, dtype=complex).reshape(-1) / math.sqrt(D)
    omega = np.array([math.sqrt(0.7), 0, 0, math.sqrt(0.3)], dtype=complex)  # A'E, a mixed marginal on A'
    # Psi on (A, A', B, E): Phi_D on (A, B) and omega on (A', E)
    psi = np.einsum("ab,ce->acbe", phi.reshape(D, D), omega.reshape(dAp, dE)).reshape(-1)

    psi3 = psi.reshape(dA, D, dE)

    def apply_A(op):
        return np.einsum("xa,abe->xbe", op, psi3).reshape(-1)

    def apply_B(op):
        return np.einsum("yb,abe->aye", op, psi3).reshape(-1)

    base = [digit_ops(qs, j) for j in range(M)]
    Hs = [herm(g.normal(size=(dA, dA)) + 1j * g.normal(size=(dA, dA))) for _ in range(M)]
    rows = []
    for tau in (0.0, 1e-2, 3e-3, 1e-3, 3e-4):
        PA, UA, PB, UB = [], [], [], []
        for j, (projs, u) in enumerate(base):
            V = sl.expm(1j * tau * Hs[j] / math.sqrt(dA))
            PA.append([V @ np.kron(P, np.eye(dAp)) @ V.conj().T for P in projs])
            UA.append(V @ np.kron(u, np.eye(dAp)) @ V.conj().T)
            PB.append([P.T for P in projs])
            UB.append(u.T)
        delta = hypotheses(PA, UA, PB, UB, (apply_A, apply_B), Q)
        best = 2.0
        for t in itertools.product(*[range(q) for q in qs]):
            WA = compose([(PA[j], UA[j], "A") for j in range(M)], t)
            WB = compose([(PB[j], UB[j], "B") for j in range(M)], t)
            v = np.einsum("ia,abe,jb->ije", WA, psi.reshape(dA, D, dE), WB, optimize=True).reshape(-1)
            # v is ordered (regA, A, A', regB, B, E); move A' next to E
            v = v.reshape(D, D, dAp, D, D, dE).transpose(0, 1, 3, 4, 2, 5).reshape(-1)
            best = min(best, distance_to_phi(v, D, D, D, dAp * dE))
        rows.append((tau, delta, best))
        rep.info(f"tau {tau:.0e}: largest hypothesis residual delta {delta:.3e}, extraction error {best:.3e}")
    exact = rows[0][2]
    rep.check("several stages (q = 4, 2, 2), unperturbed: extraction error", f"{exact:.1e}", 0, exact < 1e-10)
    ratios = [r[2] / r[1] for r in rows[1:]]
    spread = max(ratios) / min(ratios)
    rep.check("several stages, perturbed: error / delta over four perturbation sizes (largest / smallest)", f"{spread:.2f}",
              "<= 2 (linear scaling)", spread <= 2)
    worst = max(r[2] / (M ** 3 * r[1]) for r in rows[1:])
    rep.check("several stages, perturbed: error / (M^3 delta)", f"{worst:.3f}", "<= 1", worst <= 1)
    # negative control: Bob uses Alice's control matrices, which move the labels the same way as hers, instead of their
    # transposes; the partitions stay exact mirrors
    best = 2.0
    for t in itertools.product(*[range(q) for q in qs]):
        WA = compose([(digit_ops(qs, j)[0], digit_ops(qs, j)[1], "A") for j in range(M)], t)
        WB = compose([(digit_ops(qs, j)[0], digit_ops(qs, j)[1], "B") for j in range(M)], t)
        WA = np.kron(WA, np.eye(dAp))
        v = np.einsum("ia,abe,jb->ije", WA, psi.reshape(dA, D, dE), WB, optimize=True).reshape(-1)
        v = v.reshape(D, D, dAp, D, D, dE).transpose(0, 1, 3, 4, 2, 5).reshape(-1)
        best = min(best, distance_to_phi(v, D, D, D, dAp * dE))
    rep.check("negative control: Bob's controls not transposed, best target", f"{best:.3f}", ">= 0.5 (far from Phi)",
              best >= 0.5)


# ---------------------------------------------------------------- 3. cuts across the parties
def cuts_across(rep, g):
    worst, worst_mc, n = 0.0, 0.0, 0
    for (dA, dB, dE) in ((3, 3, 2), (4, 3, 3), (5, 5, 1)):
        for q in (2, 3, 4, 8):
            for eps in (0.3, 3e-2, 3e-3):
                noise = g.normal(size=dA * dB * dE) + 1j * g.normal(size=dA * dB * dE)
                R = unitary(dA, g)
                if dA == dB:
                    # a near-maximally entangled vector and a perturbed transpose of R: a small mirror error
                    phi = np.kron(np.eye(dA).reshape(-1), np.eye(dE)[0]) / math.sqrt(dA)
                    psi = phi + eps * noise / np.linalg.norm(noise)
                    S = sl.expm(1j * eps * herm(g.normal(size=(dB, dB)) + 1j * g.normal(size=(dB, dB)))) @ R.T
                else:
                    psi = noise
                    S = unitary(dB, g)
                psi = psi / np.linalg.norm(psi)
                TR, vR = sl.schur(R, output="complex")
                TS, vS = sl.schur(S, output="complex")
                tR = (np.angle(np.diag(TR)) / (2 * np.pi)) % 1.0
                tS = (np.angle(np.diag(TS)) / (2 * np.pi)) % 1.0
                P = psi.reshape(dA, dB, dE)
                coeff = np.einsum("ai,bj,abe->ije", vR.conj(), vS.conj(), P)
                wts = np.sum(np.abs(coeff) ** 2, axis=2)
                lhs = 0.0
                for i in range(dA):
                    for j in range(dB):
                        dist = abs(tR[i] - tS[j])
                        dist = min(dist, 1 - dist)
                        lhs += 2 * wts[i, j] * min(1.0, q * dist)
                RA = np.kron(np.kron(R, np.eye(dB)), np.eye(dE))
                SB = np.kron(np.kron(np.eye(dA), S), np.eye(dE))
                rhs = q / 2 * np.linalg.norm((RA - SB) @ psi)
                worst = max(worst, lhs / rhs)
                n += 1
                if eps != 3e-2:
                    continue
                # Monte Carlo over offsets, with the spectral projections built explicitly as operators on Psi
                acc = 0.0
                trials = 200
                for th in g.random(trials) / q:
                    for a in range(q):
                        lo = th + a / q
                        inA = [((tR[i] - lo) % 1.0) < 1 / q for i in range(dA)]
                        inB = [((tS[j] - lo) % 1.0) < 1 / q for j in range(dB)]
                        PRa = vR[:, inA] @ vR[:, inA].conj().T
                        PSb = vS[:, inB] @ vS[:, inB].conj().T
                        diff = np.einsum("ai,ibe->abe", PRa, P) - np.einsum("bj,aje->abe", PSb, P)
                        acc += np.linalg.norm(diff) ** 2
                mc = acc / trials
                worst_mc = max(worst_mc, abs(mc - lhs) - 0.2 * lhs)
    rep.check(f"cuts across the parties, {n} cases: E sum_a ||(P_a(R_A) - P_a(S_B)) Psi||^2 / ((q/2)||(R_A - S_B) Psi||)",
              f"{worst:.3f}", 1, worst <= 1 + 1e-12)
    rep.check("cuts across the parties: |Monte Carlo average - exact expectation| - 0.2 exact, largest",
              f"{worst_mc:.3f}", "<= 0.05", worst_mc <= 0.05)


# ---------------------------------------------------------------- 4. Fourier lamps, non-dyadic dilations
def fourier_lamp(q, m, depth):
    words = list(itertools.product(range(q), repeat=depth))
    position = {w: i for i, w in enumerate(words)}
    d = len(words)
    h = m ** depth - 1
    turns = np.array([m * sum(b[j] * m ** (depth - 1 - j) for j in range(depth)) / (q * h) for b in words])
    y = np.diag(np.exp(2j * np.pi * turns))
    x = np.zeros((d, d), dtype=complex)
    for w in words:
        x[position[w], position[w[1:] + w[:1]]] = 1
    lamps = []
    for j in range(q):
        v = np.exp(2j * np.pi * j * np.arange(q) / q) / math.sqrt(q)
        lamps.append(np.kron(np.eye(q) - 2 * np.outer(v, v.conj()), np.eye(q ** (depth - 1))))
    dil = float(np.linalg.norm(x @ y @ x.conj().T - np.linalg.matrix_power(y, m), 2))
    row = np.eye(d, dtype=complex)
    for z in lamps:
        row = row @ z
    row_err = float(np.linalg.norm(row + np.eye(d), 2))
    sq = max(float(np.linalg.norm(z @ z - np.eye(d), 2)) for z in lamps)
    lamp_err = max(float(np.linalg.norm(y @ lamps[j] @ y.conj().T - lamps[(j + 1) % q], 2)) for j in range(q))
    return dil, row_err, sq, lamp_err, 4 * math.pi * (q - 1) / (q * h)


def lamps(rep):
    worst_exact, worst_ratio, n = 0.0, 0.0, 0
    for q, m, depth in ((2, 6, 3), (2, 10, 3), (4, 12, 2), (8, 24, 2), (4, 20, 2)):
        dil, row_err, sq, lamp_err, bound = fourier_lamp(q, m, depth)
        worst_exact = max(worst_exact, dil, row_err, sq)
        worst_ratio = max(worst_ratio, lamp_err / bound)
        n += 1
    rep.check(f"Fourier lamps, {n} non-dyadic (q, m): dilation, row and squares exact (largest error)", f"{worst_exact:.1e}", 0,
              worst_exact < 1e-10)
    rep.check("Fourier lamps: lamp conjugacy defect / (4 pi (q-1)/(q(m^L - 1)))", f"{worst_ratio:.3f}", 1, worst_ratio <= 1 + 1e-9)


# ---------------------------------------------------------------- 5. depth arithmetic
def depths(rep):
    families = [[(2, 6)], [(4, 12), (2, 10)], [(2, 6), (4, 12), (8, 24), (2, 10)], [(2, 2)] * 6, [(4, 8), (2, 3 * 4)]]
    ok_all, worst_rem, n = True, 0.0, 0
    for fam in families:
        fam = [(q, m) for q, m in fam if m % q == 0]
        alpha = sum(math.log2(q) / math.log2(m) for q, m in fam) / 2
        for x in (2 ** k for k in range(10, 20)):
            Ls = [math.floor((x - 32 * math.log2(2 + x)) / math.log2(m)) for q, m in fam]
            ok = min(Ls) >= 1
            # log2(e m^L) = -x + L log2 m <= -32 log2(2 + x)
            ok &= all(L * math.log2(m) - x <= -32 * math.log2(2 + x) + 1e-9 for L, (q, m) in zip(Ls, fam))
            logD = sum(L * math.log2(q) for L, (q, m) in zip(Ls, fam))
            rem = logD - (2 * alpha * x - 64 * alpha * math.log2(2 + x))
            ok &= rem >= -sum(math.log2(q) for q, m in fam) - 1e-9 and rem <= 1e-9
            worst_rem = max(worst_rem, -rem)
            ok_all &= ok
            n += 1
    rep.check(f"depth arithmetic, {n} (family, x) pairs: L_i >= 1, e m_i^L_i <= (2+x)^-32, remainder in [-sum r_i, 0]",
              "all hold" if ok_all else "violated", "all hold", ok_all)


# ---------------------------------------------------------------- 6. the coefficient set
def coefficients(rep):
    ok = True
    for a, b, m in ((1, 3, 6), (5, 2, 10), (7, 4, 2), (3, 5, 12)):
        lhs = a / (2 * math.log2(m ** b))
        rhs = Fraction(a, b) * 1 / (2 * math.log2(m))
        ok &= abs(lhs - float(rhs)) < 1e-12
    for a, b in ((1, 1), (3, 7), (22, 5), (1, 9)):
        ok &= abs(2 * a / (2 * math.log2(2 ** b)) - a / b) < 1e-12
    rep.check("rational multiples and every positive rational: a/(2 log m^b) = (a/b)/(2 log m), 2a/(2 log 2^b) = a/b",
              "exact" if ok else "violated", "exact", ok)
    lam6 = math.log2(6)
    worst = 0.0
    for target in (0.017, 0.31, 1.0, 2.718, 9.5):
        best = min(abs(target - a / (2 * b * lam6)) for b in range(1, 200) for a in [max(1, round(2 * b * lam6 * target))])
        worst = max(worst, best)
    rep.check("the set a/(2 b log 6), b < 200: distance to five targets in (0, 10)", f"{worst:.2e}", "<= 1e-2", worst <= 1e-2)
    lam3 = math.log2(3)
    ok = True
    for target in (0.03, 0.1, 0.2, 0.4, 0.49):
        for b in (100, 1000, 10000):
            r = math.floor(2 * target * (b + lam3))
            coeff = r / (2 * (b + lam3))
            ok &= 1 <= r <= b and 0 <= target - coeff <= 1 / (2 * (b + lam3)) + 1e-14
            ok &= (3 * 2 ** b) % (2 ** r) == 0 if b <= 1000 else True
    rep.check("one-copy family r/(2(b + log 3)), m = 3 2^b, q = 2^r: 1 <= r <= b, q | m, error < 1/(2(b + log 3))",
              "all hold" if ok else "violated", "all hold", ok)


# ---------------------------------------------------------------- 7. the balanced construction of Proposition extended
def fnorm(a):
    """Normalized Frobenius norm ||a||_f = ||a||_2 / sqrt(dim)."""
    return float(np.linalg.norm(a) / math.sqrt(a.shape[0]))


def adiag(a, b):
    """The block matrix [[0, a], [b, 0]] for square blocks of one size."""
    n = a.shape[0]
    out = np.zeros((2 * n, 2 * n), dtype=complex)
    out[:n, n:] = a
    out[n:, :n] = b
    return out


def random_involution(d, g):
    h = g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))
    w, v = np.linalg.eigh(herm(h))
    return (v * np.where(w >= 0, 1.0, -1.0)) @ v.conj().T


def literal_lift(inv, ys):
    """The l lifts of Slofstra's Proposition 4.8, in order: at step j every involutive generator of the current presentation
    becomes rho(x) + I, f_j is the exchange, v_j = [[0, rho(y_j)^*], [rho(y_j), 0]], and every later y_i becomes rho(y_i) + rho(y_i).
    Returns the images of the source generators (y_j read as f_j v_j) and the exchanges f_1, ..., f_l."""
    cur, later = dict(inv), list(ys)
    for j in range(len(ys)):
        eye = np.eye(later[0].shape[0], dtype=complex)
        y = later[0]
        cur = {n: sl.block_diag(M, eye) for n, M in cur.items()}
        cur[f"f{j}"] = adiag(eye, eye)
        cur[f"v{j}"] = adiag(y.conj().T, y)
        later = [sl.block_diag(M, M) for M in later[1:]]
    images = {n: cur[n] for n in inv}
    for j in range(len(ys)):
        images[f"y{j}"] = cur[f"f{j}"] @ cur[f"v{j}"]
    return images, [cur[f"f{j}"] for j in range(len(ys))]


def swap_word(i, swaps, size):
    """g_i: the product of the exchanges over the binary digits of i, in decreasing order."""
    g = np.eye(size, dtype=complex)
    for j in reversed(range(len(swaps))):
        if (i >> j) & 1:
            g = g @ swaps[j]
    return g


def off_blocks(M, d):
    nb = M.shape[0] // d
    return max((float(np.abs(M[i * d:(i + 1) * d, j * d:(j + 1) * d]).max()) for i in range(nb) for j in range(nb) if i != j),
               default=0.0)


def balanced_marker(rep, g):
    worst_ratio, attained, worst_trace, mult_ok, worst_l1 = math.inf, 0.0, math.inf, True, 0.0
    worst_block, worst_comm, worst_parity = 0.0, 0.0, 0.0
    for l in (1, 2, 3):
        for d in (1, 2, 3):
            inv = {"a": random_involution(d, g), "b": random_involution(d, g), "w": -np.eye(d, dtype=complex)}
            ys = [unitary(d, g) for _ in range(l)]
            images, swaps = literal_lift(inv, ys)
            size = d * 2 ** l
            W, t = images["w"], swaps[-1]
            # negative control: the literal marker cannot be exchanged to its negative when l >= 2
            if l == 1:
                worst_l1 = max(worst_l1, fnorm(t @ W @ t + W))
            else:
                signs = np.real(np.diag(W))
                mult_ok &= int((signs < 0).sum()) == d and int((signs > 0).sum()) == (2 ** l - 1) * d
                hw = 2 * math.sqrt(1 - 2.0 ** (1 - l))
                tr = 2 * (1 - 2.0 ** (1 - l))
                attained = max(attained, abs(fnorm(t @ W @ t + W) / hw - 1))
                cands = [unitary(size, g) for _ in range(60)]
                nb = 2 ** l
                perms = itertools.permutations(range(nb)) if nb <= 4 else (g.permutation(nb) for _ in range(200))
                for perm in perms:
                    P = np.zeros((size, size), dtype=complex)
                    for i, pi in enumerate(perm):
                        P[pi * d:(pi + 1) * d, i * d:(i + 1) * d] = np.eye(d)
                    cands.append(P)
                for T in cands:
                    r = fnorm(T @ W @ T.conj().T + W)
                    worst_ratio = min(worst_ratio, r / hw)
                    worst_trace = min(worst_trace, r / tr)
            # positive check: the balanced construction
            for M in list(images.values()):
                worst_block = max(worst_block, off_blocks(M, d))
            k = 2 ** (l - 1)
            G = [swap_word(i, swaps, size) for i in range(k)]
            for i in range(k):
                worst_block = max(worst_block, float(np.abs(G[i][i * d:(i + 1) * d, :d] - np.eye(d)).max()))
            Ws = [Gi @ W @ Gi.conj().T for Gi in G]
            for i, Wi in enumerate(Ws):
                target = np.eye(size, dtype=complex)
                target[i * d:(i + 1) * d, i * d:(i + 1) * d] *= -1
                worst_comm = max(worst_comm, float(np.abs(Wi - target).max()))
                for Wj in Ws:
                    worst_comm = max(worst_comm, float(np.abs(Wi @ Wj - Wj @ Wi).max()))
                for Gj in G:
                    for M in images.values():
                        S = Gj @ M @ Gj.conj().T
                        worst_comm = max(worst_comm, float(np.abs(Wi @ S - S @ Wi).max()))
            p = np.eye(size, dtype=complex)
            for Wi in Ws:
                p = p @ Wi
            worst_parity = max(worst_parity, abs(float(np.real(np.trace(p)))), float(np.abs(t @ p @ t + p).max()))
    rep.check("balanced marker, negative control: literal lifts, l = 2, 3, marker multiplicities d and (2^l - 1) d",
              "as stated" if mult_ok else "violated", "as stated", mult_ok)
    rep.check("negative control: min over the last exchange, block permutations and random unitaries of "
              "||T W T^* + W||_f / (2 (1 - 2^{1-l})^{1/2})", f"{worst_ratio:.6f}", ">= 1 (no approximate sign relation)",
              worst_ratio >= 1 - 1e-9 and worst_trace >= 1 - 1e-9)
    rep.info(f"the last exchange attains the Hoffman-Wielandt bound (largest relative gap {attained:.1e}); at l = 2 it is sqrt 2; "
             f"trace bound 2 (1 - 2^(1-l)) also holds (smallest ratio {worst_trace:.3f})")
    rep.check("l = 1: the literal lift is (-I_d) + I_d and the exchange gives T W T^* = -W (largest residual)", f"{worst_l1:.1e}", 0,
              worst_l1 < 1e-12)
    rep.check("balanced construction, l = 1, 2, 3: source generators block-diagonal, g_i carries block 0 to block i "
              "(largest off-block entry or deviation)", f"{worst_block:.1e}", 0, worst_block < 1e-12)
    rep.check("balanced construction: W_i = -I on block i only, [W_i, W_j] = 0, [W_i, g_j s g_j^*] = 0 (largest entry)",
              f"{worst_comm:.1e}", 0, worst_comm < 1e-12)
    rep.check("balanced construction: parity p = W_0 ... W_{k-1} has trace 0 and f_l p f_l = -p (largest error)",
              f"{worst_parity:.1e}", 0, worst_parity < 1e-12)


def k_model(m):
    """The flat model of the companion's Example 11.11 on C^(2m): Y diagonal, A, B = -A, C = -I, X a permutation matrix."""
    d = 2 * m
    Y = np.diag(np.exp(2j * np.pi * np.arange(d) / d))
    A = np.zeros((d, d), dtype=complex)
    for j in range(d):
        A[(j + m) % d, j] = 1
    X = np.zeros((d, d), dtype=complex)
    for j in range(d):
        X[j, 2 * (j % m) + j // m] = 1
    return {"a": A, "b": -A, "c": -np.eye(d, dtype=complex), "x": X, "y": Y}


def k_relators(ops):
    a, b, c, x, y = (ops[n] for n in "abcxy")
    inv = lambda u: u.conj().T  # noqa: E731
    return [x @ y @ inv(x) @ inv(y) @ inv(y), x @ c @ inv(x) @ inv(c), y @ a @ inv(y) @ inv(b), y @ b @ inv(y) @ inv(a),
            inv(c) @ a @ b, a @ a, b @ b, c @ c]


def reversal_permutation(X):
    """An involutive permutation matrix tau with tau X tau = X^{-1}, for a permutation matrix X (reflect each cycle)."""
    n = X.shape[0]
    sigma = [int(np.argmax(np.abs(X[:, i]))) for i in range(n)]
    tau, seen = list(range(n)), [False] * n
    for start in range(n):
        if seen[start]:
            continue
        cyc, i = [], start
        while not seen[i]:
            seen[i] = True
            cyc.append(i)
            i = sigma[i]
        for s, ci in enumerate(cyc):
            tau[ci] = cyc[(-s) % len(cyc)]
    T = np.zeros((n, n), dtype=complex)
    for i in range(n):
        T[tau[i], i] = 1
    return T


def sv_reversal(rep):
    worst_exact, worst_excess, ms = 0.0, -math.inf, range(2, 13)
    for m in ms:
        ops = k_model(m)
        d = 2 * m
        eye = np.eye(d, dtype=complex)
        X, Y = ops["x"], ops["y"]
        tau = reversal_permutation(X)
        f = adiag(eye, eye)
        v = adiag(Y.conj().T, Y)
        fp, vp = sl.block_diag(tau, tau), sl.block_diag(tau @ X, tau @ X)
        lift = {"a": sl.block_diag(ops["a"], eye), "b": sl.block_diag(ops["b"], eye), "c": sl.block_diag(ops["c"], eye),
                "x": fp @ vp, "y": f @ v}
        two = np.eye(2 * d, dtype=complex)
        errs = [tau @ tau - eye, tau @ X @ tau - X.conj().T, fp @ fp - two, vp @ vp - two, fp @ vp - sl.block_diag(X, X),
                f @ lift["c"] @ f + lift["c"], f @ lift["x"] - lift["x"] @ f]
        worst_exact = max(worst_exact, max(float(np.abs(e).max()) for e in errs))
        for r_model, r_lift in zip(k_relators(ops), k_relators(lift)):
            worst_excess = max(worst_excess, fnorm(r_lift - two) - fnorm(r_model - eye))
    rep.check(f"Slofstra-Vidick group, companion flat models d = 2m, m = 2..12: tau involutive with tau X tau = X^-1, f', v' "
              "involutions with f'v' = X + X, t c t = -c, [f, x] = 0 (largest error)", f"{worst_exact:.1e}", 0, worst_exact < 1e-12)
    rep.check("Slofstra-Vidick group: lifted relator residual minus model residual, every relator of K (largest)",
              f"{worst_excess:.1e}", "<= 0 up to rounding 1e-12", worst_excess <= 1e-12)


def main():
    rep = Report("The coefficient family: cyclic extraction, cuts with q arcs, non-dyadic lamps, depths, coefficient set, balanced marker")
    g = rng(97)
    one_stage(rep)
    stage_inequality(rep, rng(131))  # its own generator, so that the later checks draw the same numbers as before
    several_stages(rep, g)
    cuts_across(rep, g)
    lamps(rep)
    depths(rep)
    coefficients(rep)
    balanced_marker(rep, rng(211))  # its own generator, so that the earlier checks draw the same numbers as before
    sv_reversal(rep)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
