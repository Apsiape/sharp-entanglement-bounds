"""Checks for Section 5: polar correction and its negative control, the corner map (exact identities, and perturbed
models with the state kept), and a complete instance of the compiler on the certified class: the eight-block lift of
the group <a, b, c, y | abc, [a,b], [a,c], [b,c], y a y^-1 b^-1, y b y^-1 a^-1, y c y^-1 c^-1> with distinguished c,
checked exactly in its negative quotient and run as a game strategy."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import math

import numpy as np
import scipy.linalg as sl
from scipy.optimize import minimize

from common import PX, PY, PZ, Report, comm, entropy_bits, herm, hs, random_state, rng, round_unitary, sqrtm_psd, unitary
from check_return import c_s, sequential


def polar_unitary(a):
    u, _, vh = np.linalg.svd(a)
    return u @ vh


def polar_checks(rep, g):
    worst_r = worst_l = worst_c = -1.0
    n = 0
    kinds = ["random", "geometric", "lowrank", "pure", "diag"]
    for d in (2, 3, 4, 8, 16):
        for t in range(40):
            a = (g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))) / math.sqrt(d)
            if t % 3 == 0:
                a[:, -1] = 0  # singular
            if t % 3 == 1:
                a = unitary(d, g) @ (np.eye(d) + 0.01 * herm(g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))))
            rho = random_state(d, g, kinds[t % 5])
            xi = sqrtm_psd(rho)
            u = polar_unitary(a)
            dr = hs((a.conj().T @ a - np.eye(d)) @ xi)
            dl = hs((a @ a.conj().T - np.eye(d)) @ xi)
            worst_r = max(worst_r, hs((u - a) @ xi) - dr)
            worst_l = max(worst_l, hs(xi @ (u - a)) - dl)
            worst_c = max(worst_c, hs(comm(u, xi)) - hs(comm(a, xi)) - dr - dl)
            n += 1
    rep.check(f"polar correction, right bound, {n} instances: max excess", f"{worst_r:.2e}", 0, worst_r <= 1e-10)
    rep.check(f"polar correction, left bound: max excess", f"{worst_l:.2e}", 0, worst_l <= 1e-10)
    rep.check(f"polar correction, commutator bound: max excess", f"{worst_c:.2e}", 0, worst_c <= 1e-10)
    best = -1e9
    for side in (0, 1):
        for start in range(4):
            def neg(p):
                a = p[:4].reshape(2, 2) + 1j * p[4:8].reshape(2, 2)
                h = np.array([[p[8], p[9] + 1j * p[10]], [p[9] - 1j * p[10], -p[8]]])
                rho = sl.expm(h)
                rho /= np.trace(rho).real
                xi = sqrtm_psd(rho)
                u = polar_unitary(a)
                if side == 0:
                    return -(hs((u - a) @ xi) - hs((a.conj().T @ a - np.eye(2)) @ xi))
                return -(hs(xi @ (u - a)) - hs((a @ a.conj().T - np.eye(2)) @ xi))

            res = minimize(neg, g.normal(size=11), method="L-BFGS-B", bounds=[(-3, 3)] * 11,
                           options={"maxiter": 150})
            best = max(best, -res.fun)
    rep.check("polar correction, optimizer (8 starts): max excess", f"{best:.2e}", 0, best <= 1e-7)
    a = np.array([[0, 0], [1, 0]], complex)
    xi = np.diag([1.0, 0.0]).astype(complex)
    u = polar_unitary(a)
    ok = hs((a.conj().T @ a - np.eye(2)) @ xi) < 1e-15 and abs(hs(xi @ (u - a)) - 1) < 1e-12
    rep.check("negative control: right defect 0 but left correction 1", ok, True, ok)


def corner(ops, w, z):
    p = (np.eye(len(w)) - w) / 2
    return {k: p @ v @ p + z @ p @ v @ p @ z for k, v in ops.items()}


def corner_exact(rep, g):
    worst = 0.0
    for d in (1, 2, 3, 4):
        w = np.kron(-PZ, np.eye(d))  # an involution; z = X (x) I gives z w z = -w
        z = np.kron(PX, np.eye(d))
        for _ in range(10):
            uu = sl.block_diag(unitary(d, g), unitary(d, g))  # commute with w
            vv = sl.block_diag(unitary(d, g), unitary(d, g))
            f = corner({"u": uu, "v": vv, "uv": uu @ vv, "w": w, "1": np.eye(2 * d)}, w, z)
            worst = max(worst, hs(f["uv"] - f["u"] @ f["v"]), hs(f["w"] + np.eye(2 * d)), hs(f["1"] - np.eye(2 * d)),
                        hs(f["u"].conj().T @ f["u"] - np.eye(2 * d)))
    rep.check("corner map: multiplicative, unital, unitary, and sends w to -1 (exact models)", f"{worst:.2e}", 0,
              worst < 1e-12)


K_RELATORS = ["xyXYY", "xcXC", "yaYB", "ybYA", "Cab", "aa", "bb", "cc"]


def word(w, gens):
    out = np.eye(gens["a"].shape[0], dtype=complex)
    for ch in w:
        m = gens[ch.lower()]
        out = out @ (m.conj().T if ch.isupper() else m)
    return out


def k_model(m):
    d = 2 * m
    y = np.diag(np.exp(2j * math.pi * np.arange(d) / d))
    a = np.zeros((d, d), complex)
    x = np.zeros((d, d), complex)
    for j in range(d):
        a[j, (j + m) % d] = 1
        x[j, 2 * (j % m) + j // m] = 1
    return {"a": a, "b": -a, "c": -np.eye(d, dtype=complex), "x": x, "y": y}


def k_error(gens, rho, extra=()):
    xi = sqrtm_psd(rho)
    eye = np.eye(len(rho))
    errs = [hs((word(r, gens) - eye) @ xi) for r in K_RELATORS]
    errs += [hs(comm(v, xi)) for v in gens.values()]
    errs += [hs((gens["c"] + eye) @ xi)]
    errs += list(extra)
    return max(errs)


def corner_perturbed(rep, g):
    """Models of a group H containing K, with c mapped to W = c (+) I and a witness z = swap with [W, z] = -1.
    The corner map returns a model of K with c -> -1 on the same, non-flat, state."""
    worst = 0.0
    for m in (1, 2, 4, 8, 16):
        d = 2 * m
        for scale in (0.0, 1e-5, 1e-3, 1e-2, 0.1):
            base = k_model(m)
            ops = {k: sl.block_diag(v, np.eye(d)) for k, v in base.items()}
            z = np.block([[np.zeros((d, d)), np.eye(d)], [np.eye(d), np.zeros((d, d))]]).astype(complex)
            if scale:
                ops = {k: sl.expm(1j * scale * herm(g.normal(size=(2 * d, 2 * d)) + 1j * g.normal(size=(2 * d, 2 * d))))
                       @ v for k, v in ops.items()}
                z = sl.expm(1j * scale * herm(g.normal(size=(2 * d, 2 * d)) + 1j * g.normal(size=(2 * d, 2 * d)))) @ z
            rho = (1 - scale) * np.eye(2 * d) / (2 * d) + scale * random_state(2 * d, g, "geometric")
            xi = sqrtm_psd(rho)
            w = ops["c"]
            eye = np.eye(2 * d)
            inp = max(k_error_input(ops, rho),
                      hs((z @ z - eye) @ xi), hs(comm(z, xi)), hs((w @ z @ w.conj().T @ z.conj().T + eye) @ xi))
            raw = corner({k: v for k, v in ops.items()}, w, z)
            out_ops = {k: polar_unitary(v) for k, v in raw.items()}
            out_ops["c"] = -eye.astype(complex)
            out = k_error(out_ops, rho)
            worst = max(worst, out / inp)
    rep.check("corner map on perturbed models, same density matrix: max output/input error", f"{worst:.3f}",
              "bounded", worst < 10)


def k_error_input(ops, rho):
    """Relator defects and state commutators of the input model; c is not required to be -1 there."""
    xi = sqrtm_psd(rho)
    eye = np.eye(len(rho))
    errs = [hs((word(r, ops) - eye) @ xi) for r in K_RELATORS]
    errs += [hs(comm(v, xi)) for v in ops.values()]
    return max(errs)


# ---- the complete compiler instance, with block-monomial matrices over words in a, b, c, y ----
A_, B_, C_, Y_ = 1, 2, 3, 4  # letters; negative numbers are inverses


def reduce_free(wd):
    out = []
    for letter in wd:
        if out and out[-1] == -letter:
            out.pop()
        else:
            out.append(letter)
    return tuple(out)


class Mono:
    """An n x n block-monomial matrix: row i has its one nonzero block in column perm[i], equal to sign * word."""

    def __init__(self, perm, entries):
        self.perm = tuple(perm)
        self.entries = tuple((int(s), reduce_free(wd)) for s, wd in entries)
        self.n = len(self.perm)

    def __matmul__(self, other):
        perm, entries = [], []
        for i in range(self.n):
            j = self.perm[i]
            sa, wa = self.entries[i]
            sb, wb = other.entries[j]
            perm.append(other.perm[j])
            entries.append((sa * sb, wa + wb))
        return Mono(perm, entries)

    def adjoint(self):
        perm, entries = [0] * self.n, [None] * self.n
        for i, j in enumerate(self.perm):
            s, wd = self.entries[i]
            perm[j] = i
            entries[j] = (s, tuple(-t for t in reversed(wd)))
        return Mono(perm, entries)

    def scaled(self, s):
        return Mono(self.perm, [(s * a, wd) for a, wd in self.entries])


def eye_m(n=1):
    return Mono(range(n), [(1, ())] * n)


def letter(x):
    return Mono([0], [(1, (x,))])


def dsum(a, b):
    return Mono(list(a.perm) + [a.n + j for j in b.perm], list(a.entries) + list(b.entries))


def adiag(a, b):
    return Mono([a.n + j for j in a.perm] + list(b.perm), list(a.entries) + list(b.entries))


def prod(items):
    out = eye_m(items[0].n)
    for m in items:
        out = out @ m
    return out


def normal_form(sign, wd):
    """In the negative quotient c = -1, b = -a, a^2 = 1 and y a = -a y, so every word is sign * a^p * y^n."""
    parity, exponent = 0, 0
    for x in wd:
        k = abs(x)
        if k in (A_, B_):
            if k == B_:
                sign = -sign
            if exponent % 2:
                sign = -sign
            parity ^= 1
        elif k == C_:
            sign = -sign
        else:
            exponent += 1 if x > 0 else -1
    return sign, parity, exponent


def is_identity(m):
    return m.perm == tuple(range(m.n)) and all(normal_form(s, wd) == (1, 0, 0) for s, wd in m.entries)


def build_instance():
    gens = {"a": A_, "b": B_, "c": C_}
    # step 1: remove the non-involutive generator y = f v; one new involution per conjugacy relation by y
    old = {k: dsum(letter(v), eye_m()) for k, v in gens.items()}
    old["f"] = adiag(eye_m(), eye_m())
    old["v"] = adiag(letter(Y_).adjoint(), letter(Y_))
    rows = [(["a", "b", "c"], 1)]
    triples = []
    for j, (left, right) in enumerate((("a", "b"), ("b", "a"), ("c", "c"))):
        name = "z" + str(j)
        old[name] = dsum(eye_m(), letter(Y_) @ letter(gens[left]) @ letter(Y_).adjoint())
        triples += [("v", left, name), ("f", name, right)]
    # step 2: the distinguished involution becomes a commutator, with u = -c and the swap t
    old["t"] = adiag(eye_m(), eye_m())
    old["u"] = old["c"].scaled(-1)
    rows.append((["c", "u"], -1))
    triples.append(("t", "c", "u"))
    # step 3: make every conjugacy relation nice
    names = list(old)
    nice = {k: dsum(v, v) for k, v in old.items()}
    nice["g"] = adiag(eye_m(2), eye_m(2))
    nice_rows = list(rows)
    nice_triples = []
    for k in names:
        p1, p2, bar = k + "'", k + "''", k + "_bar"
        nice[p1] = dsum(old[k], eye_m(2))
        nice[p2] = dsum(eye_m(2), old[k])
        nice[bar] = adiag(old[k], old[k])
        nice_rows += [([k, p1, p2], 1), ([k, "g", bar], 1)]
        nice_triples.append(("g", p1, p2))
    for j, (a, b, c) in enumerate(triples):
        s1, s2 = "s" + str(j), "s" + str(j) + "'"
        nice[s1] = dsum(old[b], old[c])
        nice[s2] = dsum(old[c], old[b])
        nice_rows += [([b + "'", c + "''", s1], 1), ([c + "'", b + "''", s2], 1)]
        nice_triples += [(a + "_bar", b + "'", c + "''"), (a + "_bar", b + "''", c + "'")]
    # step 4: six linear equations in seven new variables for each nice conjugacy relation
    lift = {k: dsum(v, v) for k, v in nice.items()}
    eqs = list(nice_rows)
    for j, (a, b, c) in enumerate(nice_triples):
        av, bv, cv = nice[a], nice[b], nice[c]
        ys = ["q%d_%d" % (j, k) for k in range(1, 8)]
        lift[ys[0]] = adiag(av, av)
        lift[ys[1]] = adiag(eye_m(4), eye_m(4))
        lift[ys[2]] = adiag(bv, bv)
        lift[ys[3]] = adiag(bv @ av, av @ bv)
        lift[ys[4]] = dsum(bv @ av @ bv, av)
        lift[ys[5]] = dsum(bv @ cv, eye_m(4))
        lift[ys[6]] = dsum(bv, cv)
        eqs += [([a, ys[0], ys[1]], 1), ([b, ys[1], ys[2]], 1), ([ys[2], ys[3], ys[4]], 1),
                ([a, ys[4], ys[5]], 1), ([c, ys[5], ys[6]], 1), ([ys[0], ys[3], ys[6]], 1)]
    return lift, eqs


def instance_checks(rep, g):
    lift, eqs = build_instance()
    checks = []
    for k, v in lift.items():
        checks.append(is_identity(v @ v))
    for row, sign in eqs:
        checks.append(is_identity(prod([lift[k] for k in row]).scaled(sign)))
        for a, b in itertools.combinations(row, 2):
            checks.append(is_identity(lift[a] @ lift[b] @ lift[a].adjoint() @ lift[b].adjoint()))
    wit = (lift["c"] @ lift["t"] @ lift["c"].adjoint() @ lift["t"].adjoint()).scaled(-1)
    checks.append(is_identity(wit))
    rep.info(f"instance: {len(lift)} variables, {len(eqs)} equations, {len(checks)} exact checks, "
             f"{lift['a'].n} blocks, longest entry {max(len(wd) for v in lift.values() for _, wd in v.entries)}")
    rep.check("instance size: 223 variables and 180 equations", (len(lift), len(eqs)), (223, 180),
              (len(lift), len(eqs)) == (223, 180))
    rep.check(f"instance: all {len(checks)} relations hold exactly in the negative quotient", all(checks), True,
              all(checks) and len(checks) == 942)
    # numerical second engine: evaluate on 2 x 2 source matrices and run the strategy of the return theorem
    worst = 0.0
    for scale in (0.0, 1e-5, 1e-3, 1e-2):
        src = {A_: PZ, B_: -sl.expm(1j * scale * PX) @ PZ @ sl.expm(-1j * scale * PX), C_: -np.eye(2, dtype=complex),
               Y_: np.exp(0.37j) * sl.expm(1j * scale * PY) @ PX}
        rho = (np.eye(2) + scale * (0.13 * PX + 0.17 * PY + 0.09 * PZ)) / 2
        vals = {k: evaluate(v, src) for k, v in lift.items()}
        big = np.kron(np.eye(8) / 8, rho)
        e = model_error(vals, eqs, big)
        bs = {k: round_unitary(v) for k, v in vals.items()}
        loss = 0.0
        for row, sign in eqs:
            _, _, dis, _ = sequential([bs[k] for k in row], big, sign)
            loss = max(loss, dis.max())
        ent = entropy_bits(big) - entropy_bits(rho)
        rep.info(f"perturbation {scale:g}: model error {e:.3e}, largest pair loss {loss:.3e}, entropy overhead {ent:.3f}")
        if e > 0:
            worst = max(worst, loss / (c_s(3) * e ** 2))
        elif loss > 1e-12:
            worst = float("inf")
    rep.check("instance: pair loss <= C_3 e^2 on perturbed models", f"{worst:.2e}", 1, worst <= 1)


def evaluate(m, src):
    d = src[A_].shape[0]
    out = np.zeros((m.n * d, m.n * d), dtype=complex)
    for i, j in enumerate(m.perm):
        s, wd = m.entries[i]
        val = s * np.eye(d, dtype=complex)
        for x in wd:
            val = val @ (src[x] if x > 0 else src[-x].conj().T)
        out[i * d:(i + 1) * d, j * d:(j + 1) * d] = val
    return out


def model_error(vals, eqs, rho):
    xi = sqrtm_psd(rho)
    eye = np.eye(len(rho))
    errs = [hs((v @ v - eye) @ xi) for v in vals.values()] + [hs(comm(v, xi)) for v in vals.values()]
    for row, sign in eqs:
        ops = [vals[k] for k in row]
        errs.append(hs((np.linalg.multi_dot(ops + [eye]) - sign * eye) @ xi))
        for a, b in itertools.combinations(ops, 2):
            errs.append(hs((a @ b @ a.conj().T @ b.conj().T - eye) @ xi))
    return max(errs)


def main():
    rep = Report("Compiling group profiles into games")
    g = rng(17)
    polar_checks(rep, g)
    corner_exact(rep, g)
    corner_perturbed(rep, g)
    instance_checks(rep, g)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
