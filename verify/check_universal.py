"""Checks for the universal compiler into boolean constraint games.

For two small source presentations with a distinguished central involution w, this script builds the boolean
constraint system of the three-sector construction (one-hot frame, coordinate swaps, pointwise blank support,
conjugacy chains for the relators, the parity row V_w A e_3 = -1, then the nice-conjugacy and six-equation gadget
stages), and checks, exactly in the negative quotient of the source:
  * every variable's 12-block image is an involution;
  * the variables of every context commute, and every forbidden assignment projector is zero;
  * sigma(F(s)) = I_12 (x) s for every source generator, where F(s) = P1 U_s P1 + A P1 U_s P1 A + T P1 U_s P1 T;
  * the inverse-letter sector of a relator is a nontrivial word, and is not tested;
  * the normal form (one-hot replaced by a parity row and three binary exclusions) has the same allowed assignments.
It then checks numerically: the lower map on perturbed models with a non-flat state (source defects linear in the
target precision, the state kept); the boolean return estimate K_k e^2 on random models; and the odd-tensor control
(tensor cubes keep every parity row and break the frame with probability 2/9).
Sources:
  S1 = <x, y, z, w | w^2, [w,x], [w,y], [w,z], x y z>, negative quotient C[F(x, y)] with z = (xy)^-1, w = -1;
  S2 = <u, v, q, w | w^2, [w,u], [w,v], [w,q], u v q^-1, u^3>, negative quotient that of C3 * Z with q = uv, w = -1."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import math

import numpy as np
import scipy.linalg as sl

from common import Report, comm, entropy_bits, hs, random_state, rng, sqrtm_psd, unitary


def free_reduce(wd):
    out = []
    for a in wd:
        if out and out[-1] == -a:
            out.pop()
        else:
            out.append(a)
    return tuple(out)


class Mono:
    """Block-monomial matrix: row i has its one nonzero block in column perm[i], equal to sign * word."""

    def __init__(self, perm, entries):
        self.p = tuple(perm)
        self.w = tuple((int(s), free_reduce(wd)) for s, wd in entries)
        self.n = len(self.p)

    def __matmul__(self, b):
        return Mono([b.p[j] for j in self.p], [(self.w[i][0] * b.w[j][0], self.w[i][1] + b.w[j][1])
                                               for i, j in enumerate(self.p)])

    def star(self):
        p, w = [0] * self.n, [None] * self.n
        for i, j in enumerate(self.p):
            s, v = self.w[i]
            p[j], w[j] = i, (s, tuple(-a for a in reversed(v)))
        return Mono(p, w)


def ident(n=1):
    return Mono(range(n), [(1, ())] * n)


def diag(a, b):
    return Mono(a.p + tuple(a.n + j for j in b.p), a.w + b.w)


def anti(a, b):
    return Mono(tuple(a.n + j for j in a.p) + b.p, a.w + b.w)


def sign_matrix(signs):
    return Mono(range(3), [(s, ()) for s in signs])


def perm_matrix(perm):
    return Mono(perm, [(1, ())] * 3)


def reflection(letter):
    """V_s = [[0, s, 0], [s^-1, 0, 0], [0, 0, 1]]."""
    return Mono((1, 0, 2), [(1, (letter,)), (1, (-letter,)), (1, ())])


class Source:
    def __init__(self, names, relators, negative, normal_form):
        self.names, self.relators, self.negative, self.nf = names, relators, negative, normal_form


def nf_s1(sign, wd):
    # letters: x=1, y=2, z=3, w=4; z -> y^-1 x^-1, w -> -1; free group on x, y
    out = []
    for a in wd:
        if abs(a) == 4:
            sign = -sign
        elif a == 3:
            out += [-2, -1]
        elif a == -3:
            out += [1, 2]
        else:
            out.append(a)
    return sign, free_reduce(out)


def nf_s2(sign, wd):
    # letters: u=1, v=2, q=3, w=4; q -> u v, w -> -1; free product C3 * Z, u-syllables taken mod 3
    expanded = []
    for a in wd:
        if abs(a) == 4:
            sign = -sign
        elif a == 3:
            expanded += [1, 2]
        elif a == -3:
            expanded += [-2, -1]
        else:
            expanded.append(a)
    syll = []
    for a in expanded:
        kind, power = abs(a), (1 if a > 0 else -1)
        if syll and syll[-1][0] == kind:
            power += syll.pop()[1]
        if kind == 1:
            power %= 3
        if power:
            syll.append((kind, power))
    return sign, tuple(syll)


S1 = Source(("x", "y", "z", "w"), [(4, 4), (4, 1, -4, -1), (4, 2, -4, -2), (4, 3, -4, -3), (1, 2, 3)], "w", nf_s1)
S2 = Source(("u", "v", "q", "w"), [(4, 4), (4, 1, -4, -1), (4, 2, -4, -2), (4, 3, -4, -3), (1, 2, -3), (1, 1, 1)], "w",
            nf_s2)


class Compiler:
    def __init__(self, src):
        self.src = src
        self.m = {"e1": sign_matrix((-1, 1, 1)), "e2": sign_matrix((1, -1, 1)), "e3": sign_matrix((1, 1, -1)),
                  "A": perm_matrix((1, 0, 2)), "T": perm_matrix((2, 1, 0))}
        self.m.update({"V_" + s: reflection(i) for i, s in enumerate(src.names, 1)})
        self.first = list(self.m)
        self.serial = 0
        self.rows = [(["V_" + src.negative, "A", "e3"], -1)]
        self.triples = []
        self.boolean = [(["e1", "e2", "e3"], [(-1, 1, 1), (1, -1, 1), (1, 1, -1)], "one-hot")]
        for n in ["A"] + ["V_" + s for s in src.names]:
            self.triples += [(n, "e1", "e2"), (n, "e2", "e1"), (n, "e3", "e3")]
            self.boolean.append((["e3", n], [(1, 1), (1, -1), (-1, 1)], "blank support"))
        self.triples += [("T", "e1", "e3"), ("T", "e3", "e1"), ("T", "e2", "e2")]
        for rel in src.relators:
            word = []
            for a in rel:
                pair = ("V_" + src.names[abs(a) - 1], "A")
                word += list(pair) if a > 0 else list(reversed(pair))
            self.impose_commutation(word, "T")
        self.n_chain = len(self.first) - len(src.names) - 5

    def new(self, stem, M):
        self.serial += 1
        name = f"{stem}_{self.serial}"
        self.first.append(name)
        self.m[name] = M
        return name

    def impose_commutation(self, word, start):
        """Conjugate T through the word by a chain of involutive conjugacies, ending at T itself."""
        mid = start
        for index, a in enumerate(reversed(word)):
            if index == len(word) - 1:
                self.triples.append((a, mid, start))
            else:
                nxt = self.new("chain", self.m[a] @ self.m[mid] @ self.m[a].star())
                self.triples.append((a, mid, nxt))
                mid = nxt

    def finish(self):
        old, names, D = dict(self.m), list(self.first), 3
        self.m = {n: diag(M, M) for n, M in old.items()}
        g = "g"
        self.m[g] = anti(ident(D), ident(D))
        p, dd, bar, nice = {}, {}, {}, []
        for a in names:
            p[a], dd[a], bar[a] = a + "'", a + "''", a + "_bar"
            self.m[p[a]] = diag(old[a], ident(D))
            self.m[dd[a]] = diag(ident(D), old[a])
            self.m[bar[a]] = anti(old[a], old[a])
            self.rows += [([a, p[a], dd[a]], 1), ([a, g, bar[a]], 1)]
            nice.append((g, p[a], dd[a]))
        for k, (a, b, c) in enumerate(self.triples):
            s1, s2 = f"s{k}", f"s{k}'"
            self.m[s1] = diag(old[b], old[c])
            self.m[s2] = diag(old[c], old[b])
            self.rows += [([p[b], dd[c], s1], 1), ([p[c], dd[b], s2], 1)]
            nice += [(bar[a], p[b], dd[c]), (bar[a], dd[b], p[c])]
        pre = dict(self.m)
        self.m = {n: diag(M, M) for n, M in pre.items()}
        D = 6
        for k, (a, b, c) in enumerate(nice):
            A, B, C = pre[a], pre[b], pre[c]
            images = [anti(A, A), anti(ident(D), ident(D)), anti(B, B), anti(B @ A, A @ B), diag(B @ A @ B, A),
                      diag(B @ C, ident(D)), diag(B, C)]
            ys = [f"y{k}_{t}" for t in range(1, 8)]
            for y, M in zip(ys, images):
                self.m[y] = M
            self.rows += [([a, ys[0], ys[1]], 1), ([b, ys[1], ys[2]], 1), ([ys[2], ys[3], ys[4]], 1),
                          ([a, ys[4], ys[5]], 1), ([c, ys[5], ys[6]], 1), ([ys[0], ys[3], ys[6]], 1)]
        self.contexts = [(row, [a for a in itertools.product((-1, 1), repeat=len(row)) if math.prod(a) == s], "parity")
                         for row, s in self.rows] + self.boolean
        return self


def is_identity(M, nf):
    return M.p == tuple(range(M.n)) and all(nf(s, w) == (1, ()) for s, w in M.w)


def polynomial_zero(terms, nf):
    cells = {}
    for coeff, M in terms:
        for i, (s, w) in enumerate(M.w):
            ns, nw = nf(s, w)
            key = (i, M.p[i], nw)
            cells[key] = cells.get(key, 0) + coeff * ns
    return not any(abs(v) > 1e-12 for v in cells.values())


def projector_terms(row, answers, mats, n):
    out = [(1.0, ident(n))]
    for name, a in zip(row, answers):
        out = out + [(c * a, M @ mats[name]) for c, M in out]
    return out


def exact_checks(rep, label, src):
    b = Compiler(src).finish()
    nf = src.nf
    n = next(iter(b.m.values())).n
    squares = all(is_identity(M @ M, nf) for M in b.m.values())
    comms = forb = 0
    ok_comm = ok_forb = True
    for row, allowed, _ in b.contexts:
        for x, y in itertools.combinations(row, 2):
            ok_comm &= is_identity(b.m[x] @ b.m[y] @ b.m[x].star() @ b.m[y].star(), nf)
            comms += 1
        for alpha in itertools.product((-1, 1), repeat=len(row)):
            if alpha not in allowed:
                ok_forb &= polynomial_zero(projector_terms(row, alpha, b.m, n), nf)
                forb += 1
    nb = sum(1 for c in b.contexts if c[2] != "parity")
    rep.info(f"{label}: {len(b.m)} variables, {len(b.contexts)} contexts ({nb} boolean), {n} blocks, "
             f"max arity {max(len(c[0]) for c in b.contexts)}")
    rep.check(f"{label}: all {len(b.m)} images are involutions in the negative quotient", squares, True, squares)
    rep.check(f"{label}: all {comms} context commutators are trivial", ok_comm, True, ok_comm)
    rep.check(f"{label}: all {forb} forbidden assignment projectors vanish", ok_forb, True, ok_forb)
    rep.check(f"{label}: amplification and arity", (n, max(len(c[0]) for c in b.contexts)), (12, 3),
              n == 12 and max(len(c[0]) for c in b.contexts) == 3)
    # sigma(F(s)) = I_12 (x) s
    E1, A, T = b.m["e1"], b.m["A"], b.m["T"]
    split = True
    for k, s in enumerate(src.names, 1):
        U = b.m["V_" + s] @ A
        corner = [(1, U), (-1, E1 @ U), (-1, U @ E1), (1, E1 @ U @ E1)]  # 4 P1 U P1
        folded = corner + [(c, A @ X @ A) for c, X in corner] + [(c, T @ X @ T) for c, X in corner]
        split &= polynomial_zero(folded + [(-4, Mono(range(12), [(1, (k,))] * 12))], nf)
    rep.check(f"{label}: sigma(F(s)) = I_12 (x) s for every source generator", split, True, split)
    # the inverse-letter sector of each relator is left untested; for x y z (resp. u v q^-1) it is nontrivial
    rel = src.relators[4]
    word = ident(3)
    for a in rel:
        U = reflection(abs(a)) @ perm_matrix((1, 0, 2))
        word = word @ (U if a > 0 else U.star())
    middle = nf(*word.w[1])
    first = nf(*word.w[0])
    ok = first == (1, ()) and middle != (1, ()) and is_identity(word @ perm_matrix((2, 1, 0)) @ word.star()
                                                                @ perm_matrix((2, 1, 0)), nf)
    rep.check(f"{label}: relator image diag(r, r~, 1) has r = 1, r~ != 1 and commutes with T", ok, True, ok)
    # normal form: one-hot <-> parity e1 e2 e3 = -1 and no two of e1, e2, e3 equal to -1
    triples = list(itertools.product((-1, 1), repeat=3))
    onehot = {(-1, 1, 1), (1, -1, 1), (1, 1, -1)}
    nform = {a for a in triples if math.prod(a) == -1 and sum(x < 0 for x in a) <= 1}
    rep.check(f"{label}: normal form allows exactly the one-hot assignments", nform == onehot, True, nform == onehot)
    return b


# ---- numerical: lower map on perturbed models, with the state kept
def evaluate(M, src_ops):
    d = next(iter(src_ops.values())).shape[0]
    out = np.zeros((M.n * d, M.n * d), complex)
    cache = {}
    for i, (s, wd) in enumerate(M.w):
        if wd not in cache:
            val = np.eye(d, dtype=complex)
            for a in wd:
                val = val @ (src_ops[a] if a > 0 else src_ops[-a].conj().T)
            cache[wd] = val
        out[i * d:(i + 1) * d, M.p[i] * d:(M.p[i] + 1) * d] = s * cache[wd]
    return out


def sign_round(U):
    H = (U + U.conj().T) / 2
    w, V = np.linalg.eigh(H)
    return (V * np.where(w >= 0, 1.0, -1.0)) @ V.conj().T


def lower_map_checks(rep, b, src, g):
    """Source model -> target model (upper map) -> lower map -> source model, all on the same state."""
    d0 = 3
    omega = np.exp(2j * np.pi / 3)
    if src is S1:
        X0, Y0 = unitary(d0, g), unitary(d0, g)
        base = {1: X0, 2: Y0, 3: np.linalg.inv(X0 @ Y0), 4: -np.eye(d0)}
    else:
        U0 = np.diag([1, omega, omega ** 2])
        V0 = unitary(d0, g)
        base = {1: U0, 2: V0, 3: U0 @ V0, 4: -np.eye(d0)}
    k = 2
    sigma = np.diag([0.8, 0.2])
    ops0 = {a: np.kron(M, np.eye(k)) for a, M in base.items()}
    rho = np.kron(np.eye(d0) / d0, sigma)
    xi = sqrtm_psd(rho)
    ratios = []
    for scale in (1e-2, 1e-3, 1e-4):
        ops = {a: sl.expm(1j * scale * (lambda H: (H + H.conj().T) / 2)(g.normal(size=(2 * d0, 2 * d0))
                                                                        + 1j * g.normal(size=(2 * d0, 2 * d0)))) @ M
               for a, M in ops0.items()}
        D = ops[1].shape[0]
        src_err = max([hs((word_eval(r, ops) - np.eye(D)) @ xi) for r in src.relators] +
                      [hs(comm(M, xi)) for M in ops.values()] + [hs((ops[4] + np.eye(D)) @ xi)])
        big_xi = np.kron(np.eye(12) / math.sqrt(12), xi)
        mats = {name: sign_round(evaluate(M, ops)) for name, M in b.m.items()
                if name in ("e1", "A", "T") or name.startswith("V_")}
        N = 12 * D
        P1 = (np.eye(N) - mats["e1"]) / 2
        rec = {}
        for kk, s in enumerate(src.names, 1):
            U = mats["V_" + s] @ mats["A"]
            Fs = P1 @ U @ P1
            Fs = Fs + mats["A"] @ Fs @ mats["A"] + mats["T"] @ Fs @ mats["T"]
            u, _, vh = np.linalg.svd(Fs)
            rec[kk] = u @ vh
        rec_err = max([hs((word_eval(r, rec) - np.eye(N)) @ big_xi) for r in src.relators] +
                      [hs(comm(M, big_xi)) for M in rec.values()] + [hs((rec[4] + np.eye(N)) @ big_xi)])
        ratios.append(rec_err / src_err)
    ok = max(ratios) < 100 and max(ratios) / max(min(ratios), 1e-300) < 10
    rep.check("lower map after upper map, perturbed non-flat models: recovered defect / source defect",
              ", ".join(f"{r:.2f}" for r in ratios), "stable, O(1)", ok)


def word_eval(rel, ops):
    D = next(iter(ops.values())).shape[0]
    out = np.eye(D, dtype=complex)
    for a in rel:
        out = out @ (ops[a] if a > 0 else ops[-a].conj().T)
    return out


# ---- numerical: the boolean return estimate
def boolean_return_checks(rep, g):
    worst = 0.0
    cases = 0
    for k in (1, 2, 3):
        Ak = k / 2 + (k - 1) * (k + 2) / 8
        Kk = 2 ** k * (Ak ** 2 + (1 + k / 2) ** 2)
        for trial in range(60):
            d = int(g.choice([2, 3, 4, 6]))
            rho = random_state(d, g, ["random", "geometric", "lowrank", "pure", "nearflat"][trial % 5])
            xi = sqrtm_psd(rho)
            # nearly commuting involutions, perturbed from a common diagonal frame
            V = unitary(d, g)
            Bs = []
            for _ in range(k):
                D0 = np.diag(g.choice([-1.0, 1.0], size=d))
                Hm = g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))
                W = sl.expm(1j * 10 ** g.uniform(-4, -0.5) * (Hm + Hm.conj().T) / 2) @ V
                Bs.append(W @ D0 @ W.conj().T)
            allowed = [a for a in itertools.product((-1, 1), repeat=k) if g.random() < 0.6] or [(1,) * k]
            P = lambda j, a: (np.eye(d) + a * Bs[j]) / 2
            R = {}
            for alpha in itertools.product((-1, 1), repeat=k):
                M = np.eye(d, dtype=complex)
                for j, a in enumerate(alpha):
                    M = M @ P(j, a)
                R[alpha] = M
            e = max([hs(comm(B, xi)) for B in Bs] + [hs(comm(Bs[i], Bs[j]) @ xi) for i in range(k) for j in range(k)]
                    + [hs(R[a] @ xi) for a in R if a not in allowed])
            repair = allowed[0]
            for j in range(k):
                loss = 0.0
                for alpha, M in R.items():
                    answer = alpha if alpha in allowed else repair
                    loss += hs(P(j, -answer[j]) @ xi @ M) ** 2
                worst = max(worst, (loss - 1e-14) / (Kk * e ** 2 + 1e-300))  # floor for 0/0 cases
                cases += 1
    rep.check(f"boolean return, {cases} queries (k <= 3, random states): loss / (K_k e^2)", f"{worst:.3f}", 1,
              worst <= 1 + 1e-9)
    rep.check("K_3 = 221/2", 2 ** 3 * ((3 / 2 + 2 * 5 / 8) ** 2 + (5 / 2) ** 2), 110.5,
              abs(2 ** 3 * ((3 / 2 + 2 * 5 / 8) ** 2 + (5 / 2) ** 2) - 110.5) < 1e-12)


# ---- odd tensor control
def tensor_checks(rep, b, src):
    """Evaluate every variable on an exact one-dimensional model of the source (w = -1); the 12 x 12 images are
    monomial with unimodular entries. Tensor cubes (dimension 1728) are kept as permutations with phases."""
    if src is S1:
        base = {1: np.exp(0.3j), 2: np.exp(-1.1j)}
        base[3] = 1 / (base[1] * base[2])
    else:
        base = {1: np.exp(2j * np.pi / 3), 2: np.exp(0.7j)}
        base[3] = base[1] * base[2]
    base[4] = -1.0

    def scalar(sign, wd):
        v = complex(sign)
        for a in wd:
            v *= base[a] if a > 0 else np.conj(base[-a])
        return v

    mono = {name: (np.array(M.p), np.array([scalar(s, w) for s, w in M.w])) for name, M in b.m.items()}

    def cube(X):
        p, u = X
        P = (p[:, None, None] * 144 + p[None, :, None] * 12 + p[None, None, :]).ravel()
        U = (u[:, None, None] * u[None, :, None] * u[None, None, :]).ravel()
        return P, U

    def mul(X, Y):
        p, u = X
        q, v = Y
        return q[p], u * v[p]

    cubes = {name: cube(X) for name, X in mono.items()}
    n3 = 12 ** 3
    rows_ok = True
    for row, allowed, kind in b.contexts:
        if kind != "parity":
            continue
        P, U = np.arange(n3), np.ones(n3, complex)
        for name in row:
            P, U = mul((P, U), cubes[name])
        s = math.prod(allowed[0])
        rows_ok &= bool(np.array_equal(P, np.arange(n3)) and np.allclose(U, s, atol=1e-9))
    diag_e = [cubes[n][1].real for n in ("e1", "e2", "e3")]  # the frame variables are diagonal sign matrices
    frame_diag = all(np.array_equal(cubes[n][0], np.arange(n3)) for n in ("e1", "e2", "e3"))
    p = float(np.mean((diag_e[0] < 0) & (diag_e[1] < 0) & (diag_e[2] < 0)))
    rep.check("odd tensor control: tensor cubes satisfy every parity row", rows_ok, True, rows_ok)
    rep.check("odd tensor control: tensor cubes violate one-hot with probability 2/9", f"{p:.6f}", "2/9",
              frame_diag and abs(p - 2 / 9) < 1e-12)


def main():
    rep = Report("The universal compiler into boolean constraint games")
    g = rng(31)
    b1 = exact_checks(rep, "S1 (free group, x y z = 1)", S1)
    b2 = exact_checks(rep, "S2 (C3 * Z)", S2)
    lower_map_checks(rep, b1, S1, g)
    lower_map_checks(rep, b2, S2, g)
    boolean_return_checks(rep, g)
    tensor_checks(rep, b2, S2)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
