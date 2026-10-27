"""Checks for the two explicit linear system games built by the cover construction of Appendix C.

Sources (distinguished central involution w):
  K4 = <a, b, c=w, y, x | a^2, b^2, w^2, abw = 1 with a, b, w commuting, y a y^-1 = b, y b y^-1 = a, y w y^-1 = w,
        x w x^-1 = w, x y x^-1 = y^4>   (the binary lamp source with m = 4; rank exponent 1/4);
  L48 = <z_0..z_3, w, y, x | z_j^2, w^2, z_0 z_1 z_2 z_3 w = 1 with all of them commuting, y z_j y^-1 = z_{j+1 mod 4},
        y w y^-1 = w, x w x^-1 = w, x y x^-1 = y^8>   (four lamps, m = 8; rank exponent 1/3).

For each source the script rebuilds the linear system from scratch:
  1. elimination of y and then of x (y = f v, x = f' v' with constant swaps f and reflections v); the power relation
     x y x^-1 = y^m becomes x v x^-1 = v (f v)^(m-1) together with [x, f] = 1, written as involutive conjugacies;
  2. the cover: the conjugates W_i of w by the swaps, their pairwise commutation, their commutation with every conjugated
     source word, the parity variable p = W_0 W_1, and a variable u with p u = J and t p t = u for the last swap t;
  3. the nice-conjugacy stage and 4. the six-equation gadget stage.
Each variable carries a 16 x 16 block-monomial image whose entries are +- words in the source generators. The script
checks exactly, in the negative quotient of the source (normal forms in the base group and Britton's lemma for the HNN
extension by x), that every image is an involution, that every equation holds and that the variables of every equation
commute; that each W_i has a single negative block; and that each conjugated source word restricts to its generator on
its block. It then plays the strategy of the return theorem (sequential measurements, last record flipped on the wrong
parity) on every equation and outcome branch, on flat source models of two sizes, and reports the losing probability
against the source precision."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmath
import itertools
import math
from functools import lru_cache

import numpy as np

from common import Report


def free_reduce(wd):
    out = []
    for a in wd:
        if out and out[-1] == -a:
            out.pop()
        else:
            out.append(a)
    return tuple(out)


class Mono:
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

    def neg(self):
        return Mono(self.p, [(-s, w) for s, w in self.w])


def ident(n=1):
    return Mono(range(n), [(1, ())] * n)


def diag(a, b):
    return Mono(a.p + tuple(a.n + j for j in b.p), a.w + b.w)


def anti(a, b):
    return Mono(tuple(a.n + j for j in a.p) + b.p, a.w + b.w)


def prod(items, n):
    out = ident(n)
    for M in items:
        out = out @ M
    return out


class Builder:
    """The cover construction for an ordered extended homogeneous-linear-plus-conjugacy presentation with two
    non-involutive generators y < x and the power relation x y x^-1 = y^m."""

    def __init__(self, letters, involutions, rows, conj, power, negative):
        self.letters = letters
        self.m = {n: Mono([0], [(1, (i,))]) for n, i in letters.items()}
        self.inv = list(involutions)
        self.noninv = ["y", "x"]
        self.rows = [(list(r), 1) for r in rows]
        self.c0 = []
        self.c1 = list(conj)
        self.powers = [power]
        self.original = {n: (n,) for n in self.m}
        self.negative = negative
        self.swaps = []
        self.serial = 0

    def new(self, stem, image):
        self.serial += 1
        name = f"{stem}_{self.serial}"
        self.m[name] = image
        self.inv.append(name)
        return name

    def word(self, wd):
        return prod([self.m[n] for n in wd], next(iter(self.m.values())).n)

    def eliminate(self):
        while self.noninv:
            y, remain = self.noninv[0], self.noninv[1:]
            old, size = self.m, self.m[y].n
            self.m = {n: (diag(v, ident(size)) if n in self.inv else diag(v, v)) for n, v in old.items() if n != y}
            f = self.new("flip_" + y, anti(ident(size), ident(size)))
            v = self.new("reflect_" + y, anti(old[y].star(), old[y]))
            self.swaps.append(f)
            self.original = {s: tuple(t for n in wd for t in ((f, v) if n == y else (n,)))
                             for s, wd in self.original.items()}
            c1 = []
            for u, j, k in self.c1:
                if u == y:
                    z = self.new("transport", self.m[v] @ self.m[j] @ self.m[v].star())
                    self.c0 += [(v, j, z), (f, z, k)]
                else:
                    c1.append((u, j, k))
            for u in remain:
                c1.append((u, f, f))
            powers = []
            for u, j, L in self.powers:
                if j != y:
                    powers.append((u, j, L))
                    continue
                middle = v if L % 2 else f
                for k in reversed(range(L - 1)):
                    a = v if k % 2 == 0 else f
                    nxt = self.new("palindrome", self.m[a] @ self.m[middle] @ self.m[a].star())
                    self.c0.append((a, middle, nxt))
                    middle = nxt
                c1.append((u, v, middle))
            self.c1, self.powers, self.noninv = c1, powers, remain

    def conjugate(self, wd, start, memo):
        key = (tuple(wd), start)
        if key in memo:
            return memo[key]
        if not wd:
            return start
        mid = self.conjugate(wd[1:], start, memo)
        a = wd[0]
        res = self.new("conj", self.m[a] @ self.m[mid] @ self.m[a].star())
        self.c0.append((a, mid, res))
        memo[key] = res
        return res

    def cover(self):
        size = next(iter(self.m.values())).n
        paths, queue = {0: ()}, [0]
        while queue:
            i = queue.pop(0)
            for f in self.swaps:
                j = self.m[f].p[i]
                if j not in paths:
                    paths[j] = (f,) + paths[i]
                    queue.append(j)
        assert len(paths) == size
        self.paths, memo = paths, {}
        self.W = [self.conjugate(paths[i], self.negative, memo) for i in range(size)]
        for i in range(size):
            for j in range(i):
                self.c0.append((self.W[i], self.W[j], self.W[j]))
        for j in range(size):
            g = paths[j]
            for s, wd in self.original.items():
                cw = g + wd + tuple(reversed(g))
                for Wn in self.W:
                    if len(cw) == 1:
                        self.c0.append((cw[0], Wn, Wn))
                    else:
                        self.c0.append((cw[0], self.conjugate(cw[1:], Wn, memo), Wn))
        t = self.swaps[-1]
        half = size // 2
        assert set(self.m[t].p[:half]) == set(range(half, size))
        p = self.new("parity", prod([self.m[self.W[i]] for i in range(half)], size))
        self.rows.append(([p] + [self.W[i] for i in range(half)], 1))
        u = self.new("opposite", self.m[p].neg())
        self.rows.append(([p, u], -1))
        self.c0.append((t, p, u))

    def finish(self):
        old, names = self.m, list(self.inv)
        size = next(iter(old.values())).n
        self.m = {n: diag(v, v) for n, v in old.items()}
        g = self.new("nice_flip", anti(ident(size), ident(size)))
        nice, prime, double, bar = [], {}, {}, {}
        for a in names:
            prime[a] = self.new("prime", diag(old[a], ident(size)))
            double[a] = self.new("double", diag(ident(size), old[a]))
            bar[a] = self.new("bar", anti(old[a], old[a]))
            self.rows += [([a, prime[a], double[a]], 1), ([a, g, bar[a]], 1)]
            nice.append((g, prime[a], double[a]))
        for a, b, c in self.c0:
            s = self.new("nice_s", diag(old[b], old[c]))
            sp = self.new("nice_sp", diag(old[c], old[b]))
            self.rows += [([prime[b], double[c], s], 1), ([prime[c], double[b], sp], 1)]
            nice += [(bar[a], prime[b], double[c]), (bar[a], double[b], prime[c])]
        size *= 2
        self.m = {n: diag(v, v) for n, v in self.m.items()}
        pre = {n: Mono(v.p[:size], v.w[:size]) for n, v in self.m.items()}
        for a, b, c in nice:
            A, B, C = pre[a], pre[b], pre[c]
            images = [anti(A, A), anti(ident(size), ident(size)), anti(B, B), anti(B @ A, A @ B),
                      diag(B @ A @ B, A), diag(B @ C, ident(size)), diag(B, C)]
            ys = [self.new("gadget", M) for M in images]
            self.rows += [([a, ys[0], ys[1]], 1), ([b, ys[1], ys[2]], 1), ([ys[2], ys[3], ys[4]], 1),
                          ([a, ys[4], ys[5]], 1), ([c, ys[5], ys[6]], 1), ([ys[0], ys[3], ys[6]], 1)]


# ---------------------------------------------------------------- the two sources and their negative quotients
def hnn_scalar(word, base_letter, x_letter, m, base_mul, unit_base):
    """+1 or -1 if the word is a scalar in the negative quotient, else 0. Base elements are (lamp data, power of y);
    the stable letter x conjugates y to y^m (and fixes the scalar w = -1)."""
    sign, bases, xs = 1, [unit_base], []
    for a in word:
        if abs(a) == x_letter:
            xs.append(1 if a > 0 else -1)
            bases.append(unit_base)
            continue
        s, term = base_letter(a)
        sign *= s
        s, bases[-1] = base_mul(bases[-1], term)
        sign *= s
    while True:
        for i in range(len(xs) - 1):
            data, n = bases[i + 1]
            if data != unit_base[0] or xs[i] == xs[i + 1]:
                continue
            if xs[i] == 1:
                mid = (unit_base[0], m * n)
            elif n % m == 0:
                mid = (unit_base[0], n // m)
            else:
                continue
            s1, tmp = base_mul(bases[i], mid)
            s2, tmp = base_mul(tmp, bases[i + 2])
            sign *= s1 * s2
            bases[i:i + 3] = [tmp]
            del xs[i:i + 2]
            break
        else:
            break
    return 0 if (xs or bases[0] != unit_base) else sign


# K4: letters a=1, b=2, w=3, y=4, x=5; in the negative quotient w = -1, b = -a, y a y^-1 = -a.
def k4_base_letter(a):
    k = abs(a)
    if k == 1:
        return 1, (1, 0)
    if k == 2:
        return -1, (1, 0)
    if k == 3:
        return -1, (0, 0)
    return 1, (0, 1 if a > 0 else -1)


def k4_base_mul(u, v):
    (p, n), (q, k) = u, v
    return (-1 if (n * q) % 2 else 1), (p ^ q, n + k)


@lru_cache(maxsize=None)
def k4_scalar(word):
    return hnn_scalar(word, k4_base_letter, 5, 4, k4_base_mul, (0, 0))


# L48: letters z_j = j+1 (j < 4), w = 5, y = 6, x = 7; the lamps commute, z_0 z_1 z_2 z_3 = -1, y z_j y^-1 = z_{j+1}.
Q_L = 4


def lamp_canonical(mask):
    if mask & (1 << (Q_L - 1)):
        return -1, mask ^ ((1 << Q_L) - 1)
    return 1, mask


def lamp_rotate(mask, k):
    k %= Q_L
    return ((mask << k) | (mask >> (Q_L - k))) & ((1 << Q_L) - 1)


def l48_base_mul(u, v):
    (mask, k), (other, l) = u, v
    s, new = lamp_canonical(mask ^ lamp_rotate(other, k))
    return s, (new, k + l)


def l48_base_letter(a):
    k = abs(a)
    if k <= Q_L:
        s, mask = lamp_canonical(1 << (k - 1))
        return s, (mask, 0)
    if k == Q_L + 1:
        return -1, (0, 0)
    return 1, (0, 1 if a > 0 else -1)


@lru_cache(maxsize=None)
def l48_scalar(word):
    return hnn_scalar(word, l48_base_letter, 7, 8, l48_base_mul, (0, 0))


def k4_builder():
    letters = {"a": 1, "b": 2, "w": 3, "y": 4, "x": 5}
    return Builder(letters, ["a", "b", "w"], [["a", "b", "w"]],
                   [("y", "a", "b"), ("y", "b", "a"), ("y", "w", "w"), ("x", "w", "w")], ("x", "y", 4), "w"), k4_scalar


def l48_builder():
    letters = {"z0": 1, "z1": 2, "z2": 3, "z3": 4, "w": 5, "y": 6, "x": 7}
    conj = [("y", f"z{j}", f"z{(j + 1) % 4}") for j in range(4)] + [("y", "w", "w"), ("x", "w", "w")]
    return Builder(letters, ["z0", "z1", "z2", "z3", "w"], [["z0", "z1", "z2", "z3", "w"]], conj, ("x", "y", 8),
                   "w"), l48_scalar


def source_relators(name):
    if name == "K4":
        return [(1, 1), (2, 2), (3, 3), (1, 2, 3), (4, 1, -4, -2), (4, 2, -4, -1), (4, 3, -4, -3), (5, 3, -5, -3),
                (5, 4, -5) + (-4,) * 4]
    rels = [(j, j) for j in range(1, 6)] + [(1, 2, 3, 4, 5)]
    rels += [(6, j, -6, -(j % 4 + 1)) for j in range(1, 5)] + [(6, 5, -6, -5), (7, 5, -7, -5), (7, 6, -7) + (-6,) * 8]
    return rels


def build_and_check(rep, label, make):
    b, scalar = make()
    rels_ok = all(scalar(r) == 1 for r in source_relators(label))
    b.eliminate()
    b.cover()
    size = next(iter(b.m.values())).n
    table = all(b.m[Wn].p == tuple(range(size)) and [s * scalar(w) for s, w in b.m[Wn].w] ==
                [-1 if i == k else 1 for i in range(size)] for k, Wn in enumerate(b.W))
    split = True
    for name, wd in b.original.items():
        for i in range(size):
            Mw = b.word(b.paths[i] + wd + tuple(reversed(b.paths[i])))
            s, w = Mw.w[i]
            split &= Mw.p[i] == i and scalar(w + (-b.letters[name],)) == s
    rep.check(f"{label}: source relators are 1 in the negative-quotient engine", rels_ok, True, rels_ok)
    rep.check(f"{label}: each W_i is diagonal with a single negative block (four blocks)", table and size == 4, True,
              table and size == 4)
    rep.check(f"{label}: each conjugated source word restricts to its generator on its block", split, True, split)
    b.finish()
    in_rows = sorted({n for row, _ in b.rows for n in row})
    nb = next(iter(b.m.values())).n
    sq = all(b.m[n].p == tuple(range(nb)) or True for n in in_rows)

    def unit(M):
        return M.p == tuple(range(M.n)) and all(scalar(w) == s for s, w in M.w)

    sq = all(unit(b.m[n] @ b.m[n]) for n in in_rows)
    rows_ok = all(unit(prod([b.m[n] for n in row], nb) if s > 0 else prod([b.m[n] for n in row], nb).neg())
                  for row, s in b.rows)
    nc, comm_ok = 0, True
    for row, _ in b.rows:
        for x, y in itertools.combinations(row, 2):
            comm_ok &= unit(b.m[x] @ b.m[y] @ b.m[x].star() @ b.m[y].star())
            nc += 1
    counts = (len(in_rows), len(b.rows), max(len(r) for r, _ in b.rows), nb)
    rep.info(f"{label}: {counts[0]} variables, {counts[1]} equations, maximum arity {counts[2]}, {nb} blocks "
             f"(the central generator J is not a variable)")
    rep.check(f"{label}: all images are involutions, all equations hold, all {nc} commutators are trivial",
              sq and rows_ok and comm_ok, True, sq and rows_ok and comm_ok)
    return b, in_rows, counts


# ---------------------------------------------------------------- complete play on flat source models
def k4_model(n):
    d, m = 2 ** n, 4
    den = m ** n - 1
    idx = np.arange(d)
    numer = np.array([sum(((int(i) >> (n - 1 - j)) & 1) * (m // 2) * m ** (n - 1 - j) for j in range(n)) for i in idx])
    A = np.zeros((d, d), complex)
    A[idx ^ (1 << (n - 1)), idx] = 1  # flip the first digit
    X = np.zeros((d, d), complex)
    X[idx, ((idx << 1) & (d - 1)) + (idx >> (n - 1))] = 1  # X_{b, Rb} = 1
    Y = np.diag(np.exp(2j * np.pi * numer / den))
    ops = {1: A, 2: -A, 3: -np.eye(d), 4: Y, 5: X}
    return ops, 2 * math.sin(math.pi / (2 * den))


def l48_model(L):
    q, m = 4, 8
    d, h = q ** L, m ** L - 1
    words = list(itertools.product(range(q), repeat=L))
    index = {w: i for i, w in enumerate(words)}
    t = [(m // q) * sum(bj * m ** (L - 1 - j) for j, bj in enumerate(w)) / h for w in words]
    Y = np.diag(np.exp(2j * np.pi * np.array(t)))
    X = np.zeros((d, d), complex)
    for w in words:
        X[index[w], index[w[1:] + w[:1]]] = 1
    zeta = cmath.exp(2j * math.pi / q)
    ops = {}
    for j in range(q):
        v = np.array([zeta ** (j * a) for a in range(q)]) / math.sqrt(q)
        Zj = np.eye(q) - 2 * np.outer(v, v.conj())
        ops[j + 1] = np.kron(Zj, np.eye(q ** (L - 1)))
    ops[5] = -np.eye(d)
    ops[6] = Y
    ops[7] = X
    e = max(np.linalg.norm((ops[6] @ ops[j + 1] @ ops[6].conj().T - ops[(j + 1) % q + 1]), 2) for j in range(q))
    return ops, e


def play(b, in_rows, ops):
    d = ops[1].shape[0]
    full = dict(ops)
    full.update({-k: v.conj().T for k, v in ops.items()})
    cache = {}

    def word(w):
        if w not in cache:
            M = np.eye(d, dtype=complex)
            for a in w:
                M = M @ full[a]
            cache[w] = M
        return cache[w]

    nb = 16
    mats = {}
    for name in in_rows:
        M = b.m[name]
        mats[name] = (np.array(M.p), np.stack([s * word(w) for s, w in M.w]))
    N = nb * d

    def mul(X, Y):
        p, A = X
        q, B = Y
        return q[p], A @ B[p]

    def inner(X, Y):
        p, A = X
        q, B = Y
        mask = p == q
        return np.vdot(A[mask], B[mask]) / N

    one = (np.arange(nb), np.stack([np.eye(d, dtype=complex)] * nb))
    total, worst_bad = 0.0, 0.0
    for row, sign in b.rows:
        k = len(row)
        Bs = [mats[x] for x in row]
        terms, subsets = [one], [()]
        for j, B in enumerate(Bs):
            terms = terms + [mul(T, B) for T in terms]
            subsets = subsets + [s + (j,) for s in subsets]
        G = np.array([[inner(X, Y) for Y in terms] for X in terms])
        H = [np.array([[inner(X, mul(B, Y)) for Y in terms] for X in terms]) for B in Bs]
        loss = bad = 0.0
        for alpha in itertools.product((-1, 1), repeat=k):
            coeff = np.array([math.prod(alpha[j] for j in s) for s in subsets]) / 2 ** k
            p = float((coeff @ G @ coeff).real)
            report = alpha
            if math.prod(alpha) != sign:
                bad += p
                report = alpha[:-1] + (-alpha[-1],)
            for j, sg in enumerate(report):
                corr = float((coeff @ H[j] @ coeff).real)
                loss += (p - sg * corr) / 2 / k
        worst_bad = max(worst_bad, bad)
        total += loss / len(b.rows)
    return N, total


def main():
    rep = Report("Explicit linear system games from the cover construction")
    results = {}
    for label, make, expect, models, exponent in [
            ("K4", k4_builder, (5206, 4325, 3, 16), [k4_model(2), k4_model(3)], 0.25),
            ("L48", l48_builder, (6798, 5653, 5, 16), [l48_model(1), l48_model(2)], 1 / 3)]:
        b, in_rows, counts = build_and_check(rep, label, make)
        rep.check(f"{label}: size (variables, equations, arity, blocks)", counts, expect, counts == expect)
        recs = []
        for ops, e in models:
            N, loss = play(b, in_rows, ops)
            recs.append((N, loss, e))
            rep.info(f"{label}: Schmidt rank {N}, source precision {e:.3e}, complete-play loss {loss:.3e}, "
                     f"loss/e^2 {loss / e ** 2:.3e}")
        ratios = [loss / e ** 2 for _, loss, e in recs]
        slope = math.log(recs[1][0] / recs[0][0]) / math.log(recs[0][1] / recs[1][1])
        ok = max(ratios) < 1 and max(ratios) / min(ratios) < 2
        rep.check(f"{label}: complete play, loss / (source precision)^2", ", ".join(f"{x:.2e}" for x in ratios),
                  "stable and small", ok)
        rep.check(f"{label}: observed rank exponent between the two sizes", f"{slope:.4f}",
                  f"approaching {exponent:.4f}", abs(slope - exponent) < 0.05)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
