"""Checks for the lamp sources L(q, m), q = 2^r dividing m, and for the measure-theoretic steps of their profile.

L(q, m) = <z_0..z_{q-1}, w, y, x | z_j^2, w^2, the z_j and w commute, z_0 ... z_{q-1} w = 1, y z_j y^-1 = z_{j+1 mod q},
           x w x^-1 = w, x y x^-1 = y^m>, distinguished central involution w.
1. Fourier-lamp upper models of dimension q^L: lamps are exact commuting involutions with product -I = W, the dilation
   holds exactly, and every lamp conjugacy has operator defect at most 4 pi (q-1)/(q (m^L - 1)).
2. The rotation selector T (graded character projections over the odd-parity orbits): T*T = TT* = (1-w)/2 and
   T y = zeta y T in the group algebra of <z_j, y>, for q = 2, 4; every odd-parity orbit has length divisible by
   2^{v_2(q)}, for q <= 12; and, on perturbed models with non-flat states, the polar correction of T rotates Y by zeta
   in the state at a cost linear in the model precision.
3. The spread bound nu(I) <= (q|I|)^s and the quantization bound for laws invariant under t -> m t and t -> t + 1/q
   (Bernoulli digit laws and a mixture), on the normalized circle R/Z.
4. The shadowing identity: the corrected sequence V_t is an exact orbit, its displacement is bounded by the error
   series, and rotating U_t rotates V_t.
5. Entropy from transport against a spread law, on random states.
6. On perturbed binary lamp models: delta_D <= pi (m+3) e, and delta_S / e bounded.
7. Controls: odd q has an exact one-dimensional model; q = 6 has models of dimension 2^L.
Single-threaded; small matrices only."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmath
import itertools
import math
from fractions import Fraction

import numpy as np
import scipy.linalg as sl

from common import Report, comm, entropy_bits, hs, random_state, rng, sqrtm_psd, unitary


# ---------------------------------------------------------------- 1. upper models
def lamp_model(q, m, L, d_lamps=None):
    """Words b in {0..d-1}^L (d = d_lamps, default q); t_b = (m/d) sum_j b_j m^(L-1-j) / (m^L - 1); X_{b,Rb} = 1;
    Fourier lamps Z_j = (I - 2|v_j><v_j|) on the first digit, v_j(a) = d^(-1/2) zeta^(j a), indices taken mod d."""
    d = d_lamps or q
    h = m ** L - 1
    words = list(itertools.product(range(d), repeat=L))
    index = {b: i for i, b in enumerate(words)}
    t = [Fraction((m // d) * sum(bj * m ** (L - 1 - j) for j, bj in enumerate(b)), h) for b in words]
    n = len(words)
    Y = np.diag([cmath.exp(2j * math.pi * float(x)) for x in t])
    X = np.zeros((n, n), complex)
    for i, b in enumerate(words):
        X[i, index[b[1:] + b[:1]]] = 1
    zeta = cmath.exp(2j * math.pi / d)
    U = {"y": Y, "x": X, "w": -np.eye(n, dtype=complex)}
    for j in range(q):
        v = np.array([zeta ** ((j % d) * a) for a in range(d)]) / math.sqrt(d)
        U[f"z{j}"] = np.kron(np.eye(d) - 2 * np.outer(v, v.conj()), np.eye(d ** (L - 1)))
    rot_ok = all((m * t[i] - t[index[b[1:] + b[:1]]]).denominator == 1 for i, b in enumerate(words))
    return U, rot_ok


def relators(q, m, U):
    inv = lambda M: M.conj().T
    Y, X, W = U["y"], U["x"], U["w"]
    Z = [U[f"z{j}"] for j in range(q)]
    rels = {f"z{j}^2": Z[j] @ Z[j] for j in range(q)}
    rels["w^2"] = W @ W
    gens = Z + [W]
    for i in range(len(gens)):
        for k in range(i):
            rels[f"[g{i},g{k}]"] = gens[i] @ gens[k] @ inv(gens[i]) @ inv(gens[k])
    row = np.eye(Y.shape[0], dtype=complex)
    for M in gens:
        row = row @ M
    rels["row"] = row
    for j in range(q):
        rels[f"y z{j} y^-1 z{(j + 1) % q}^-1"] = Y @ Z[j] @ inv(Y) @ inv(Z[(j + 1) % q])
    rels["x w x^-1 w^-1"] = X @ W @ inv(X) @ inv(W)
    rels["x y x^-1 y^-m"] = X @ Y @ inv(X) @ inv(np.linalg.matrix_power(Y, m))
    return rels


def model_error(q, m, U, xi):
    d = xi.shape[0]
    errs = [hs((R - np.eye(d)) @ xi) for R in relators(q, m, U).values()]
    errs += [hs(comm(V, xi)) for V in U.values()]
    errs.append(hs((U["w"] + np.eye(d)) @ xi))
    return max(errs)


def upper_checks(rep):
    for q, m, L in [(2, 2, 4), (2, 4, 3), (2, 6, 3), (2, 8, 3), (4, 4, 2), (4, 8, 2), (4, 12, 2), (8, 8, 1),
                    (8, 16, 2), (4, 16, 3), (2, 4, 6)]:
        U, rot_ok = lamp_model(q, m, L)
        n = U["y"].shape[0]
        rels = relators(q, m, U)
        bound = 4 * math.pi * (q - 1) / (q * (m ** L - 1))
        exact = all(np.allclose(R, np.eye(n), atol=1e-11) for k, R in rels.items() if not k.startswith("y z"))
        worst = max(np.linalg.norm(R - np.eye(n), 2) for k, R in rels.items() if k.startswith("y z"))
        ok = rot_ok and exact and worst <= bound + 1e-12
        rep.check(f"lamp model q={q} m={m} L={L} (d={n}): exact relations; lamp conjugacy defect", f"{worst:.3e}",
                  f"<= {bound:.3e}", ok)
    worst = 0.0
    for q, m in [(2, 4), (2, 6), (4, 8), (8, 8)]:
        s = math.log(q) / math.log(m)
        for L in range(1, 9):
            e = 4 * math.pi * (q - 1) / (q * (m ** L - 1))
            worst = max(worst, q ** L * e ** s)
    rep.check("lamp models: q^L * e^s stays bounded for L = 1..8", f"{worst:.3f}", "<= 4 pi q",
              worst <= 4 * math.pi * 8)


# ---------------------------------------------------------------- 2. the rotation selector
def rotate(mask, k, q):
    k %= q
    full = (1 << q) - 1
    return ((mask << k) | (mask >> (q - k))) & full if k else mask


def ga_mul(x, y, q):
    """Group algebra of Z_2^q semidirect Z (lamps z_j, y z_j y^-1 = z_{j+1}); elements (mask, n) = z_mask y^n."""
    out = {}
    for (mask, n), u in x.items():
        for (mask2, n2), v in y.items():
            key = (mask ^ rotate(mask2, n, q), n + n2)
            out[key] = out.get(key, 0) + u * v
    return {k: v for k, v in out.items() if abs(v) > 1e-13}


def ga_star(x, q):
    out = {}
    for (mask, n), u in x.items():
        key = (rotate(mask, -n, q), -n)  # (z_mask y^n)^-1 = y^-n z_mask = z_{rot(mask, -n)} y^-n
        out[key] = out.get(key, 0) + u.conjugate()
    return out


def ga_sub(x, y):
    out = dict(x)
    for k, v in y.items():
        out[k] = out.get(k, 0) - v
    return {k: v for k, v in out.items() if abs(v) > 1e-11}


def odd_orbits(q):
    seen, orbits = set(), []
    for chi in itertools.product((1, -1), repeat=q):
        if math.prod(chi) != -1 or chi in seen:
            continue
        orbit, cur = [], chi
        while cur not in orbit:
            orbit.append(cur)
            seen.add(cur)
            cur = cur[-1:] + cur[:-1]  # (sigma chi)_k = chi_{k-1}
        orbits.append(orbit)
    return orbits


def lamp_selector(q):
    d = q & -q
    zeta = cmath.exp(2j * math.pi / d)
    T = {}
    for orbit in odd_orbits(q):
        for a, chi in enumerate(orbit):
            P = {(0, 0): 1.0}
            for j, c in enumerate(chi):
                P = ga_mul(P, {(0, 0): 0.5, (1 << j, 0): 0.5 * c}, q)
            for key, v in P.items():
                T[key] = T.get(key, 0) + zeta ** a * v
    return T, zeta


def selector_checks(rep, g):
    for q in (2, 4):
        T, zeta = lamp_selector(q)
        half = {(0, 0): 0.5, ((1 << q) - 1, 0): -0.5}  # (1 - w)/2 with w = z_0 ... z_{q-1}
        t1 = ga_sub(ga_mul(ga_star(T, q), T, q), half)
        t2 = ga_sub(ga_mul(T, ga_star(T, q), q), half)
        y = {(0, 1): 1.0}
        t3 = ga_sub(ga_mul(T, y, q), {k: zeta * v for k, v in ga_mul(y, T, q).items()})
        ok = not t1 and not t2 and not t3
        rep.check(f"lamp selector q={q}: T*T = TT* = (1-w)/2 and T y = zeta y T", ok, True, ok)
    ok = all(len(o) % (q & -q) == 0 for q in range(1, 13) for o in odd_orbits(q))
    rep.check("odd-parity lamp orbits: length divisible by 2^{v_2(q)} for q = 1..12", ok, True, ok)
    for q, m, L in [(4, 4, 1), (4, 8, 1), (2, 4, 3)]:
        U0, rho = ancilla_model(lamp_model(q, m, L)[0], g)
        dim = rho.shape[0]
        zeta = cmath.exp(2j * math.pi / q)
        ratios = []
        for scale in (1e-2, 1e-3, 1e-4):
            U = perturb(U0, scale, g)
            xi = sqrtm_psd(rho)
            e = model_error(q, m, U, xi)
            B = np.zeros((dim, dim), complex)
            for orbit in odd_orbits(q):
                for a, chi in enumerate(orbit):
                    P = np.eye(dim, dtype=complex)
                    for j, cj in enumerate(chi):
                        P = P @ (np.eye(dim) + cj * U[f"z{j}"]) / 2
                    B = B + zeta ** a * P
            u, _, vh = np.linalg.svd(B)
            Upol = u @ vh
            ratios.append(hs((Upol @ U["y"] @ Upol.conj().T - zeta * U["y"]) @ xi) / e)
        ok = max(ratios) < 100 * q and max(ratios) / min(ratios) < 5
        rep.check(f"lamp selector on perturbed models q={q} m={m}: rotation defect / model precision",
                  ", ".join(f"{r:.2f}" for r in ratios), "stable, O(1)", ok)


def ancilla_model(U0, g, k=3):
    """The model U0 tensored with the identity on an ancilla, with state I/d tensored with a non-flat ancilla state."""
    d = next(iter(U0.values())).shape[0]
    sigma = np.diag(0.6 ** np.arange(k))
    sigma /= np.trace(sigma)
    V = unitary(k, g)
    sigma = V @ sigma @ V.conj().T
    return {key: np.kron(v, np.eye(k)) for key, v in U0.items()}, np.kron(np.eye(d) / d, sigma)


def perturb(U0, scale, g):
    out = {}
    for key, v in U0.items():
        n = v.shape[0]
        H = g.normal(size=(n, n)) + 1j * g.normal(size=(n, n))
        out[key] = sl.expm(1j * scale * (H + H.conj().T) / 2) @ v
    return out


# ---------------------------------------------------------------- 3. spread and quantization
def invariant_laws(q, m, g):
    """Bernoulli digit laws on Z_m invariant under +m/q, and a mixture of two of them."""
    step = m // q
    cantor = np.zeros(m)
    cantor[::step] = 1.0 / q
    laws = [("Cantor digits", [(1.0, cantor)]), ("uniform digits", [(1.0, np.ones(m) / m)])]
    base = None
    for _ in range(2):
        base = g.random(step)
        base /= base.sum() * q
        laws.append(("random invariant digits", [(1.0, np.tile(base, q))]))
    laws.append(("mixture", [(0.3, cantor), (0.7, np.tile(base, q))]))
    return laws


def cdf(law, m, t, depth=60):
    """CDF on [0,1) of a mixture of Bernoulli digit laws, by digit recursion; t may be an array."""
    t = np.asarray(t, dtype=float)
    total = np.zeros_like(t)
    for w, p in law:
        cum = np.concatenate([[0.0], np.cumsum(p)])
        acc, mult, x = np.zeros_like(t), np.ones_like(t), t.copy()
        for _ in range(depth):
            x = x * m
            dgt = np.minimum(np.floor(x).astype(int), m - 1)
            acc += mult * cum[dgt]
            mult *= p[dgt]
            x -= dgt
            if mult.max() < 1e-18:
                break
        total += w * acc
    return total if total.ndim else float(total)


_GRID = {}


def arc_mass(law, m, a, length):
    a %= 1.0
    b = a + length
    if b <= 1:
        return cdf(law, m, b) - cdf(law, m, a) if b < 1 else 1 - cdf(law, m, a)
    return (1 - cdf(law, m, a)) + cdf(law, m, b - 1)


def sample(law, m, n, g, depth=40):
    weights = np.array([w for w, _ in law])
    comp = g.choice(len(law), size=n, p=weights / weights.sum())
    out = np.zeros(n)
    for c, (_, p) in enumerate(law):
        idx = np.where(comp == c)[0]
        digits = g.choice(m, size=(len(idx), depth), p=p)
        out[idx] = (digits * (float(m) ** -np.arange(1, depth + 1))).sum(axis=1)
    return out % 1.0


def circle_dist(x, y):
    d = np.abs((x - y) % 1.0)
    return np.minimum(d, 1 - d)


def spread_checks(rep, g):
    for q, m in [(2, 2), (2, 4), (2, 6), (4, 8), (4, 4), (8, 16)]:
        s = math.log(q) / math.log(m)
        worst = 0.0
        for _, law in invariant_laws(q, m, g):
            for _ in range(300):
                length = 10 ** g.uniform(-7, math.log10(0.5))
                worst = max(worst, arc_mass(law, m, g.random(), length) / (q * length) ** s)
        rep.check(f"spread q={q} m={m}: max nu(I)/(q|I|)^s over 1500 arcs", f"{worst:.4f}", 1, worst <= 1 + 1e-9)
    for q, m, L in [(2, 4, 3), (2, 6, 3), (4, 8, 2), (2, 2, 6)]:
        s = math.log(q) / math.log(m)
        law = invariant_laws(q, m, g)[0][1]
        pts = sample(law, m, 100000, g)
        worst_nb, worst_w = 0.0, float("inf")
        h = m ** L - 1
        atoms = np.array([float(Fraction((m // q) * sum(bj * m ** (L - 1 - j) for j, bj in enumerate(b)), h))
                          for b in itertools.product(range(q), repeat=L)])
        sets = [atoms]
        for d in (4, 16, 64):
            sets += [g.random(d), np.sort(sample(law, m, d, g)), (np.arange(d) + 0.5) / d]
        for F in sets:
            d = len(F)
            dist = np.full(len(pts), 1.0)
            for f in F:
                dist = np.minimum(dist, circle_dist(pts, f))
            r = (2 * d) ** (-1 / s) / (2 * q)
            worst_nb = max(worst_nb, float(np.mean(dist < r)) / (d * (2 * q * r) ** s))
            worst_w = min(worst_w, float(np.mean(dist)) / (d ** (-1 / s) / (4 * q * m)))
        rep.check(f"quantization q={q} m={m}: neighbourhood mass / d(2qr)^s (sampled)", f"{worst_nb:.3f}", 1,
                  worst_nb <= 1.05)
        rep.check(f"quantization q={q} m={m}: E dist(., F) / (d^(-1/s)/(4qm)) (sampled)", f"{worst_w:.2f}", ">= 1",
                  worst_w >= 1)


# ---------------------------------------------------------------- 4. shadowing identity
def shadowing_checks(rep, g):
    for q, m in [(2, 2), (2, 4), (4, 8), (8, 8)]:
        N = 60
        worst_orbit = worst_disp = worst_rot = 0.0
        for _ in range(20):
            U = g.random(N + 1)
            for t in range(N):
                U[t + 1] = (m * U[t] + 0.01 * g.normal()) % 1.0
            E = np.array([((m * U[t] - U[t + 1] + 0.5) % 1.0) - 0.5 for t in range(N)])

            def V(Useq, Eseq, t):
                return (Useq[t] - sum(m ** (-j - 1) * Eseq[t + j] for j in range(N - t))) % 1.0

            for t in range(N - 30):
                worst_orbit = max(worst_orbit, float(circle_dist(m * V(U, E, t), V(U, E, t + 1))))
                bound = sum(m ** (-j - 1) * abs(E[t + j]) for j in range(N - t))
                worst_disp = max(worst_disp, float(circle_dist(U[t], V(U, E, t))) - bound)
                U2 = U.copy()
                U2[t] = (U2[t] + 1.0 / q) % 1.0
                E2 = np.array([((m * U2[k] - U2[k + 1] + 0.5) % 1.0) - 0.5 for k in range(N)])
                same_future = np.allclose(E2[t:], E[t:], atol=1e-9)
                shift = float(circle_dist(V(U2, E2, t), (V(U, E, t) + 1.0 / q) % 1.0))
                worst_rot = max(worst_rot, shift if same_future else 1.0)
        ok = worst_orbit < 1e-12 and worst_disp < 1e-12 and worst_rot < 1e-12
        rep.check(f"shadowing q={q} m={m}: exact orbit, displacement bound, rotation equivariance",
                  f"{worst_orbit:.1e}, {worst_disp:.1e}, {worst_rot:.1e}", "0", ok)


# ---------------------------------------------------------------- 5. entropy from transport against a spread law
def gibbs(a, t):
    return math.log2(1 / math.tanh(a / 2)) + a * t / math.log(2)


def w1_to_law(atoms, weights, law, m, grid=200000):
    """Upper estimate of the W1 distance on R/Z between a discrete law and a digit law, via CDFs on a grid."""
    ts = (np.arange(grid) + 0.5) / grid
    key = (id(law), grid)
    if key not in _GRID:
        _GRID[key] = cdf(law, m, ts)
    Fnu = _GRID[key]
    order = np.argsort(atoms)
    Fmu = np.concatenate([[0.0], np.cumsum(weights[order])])[np.searchsorted(atoms[order], ts, side="right")]
    diff = Fmu - Fnu
    return float(np.mean(np.abs(diff - np.median(diff)))) + (len(atoms) + 2) / grid


def entropy_checks(rep, g):
    worst, n = -1e9, 0
    for q, m in [(2, 4), (4, 8), (2, 2)]:
        s = math.log(q) / math.log(m)
        law = invariant_laws(q, m, g)[0][1]
        for trial in range(12):
            d = int(g.choice([4, 6, 8, 12]))
            atoms = np.sort(sample(law, m, d, g)) if trial % 2 == 0 else g.random(d)
            V = unitary(d, g)
            Y = V @ np.diag(np.exp(2j * math.pi * atoms)) @ V.conj().T
            rho = random_state(d, g, ["random", "geometric", "lowrank", "pure"][trial % 4])
            xi = sqrtm_psd(rho)
            delta = w1_to_law(atoms, np.real(np.diag(V.conj().T @ rho @ V)), law, m)
            c0 = hs(comm(Y, xi))
            S = entropy_bits(rho)
            for L in (2, 4, 8, 16, 64):
                for a, b in ((0.5, 0.5), (1.0, 1.0), (0.2, 0.5)):
                    bound = s * math.log2(L) - s * math.log2(q) - gibbs(a, 1 + L * delta) - gibbs(b, 1 + L * c0 / 4)
                    worst = max(worst, bound - S)
                    n += 1
    rep.check(f"entropy from transport against spread laws, {n} cases: max (bound - S)", f"{worst:.3f}", "<= 0",
              worst <= 1e-9)


# ---------------------------------------------------------------- 6. invariances on perturbed binary lamp models
def spectral_law(Y, rho):
    T, Z = sl.schur(Y, output="complex")
    return (np.angle(np.diag(T)) / (2 * math.pi)) % 1.0, np.real(np.einsum("ij,jk,ki->i", Z.conj().T, rho, Z))


def w1_discrete(t1, w1, t2, w2):
    pts = np.sort(np.concatenate([t1, t2, [0.0, 1.0]]))
    seg = np.diff(pts)
    diff = np.array([np.sum(w1[t1 <= x]) - np.sum(w2[t2 <= x]) for x in pts[:-1]])
    order = np.argsort(diff)
    cum = np.cumsum(seg[order])
    c = diff[order][np.searchsorted(cum, cum[-1] / 2)]
    return float(np.sum(seg * np.abs(diff - c)))


def invariance_checks(rep, g):
    worst_D = worst_S = 0.0
    for m in (2, 4, 6):
        for L in (2, 3):
            U0, rho = ancilla_model(lamp_model(2, m, L)[0], g)
            for scale in (1e-3, 1e-2, 5e-2):
                U = perturb(U0, scale, g)
                xi = sqrtm_psd(rho)
                e = model_error(2, m, U, xi)
                t, w = spectral_law(U["y"], rho)
                worst_D = max(worst_D, 2 * math.pi * w1_discrete((m * t) % 1.0, w, t, w) / (math.pi * (m + 3) * e))
                worst_S = max(worst_S, 2 * math.pi * w1_discrete((t + 0.5) % 1.0, w, t, w) / e)
    rep.check("perturbed binary lamp models: delta_D / (pi (m+3) e), radians", f"{worst_D:.3f}", 1, worst_D <= 1)
    rep.check("perturbed binary lamp models: delta_S / e, radians", f"{worst_S:.3f}", "O(1)", worst_S < 50)


# ---------------------------------------------------------------- 7. controls
def control_checks(rep):
    ok = True
    for q in (1, 3, 5):
        U = {f"z{j}": -np.eye(1, dtype=complex) for j in range(q)}
        U.update({"w": -np.eye(1, dtype=complex), "x": np.eye(1, dtype=complex), "y": np.eye(1, dtype=complex)})
        ok &= model_error(q, 4, U, np.eye(1)) < 1e-12
    rep.check("odd q: exact one-dimensional model of the lamp presentation", ok, True, ok)
    out = []
    for L in (2, 3, 4):
        U, rot_ok = lamp_model(6, 6, L, d_lamps=2)
        e = model_error(6, 6, U, np.eye(2 ** L) / math.sqrt(2 ** L))
        out.append((2 ** L, e, rot_ok, 4 * math.pi / (2 * (6 ** L - 1))))
    ok = all(r and e <= b + 1e-12 for _, e, r, b in out)
    rep.check("q = 6 control: two Fourier lamps repeated three times, dimension 2^L",
              ", ".join(f"d={d}: {e:.2e}" for d, e, _, _ in out), "<= 2 pi/(6^L - 1)", ok)


def main():
    rep = Report("Forced branching: the lamp sources and their profile")
    g = rng(29)
    upper_checks(rep)
    selector_checks(rep, g)
    spread_checks(rep, g)
    shadowing_checks(rep, g)
    entropy_checks(rep, g)
    invariance_checks(rep, g)
    control_checks(rep)
    return rep.done()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
