"""Shared helpers for the numerical checks. Single-threaded; small dense matrices only."""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import itertools
import math

import numpy as np

PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def rng(seed):
    return np.random.default_rng(seed)


def hs(a):
    """Unnormalized Hilbert-Schmidt norm."""
    return float(np.linalg.norm(a))


def comm(a, b):
    return a @ b - b @ a


def acomm(a, b):
    return a @ b + b @ a


def herm(a):
    return (a + a.conj().T) / 2


def sqrtm_psd(a):
    w, v = np.linalg.eigh(herm(a))
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T


def entropy_bits(r):
    w = np.linalg.eigvalsh(herm(r))
    w = w[w > 1e-15]
    return float(-(w * np.log2(w)).sum())


def h2(p):
    p = min(max(p, 0.0), 1.0)
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def rel_entropy_bits(a, b):
    """D(a||b) = Tr a (log a - log b) for positive a, b with supp a in supp b (b may be singular)."""
    wa, va = np.linalg.eigh(herm(a))
    wb, vb = np.linalg.eigh(herm(b))
    la = np.where(wa > 1e-14, np.log2(np.clip(wa, 1e-300, None)), 0.0)
    lb = np.where(wb > 1e-14, np.log2(np.clip(wb, 1e-300, None)), 0.0)
    loga = (va * la) @ va.conj().T
    logb = (vb * lb) @ vb.conj().T
    return float(np.real(np.trace(a @ (loga - logb))))


def unitary(d, g):
    z = g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def sgn_inv(h):
    """sgn of a Hermitian operator, with sgn(0) = +1; an involution."""
    w, v = np.linalg.eigh(herm(h))
    return (v * np.where(w >= 0, 1.0, -1.0)) @ v.conj().T


def round_unitary(u):
    """The involution r(U): eigenvalues with nonnegative real part go to +1, the others to -1."""
    import scipy.linalg as sl
    # for a normal matrix the complex Schur form is diagonal and the Schur vectors are orthonormal eigenvectors
    tt, zz = sl.schur(u, output="complex")
    lam = np.diag(tt)
    s = np.where(lam.real >= 0, 1.0, -1.0)
    return (zz * s) @ zz.conj().T


def perturbed_involution(y, eta, g):
    d = y.shape[0]
    h = g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))
    return sgn_inv(y + eta * herm(h) / math.sqrt(d))


def perturbed_unitary(u, eta, g):
    import scipy.linalg as sl
    d = u.shape[0]
    h = g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))
    return sl.expm(1j * eta * herm(h) / math.sqrt(d)) @ u


def random_state(d, g, kind):
    """Density matrices of several shapes: 'random', 'geometric', 'lowrank', 'nearflat', 'pure', 'diag'."""
    u = unitary(d, g)
    if kind == "random":
        a = g.normal(size=(d, d)) + 1j * g.normal(size=(d, d))
        r = a @ a.conj().T
    elif kind == "geometric":
        q = g.uniform(0.2, 0.8)
        r = u @ np.diag(q ** np.arange(d)) @ u.conj().T
    elif kind == "lowrank":
        k = max(1, d // 3)
        a = g.normal(size=(d, k)) + 1j * g.normal(size=(d, k))
        r = a @ a.conj().T
    elif kind == "nearflat":
        r = u @ np.diag(1 + 0.2 * g.uniform(-1, 1, d)) @ u.conj().T
    elif kind == "pure":
        v = g.normal(size=d) + 1j * g.normal(size=d)
        r = np.outer(v, v.conj())
    elif kind == "diag":
        r = np.diag(g.dirichlet(np.ones(d) * 0.5)).astype(complex)
    else:
        raise ValueError(kind)
    r = herm(r)
    return r / np.trace(r).real


def kron_list(ms):
    out = np.array([[1.0 + 0j]])
    for m in ms:
        out = np.kron(out, m)
    return out


def paulis(m, extra=1):
    xs, zs = [], []
    for i in range(m):
        xs.append(np.kron(kron_list([PX if j == i else I2 for j in range(m)]), np.eye(extra)))
        zs.append(np.kron(kron_list([PZ if j == i else I2 for j in range(m)]), np.eye(extra)))
    return xs, zs


def purification(rho):
    """Canonical purification: the vector Theta^{-1}(xi), with Theta(v (x) w) = w v^T.
    Returned as a d x d array psi[a, b] (Alice index a, Bob index b), so that
    (Q (x) I) psi <-> xi Q^T and (I (x) Y) psi <-> Y xi."""
    xi = sqrtm_psd(rho)
    return xi.T.copy()  # psi[a, b] = xi[b, a]


class Report:
    """Collects named checks; prints a line per check and a final verdict."""

    def __init__(self, title):
        self.title = title
        self.rows = []
        print("=" * 78)
        print(title)
        print("=" * 78)

    def check(self, name, value, bound, ok, note=""):
        self.rows.append(ok)
        flag = "ok  " if ok else "FAIL"
        print(f"[{flag}] {name}: {value} (bound {bound}) {note}".rstrip())

    def info(self, text):
        print("      " + text)

    def done(self):
        n_fail = sum(1 for ok in self.rows if not ok)
        print("-" * 78)
        print(f"{self.title}: {len(self.rows)} checks, {n_fail} failures")
        return n_fail
