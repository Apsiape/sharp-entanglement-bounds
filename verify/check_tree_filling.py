"""Single-process checks of the tree-of-faces argument behind the linear filling charge.

The paper proves it in the section on the point-Pauli game (bounded transport, substitution of whole
words into one face, the tree of faces, regular diagrams, and the theorem on the linear filling charge).
Only the Python standard library is required; the results are printed. Integer edge words are checked
BEFORE group labels are substituted, so that no group relation can hide a topological mistake. The
regularization here also subdivides the bigons of degree-one vertices, so its face count is checked
against the conservative bound A + 12E. These finite checks support the proof and do not replace it.
"""
import json
import math
import random
from collections import defaultdict, deque


def inverse(w):
    return [-x for x in reversed(w)]


def reduce_word(w):
    out = []
    for x in w:
        if not x:
            continue
        if out and out[-1] == -x:
            out.pop()
        else:
            out.append(x)
    return out


class DSU:
    def __init__(self, vs):
        self.p = {v: v for v in vs}

    def find(self, v):
        while self.p[v] != v:
            self.p[v] = self.p[self.p[v]]
            v = self.p[v]
        return v

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.p[a] = b
        return True


def plane_graph(points, pairs):
    """Rotation system from a noncrossing straight-line plane drawing."""
    tails, heads, rotation = {}, {}, defaultdict(list)
    for i, (u, v) in enumerate(pairs, 1):
        tails[i], heads[i] = u, v
        tails[-i], heads[-i] = v, u
        rotation[u].append(i)
        rotation[v].append(-i)
    for v, hs in rotation.items():
        hs.sort(key=lambda h: math.atan2(points[heads[h]][1] - points[v][1],
                                        points[heads[h]][0] - points[v][0]))
    # Face on the left: take the clockwise neighbour of the arriving inverse.
    nxt = {}
    for h in tails:
        hs = rotation[heads[h]]
        nxt[h] = hs[(hs.index(-h) - 1) % len(hs)]
    cycles, seen = [], set()
    for h in tails:
        if h in seen:
            continue
        cyc, t = [], h
        while t not in seen:
            seen.add(t)
            cyc.append(t)
            t = nxt[t]
        assert t == h
        cycles.append(cyc)
    def area(c):
        return sum(points[tails[h]][0] * points[heads[h]][1]
                   - points[tails[h]][1] * points[heads[h]][0] for h in c) / 2
    outer = min(range(len(cycles)), key=lambda i: area(cycles[i]))
    assert len(points) - len(pairs) + len(cycles) == 2
    return tails, heads, rotation, nxt, cycles, outer


def regularize(points, pairs):
    """Bands + identity vertex polygons + caps of bounded face holes.

    Every cap has its old edge word interleaved with identities. Band and
    vertex faces are freely trivial. Repeated face edges and exterior stems
    are permitted in the input rotation system.
    """
    tails, heads, rot, nxt, cycles, outer = plane_graph(points, pairs)
    edges, labels, faces = {}, {}, []
    edge_lookup = {}
    edge_counter = 0
    def add_edge(u, v, label=0, fresh=False):
        nonlocal edge_counter
        assert u != v
        key = frozenset((u, v))
        if key in edge_lookup and not fresh:
            j = edge_lookup[key]
            return j if edges[j] == (u, v) else -j
        edge_counter += 1
        j = edge_counter
        edges[j] = (u, v)
        labels[j] = label
        edge_lookup[key] = j
        return j
    def a(h):
        return ('a', h)
    def b(h):
        return ('b', h)
    long, short, corner = {}, {}, {}
    for h in tails:
        long[h] = add_edge(b(h), a(-h), h)
        short[h] = add_edge(a(h), b(h))
    for h in tails:
        corner[-h] = add_edge(a(-h), b(nxt[h]), fresh=True)
    hole_words = []
    for c in cycles:
        hole_words.append([x for h in c for x in (long[h], corner[-h])])
    for i, c in enumerate(hole_words):
        if i != outer:
            faces.append(c)
    original_count = len(faces)
    for j in range(1, len(pairs) + 1):
        faces.append([-long[j], -short[j], -long[-j], -short[-j]])
    prev_tail = {nxt[-h]: h for h in tails}
    for v, hs in rot.items():
        # The vertex polygon has sides short[h], reverse(corner[h]).
        seq, h = [], hs[0]
        while True:
            previous = prev_tail[h]
            seq += [short[h], -corner[previous]]
            h = previous
            if h == hs[0]:
                break
        # A degree-one vertex is an identity digon. Subdivide one of its
        # sides with a new vertex, consistently in all incident polygons.
        if len(seq) == 2:
            old = seq[1]
            j = abs(old)
            u, w = edges[j] if old > 0 else edges[j][::-1]
            mid = ('leaf-mid', v)
            e1, e2 = add_edge(u, mid), add_edge(mid, w)
            def substitute(poly):
                out = []
                for t in poly:
                    out.extend([e1, e2] if t == old else
                               [-e2, -e1] if t == -old else [t])
                return out
            faces = [substitute(poly) for poly in faces]
            hole_words = [substitute(poly) for poly in hole_words]
            seq = substitute(seq)
            del edges[j]
            del labels[j]
            edge_lookup[frozenset((u, w))] = abs(seq[0])
        # Fan triangulation; every vertex-polygon edge has identity label.
        start = lambda t: edges[abs(t)][0 if t > 0 else 1]
        end = lambda t: edges[abs(t)][1 if t > 0 else 0]
        nodes = [start(t) for t in seq]
        assert all(end(seq[i]) == nodes[(i + 1) % len(seq)]
                   for i in range(len(seq)))
        for i in range(1, len(seq) - 1):
            faces.append([add_edge(nodes[0], nodes[i]), seq[i],
                          add_edge(nodes[i + 1], nodes[0])])
    boundary = inverse(hole_words[outer])
    # Drop unused obsolete identity edges from degree-one subdivision.
    used = {abs(t) for f in faces for t in f} | {abs(t) for t in boundary}
    edges = {j: uv for j, uv in edges.items() if j in used}
    labels = {j: x for j, x in labels.items() if j in used}
    return edges, labels, faces, boundary, original_count


def check_disk(edges, faces, boundary, rng):
    incidence = defaultdict(list)
    for i, f in enumerate(faces):
        for t in f:
            incidence[abs(t)].append((i, t))
    bd = {abs(t) for t in boundary}
    for j, inc in incidence.items():
        assert len(inc) == (1 if j in bd else 2)
        if len(inc) == 2:
            assert inc[0][1] == -inc[1][1]
    verts = set(v for uv in edges.values() for v in uv)
    assert len(verts) - len(edges) + len(faces) == 1
    dsu, tree = DSU(verts), set()
    # The boundary is simple after regularization.
    root_edge = boundary[0]
    for t in boundary[1:]:
        assert dsu.join(*edges[abs(t)])
        tree.add(abs(t))
    js = list(edges)
    rng.shuffle(js)
    for j in js:
        if dsu.join(*edges[j]):
            tree.add(j)
    assert len(tree) == len(verts) - 1
    dual = defaultdict(list)
    for j, inc in incidence.items():
        if j not in tree and len(inc) == 2:
            (i, _), (k, _) = inc
            dual[i].append((k, j))
            dual[k].append((i, j))
    assert sum(map(len, dual.values())) == 2 * (len(faces) - 1)
    root = incidence[abs(root_edge)][0][0]
    seen = set()
    def expand(i, initial, parent):
        assert i not in seen
        seen.add(i)
        poly = faces[i]
        idx = poly.index(initial)
        poly = poly[idx:] + poly[:idx]
        children = {j: k for k, j in dual[i] if k != parent}
        out = []
        for t in poly:
            if abs(t) in children:
                child = expand(children[abs(t)], -t, i)
                assert child[0] == -t
                out.extend(child[1:])
            else:
                out.append(t)
        return out
    expanded = expand(root, root_edge, None)
    assert len(seen) == len(faces)
    assert reduce_word(expanded) == reduce_word(boundary)
    return len(expanded), len(faces), max(map(len, faces))


def grid(n, m, stems=False):
    points = {(i, j): (i, j) for i in range(n + 1) for j in range(m + 1)}
    pairs = []
    for i, j in points:
        if i < n:
            pairs.append(((i, j), (i + 1, j)))
        if j < m:
            pairs.append(((i, j), (i, j + 1)))
    if stems:
        # Exterior tether, and a dangling tree INSIDE a bounded face.
        for k in range(1, 4):
            points[('out', k)] = (-k, -0.2 * k)
            pairs.append(((0, 0) if k == 1 else ('out', k - 1), ('out', k)))
        points['inside'] = (0.31, 0.24)
        points['inside2'] = (0.61, 0.34)
        pairs += [((0, 0), 'inside'), ('inside', 'inside2')]
    return points, pairs


def nested():
    # An annular face slit by a bridge; its boundary walk repeats that bridge.
    p = {0: (0, 0), 1: (8, 0), 2: (8, 8), 3: (0, 8),
         4: (2, 2), 5: (6, 2), 6: (6, 6), 7: (2, 6)}
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4),
         (4, 5), (5, 6), (6, 7), (7, 4)]
    return p, e


def algebra_check(rng):
    # Complex 2x2 matrices, independent of the topological implementation.
    def mm(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(2))
                 for j in range(2)] for i in range(2)]
    def adj(a):
        return [[a[j][i].conjugate() for j in range(2)] for i in range(2)]
    def sub(a, b):
        return [[a[i][j] - b[i][j] for j in range(2)] for i in range(2)]
    def norm(a):
        return math.sqrt(sum(abs(z) ** 2 for row in a for z in row))
    I = [[1+0j, 0j], [0j, 1+0j]]
    def unitary():
        t, p, q = [rng.uniform(-math.pi, math.pi) for _ in range(3)]
        a = math.cos(t) * complex(math.cos(p), math.sin(p))
        b = math.sin(t) * complex(math.cos(q), math.sin(q))
        return [[a, b], [-b.conjugate(), a.conjugate()]]
    max_identity_gap, max_violation = 0.0, 0.0
    for _ in range(1000):
        d = rng.randrange(2, 9)
        old = [unitary() for _ in range(d)]
        new = [unitary() if rng.random() < .5 else x for x in old]
        p = (0.0, 1.0, 0.5, rng.random())[_ % 4]
        xi = [[math.sqrt(p)+0j, 0j], [0j, math.sqrt(1-p)+0j]]
        def product(xs):
            v = I
            for x in xs:
                v = mm(v, x)
            return v
        defect = lambda u: norm(mm(sub(u, I), xi))
        comm = lambda u: norm(sub(mm(u, xi), mm(xi, u)))
        # q_i - t_i = t_i (t_i^* q_i - I).
        lhs = defect(product(new))
        rhs = defect(product(old))
        for i, (t, q) in enumerate(zip(old, new)):
            if t is q:
                continue
            v = mm(adj(t), q)
            rhs += defect(v) + 2 * sum(comm(x) for x in old[i+1:])
            gap = norm(sub(sub(q, t), mm(t, sub(v, I))))
            max_identity_gap = max(max_identity_gap, gap)
        max_violation = max(max_violation, lhs - rhs)
        assert lhs <= rhs + 2e-12
    return {'trials': 1000, 'max_identity_gap': max_identity_gap,
            'max_inequality_violation': max_violation}


def main():
    rng = random.Random(271026)
    records = []
    for n in range(1, 9):
        for m in range(1, 7):
            for stems in (False, True):
                p, e = grid(n, m, stems)
                regular = regularize(p, e)
                edges, labels, faces, boundary, count = regular
                for repetition in range(4):
                    stats = check_disk(edges, faces, boundary, rng)
                    records.append(stats)
                for f in faces[count:]:
                    assert reduce_word([labels[abs(t)] * (1 if t > 0 else -1)
                                        for t in f]) == []
                assert len(faces) <= count + 12 * len(e)
    p, e = nested()
    edges, labels, faces, boundary, count = regularize(p, e)
    for _ in range(16):
        records.append(check_disk(edges, faces, boundary, rng))
    pval = math.log2(107)
    # Literal, unrounded source words; capitals denote inverse letters.
    commutator = lambda u, v: u + v + inverse(u) + inverse(v)
    a, b, s, x, z, J = ([i] for i in range(1, 7))
    conjugate = lambda u, v: u + v + inverse(u)
    relators = [s+s, (s+a+s+inverse(a))*3,
                commutator(s, conjugate(a+a, s)),
                commutator(a, b)+inverse(s),
                a+s+inverse(a)+b+inverse(s)+inverse(b),
                J+J, x+x, z+z]
    relators += [commutator(J, u) for u in (a, b, s, x, z)]
    relators += [commutator(x, z)+inverse(J)]
    relators += [commutator(u, v) for u in (x, z) for v in (s+a, s+b)]
    relators += [commutator(u, conjugate(a, v)) for u in (x, z) for v in (x, z)]
    assert len(relators) == 22 and max(map(len, relators)) == 12
    K = 12
    k = max(2*K, 4)
    constant = 1+2*K+2*k+5*k*(K+1)
    assert constant == 1633
    out = {'status': 'CHECKED: finite combinatorial and floating-point checks',
           'disk_trials': len(records), 'max_faces': max(r[1] for r in records),
           'max_expanded_perimeter': max(r[0] for r in records),
           'cases': ['rectangular grids', 'exterior stems', 'interior dangling trees',
                     'nested cycles joined by a bridge'],
           'algebra': algebra_check(rng),
           'literal_source': {'relator_count': len(relators),
                              'relator_lengths': list(map(len, relators)),
                              'K': K, 'conservative_C_K': constant},
           'arithmetic': {'p': pval, 'old_beta': 1/(4*pval+1),
                          'new_beta': 1/(2*pval+1),
                          'source_N_power': 2/(2*pval+1),
                          'Ne_decay_power': 1-2/(2*pval+1),
                          'balance_power_of_e': 2-(2*pval+1)*2/(2*pval+1)}}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
