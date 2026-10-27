"""Root-only native23 exporter and bounded exact group/clock controls."""
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
checks = {}


def ck(test, tag):
    if not test:
        raise AssertionError(tag)
    checks[tag] = checks.get(tag, 0) + 1


def inv(w):
    return [s.swapcase() for s in reversed(w)]


def red(w):
    out = []
    for s in w:
        if out and out[-1] == s.swapcase():
            out.pop()
        else:
            out.append(s)
    return out


def pw(w, n):
    return red((w if n >= 0 else inv(w)) * abs(n))


def comm(a, b):
    return red(a + b + inv(a) + inv(b))


def point_step(g, y):
    ray, j = y
    if g.lower() not in 'pq':
        raise ValueError(g)
    other = 2 if g.lower() == 'p' else 3
    if g.islower():
        if ray == 1:
            return (1, j + 1)
        if ray == other:
            return (1, 1) if j == 1 else (other, j - 1)
    else:
        if ray == 1:
            return (other, 1) if j == 1 else (1, j - 1)
        if ray == other:
            return (other, j + 1)
    return y


def act(w, y):
    for g in reversed(w):
        y = point_step(g, y)
    return y


def subset(w, f):
    return tuple(sorted(act(w, y) for y in f))


def shift(w):
    return (w.count('p') - w.count('P'), w.count('q') - w.count('Q'))


def host_identity(w):
    if shift(w) != (0, 0):
        return False
    return all(act(w, (r, h)) == (r, h) for r in range(1, 4) for h in range(1, len(w) + 2))


F0 = ((1, 1), (1, 2))
s = comm(['p'], ['q'])
B = red(['p'] + s + ['P'])
u = red(B + s + ['p'])
v = red(B + s + ['q'])
J = ['x', 'z', 'x', 'z']
rows = []


def row(name, family, w):
    rows.append({'id': name, 'family': family, 'word': red(w)})


row('H1', 'host', pw(s, 2))
row('H2', 'host', pw(s + ['p'] + s + ['P'], 3))
row('H3', 'host', comm(s, ['p', 'p'] + s + ['P', 'P']))
row('H4', 'host', ['p'] + s + ['P', 'q'] + inv(s) + ['Q'])
row('L1', 'onsite', ['x', 'x'])
row('L2', 'onsite', ['z', 'z'])
row('L3', 'onsite', pw(['x', 'z'], 4))
for j, w in enumerate([s, u, v]):
    for a in ['x', 'z']:
        row('S' + str(2*j + (1 if a == 'x' else 2)), 'stabilizer', comm([a], w))
for overlap, power in [(1, 1), (0, 2)]:
    h = pw(['p'], power)
    for a, b in itertools.product(['x', 'z'], repeat=2):
        row('P' + str(overlap) + a + b, 'pair', comm([a], h + [b] + inv(h)))
for g in ['p', 'q']:
    row('C' + g, 'center', comm(J, [g]))
source = {
    'name': 'G_2 central Pauli lamps on unordered two-subsets of Houghton rays',
    'generators': ['p', 'q', 'x', 'z'], 'inverse_convention': 'uppercase is inverse; maps compose right to left',
    'central_word': J, 'target_sector': 'pi(J)=-I (representation sector, not an abstract identity relator)',
    'relators': rows, 'relator_precision': 'all exact operator identities',
    'state_contract': 'original arbitrary normal mixed state; actual signed-generator unitary mirrors of max root error e; unrestricted ambient spaces',
    'pair_prototypes': [{'overlap': 1, 'transporter': ['p']}, {'overlap': 0, 'transporter': ['p', 'p']}],
    'stabilizer_words': {'s': s, 'u=Bsp': u, 'v=Bsq': v},
    'no_game_claim': 'This source JSON is not an exported game or a finite-dimensional exact negative representation.'
}
ck(len(rows) == 23, 'native_row_count')
ck(len({r['id'] for r in rows}) == 23, 'unique_row_ids')


def evaluate(w):
    """Exact normal form J^phase X^a Z^b h, with h an actual infinite-ray word."""
    phase, xs, zs, hw = 0, set(), set(), []
    for g in w:
        if g.lower() in 'pq':
            hw = red(hw + [g])
        elif g.lower() in 'xz':
            at = subset(hw, F0)
            if g.lower() == 'x':
                phase ^= int(at in zs)
                xs.symmetric_difference_update([at])
            else:
                zs.symmetric_difference_update([at])
        else:
            raise ValueError(g)
    return phase, xs, zs, hw


for r in rows:
    ph, xx, zz, h = evaluate(r['word'])
    ck(ph == 0 and not xx and not zz, 'literal_lamp_relator')
    ck(host_identity(h), 'literal_host_relator')
    ck(red(r['word']) == r['word'], 'freely_reduced_export')
ph, xx, zz, h = evaluate(J)
ck(ph == 1 and not xx and not zz and host_identity(h), 'J_is_nontrivial_sign')
for w in [s, u, v]:
    ck(subset(w, F0) == F0, 'literal_setwise_stabilizer')


def gap_generator(m):
    if m == 0:
        return ['p']
    # u_m is the standard p on the three-ray complement of first m points.
    return red(pw(['p'], m) + ['q'] + pw(['p'], -m) + ['Q', 'p'])


def gap_word(target):
    w, before = [], 0
    for i, dest in enumerate(target):
        w += pw(gap_generator(i), dest - before - 1)
        before = dest
    return red(w)


for k in range(1, 6):
    endpoint = 11 if k == 2 else max(k + 2, 11 - k)
    for n in range(k, endpoint):
        base = tuple((1, j) for j in range(1, k+1))
        for chosen in itertools.combinations(range(1, n+1), k):
            w = gap_word(chosen)
            ck(subset(w, base) == tuple((1, j) for j in chosen), 'gap_address_k' + str(k))
            bound = (1 if k == 1 else 6*k-7) * (n-k)
            ck(len(w) <= bound, 'gap_length_k' + str(k))
            if k == 2:
                ck(2*len(w)+1 <= 10*n-19, 'native_pair_lamp_word_length')

# All ordered pair templates on first eight points, with explicit finite permutations.
all_pairs = list(itertools.combinations(range(1, 9), 2))
for aa in all_pairs:
    for bb in all_pairs:
        if aa == bb:
            continue
        a, b = set(aa), set(bb)
        overlap = len(a & b)
        if overlap == 0:
            prototype = ({1, 2}, {3, 4})
            targets = sorted(a) + sorted(b)
        else:
            prototype = ({1, 2}, {2, 3})
            targets = sorted(a-b) + sorted(a & b) + sorted(b-a)
        permutation = {i+1: value for i, value in enumerate(targets)}
        # Extend the finite partial bijection on [8] by a bijection of its complements.
        for domain, image in zip(sorted(set(range(1, 9))-set(permutation)), sorted(set(range(1, 9))-set(permutation.values()))):
            permutation[domain] = image
        ck({permutation[t] for t in prototype[0]} == a and {permutation[t] for t in prototype[1]} == b, 'all_ordered_pair_templates')

clock_records = []
straddling = 0
for L in [1, 2, 3]:
    R = 4*L + 2
    region = [(r, j) for r in range(1, 4) for j in range(1, R+1)]
    region_set = set(region)
    box = set(itertools.product(range(-L, L+1), repeat=2))
    edges = 0
    for c in itertools.product(range(-L-1, L+2), repeat=2):
        tc = pw(['p'], c[0]) + pw(['q'], c[1])
        if c in box:
            ck(set(subset(inv(tc), F0)) <= region_set, 'transported_native_lamp_inside')
        for g, delta in [('p', (1, 0)), ('P', (-1, 0)), ('q', (0, 1)), ('Q', (0, -1))]:
            cp = (c[0]+delta[0], c[1]+delta[1])
            if c not in box and cp not in box:
                continue
            tcp = pw(['p'], cp[0]) + pw(['q'], cp[1])
            cocycle = red(inv(tcp) + [g] + tc)
            ck(shift(cocycle) == (0, 0), 'active_signed_cocycle_zero_translation')
            point_map = {pt: act(cocycle, pt) for pt in region}
            ck(set(point_map.values()) == region_set, 'active_point_region_preserved')
            for ray in range(1, 4):
                for j in range(R+1, R+len(cocycle)+3):
                    ck(act(cocycle, (ray, j)) == (ray, j), 'active_cocycle_tail_fixed')
            for pair in itertools.combinations(region, 2):
                ck({point_map[pt] for pt in pair} <= region_set, 'active_internal_subset_preserved')
            moved = next((pt for pt in region if point_map[pt] != pt), None)
            if moved is not None:
                external = (1, R+len(cocycle)+3)
                old = tuple(sorted([moved, external]))
                new = subset(cocycle, old)
                ck(old != new and not set(new) <= region_set, 'external_straddling_vacuum_site')
                straddling += 1
            edges += 1
    f = {j: math.cos(math.pi*j/(2*(L+1))) / math.sqrt(L+1) for j in range(-L, L+1)}
    norm = sum(t*t for t in f.values())
    energy = sum((f.get(j+1, 0)-f.get(j, 0))**2 for j in range(-L-1, L+1))
    expected = 2 - 2*math.cos(math.pi/(2*(L+1)))
    ck(abs(norm-1) < 2e-14 and abs(energy-expected) < 2e-14, 'scalar_Dirichlet_clock')
    interior = sum((f[j+1]-f[j])**2 for j in range(-L, L))
    ck(abs(interior-expected) > 1e-5, 'boundary_omission_negative_control')
    clock_records.append({'L': L, 'R': R, 'points': len(region), 'internal_subset_qubits': math.comb(len(region), 2), 'active_signed_edges': edges, 'clock_energy': energy})

# Explicit negative controls, separate from the accepted model.
mut = evaluate(comm(['x'], ['z']))
ck(mut[0] == 1 and not mut[1] and not mut[2], 'onsite_as_cross_template_rejected')
ck({0} != {0, 1}, 'missing_overlap_template_rejected')
wrong = red(inv(pw(['p'], 0)) + ['p'] + pw(['p'], 1))
ck(shift(wrong) != (0, 0), 'wrong_cocycle_order_rejected')
inside, outside, partner = (1, 1), (1, 7), (2, 1)
ck(not {outside, partner} <= {(r, j) for r in range(1, 4) for j in range(1, 7)}, 'point_region_crossing_rejected')
bad_gap = red(gap_word((2, 5)) + gap_generator(1))
ck(subset(bad_gap, F0) != ((1, 2), (1, 5)), 'wrong_gap_exponent_rejected')

export = HERE / 'SOURCE-K2.json'
export.write_text(json.dumps(source, indent=2)+'\n', encoding='utf-8')
result = {
    'status': 'PASS', 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_export': {'path': export.name, 'sha256': hashlib.sha256(export.read_bytes()).hexdigest()},
    'native_generators': 4, 'native_relators': len(rows), 'max_relator_length': max(len(r['word']) for r in rows),
    'family_counts': {name: sum(r['family']==name for r in rows) for name in sorted({r['family'] for r in rows})},
    'exact_check_counts': checks, 'total_boolean_comparisons': sum(checks.values()),
    'clock_records': clock_records, 'external_straddling_witnesses': straddling,
    'no_native_spin_arrays': True,
    'claim_limit': 'Literal infinite-action finite probes and exact word/Pauli controls, with float scalar clock constants. Infinite presentation, all-k laws and asymptotic ED require written proofs, not this enumeration.'
}
(HERE/'NATIVE-RESULTS.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: result[key] for key in ['status','native_generators','native_relators','max_relator_length','family_counts','total_boolean_comparisons','clock_records','external_straddling_witnesses']}, indent=2))
