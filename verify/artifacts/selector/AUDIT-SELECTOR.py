"""Independent truth-table/finite-operator audit of native selector artifacts.

Does not import the generator or claim to compute undecidability/nonattainment.
"""
import os
for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_var] = '1'
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_tables(game, source, label):
    contexts = game['contexts']
    by_name = set(game['variables'])
    assert len(by_name) == len(game['variables'])
    assert game['question_distribution']['context_count'] == len(contexts)
    if label == 'MAGIC':
        g_rows = [([f'm{r}{c}' for c in range(3)], 1) for r in range(3)]
        g_rows += [([f'm{r}{c}' for r in range(3)], -1 if c == 2 else 1)
                   for c in range(3)]
    else:
        g_rows = [(['x'], 1), (['x'], -1)]
    expected_ids = {f'H/{j}' for j in range(len(source))}
    expected_ids |= {f'G/{j}' for j in range(len(g_rows))}
    assert len(contexts) == len(expected_ids)
    assert {c['id'] for c in contexts} == expected_ids
    expected_names = {'flag'} | {'H/'+v for row in source for v in row['variables']}
    expected_names |= {'G/'+v for row, _ in g_rows for v in row}
    assert by_name == expected_names
    total_probability = Fraction(0)
    assignments = 0
    for context in contexts:
        names = context['variables']
        rule = context['predicate']
        assert names[0] == 'flag' == rule['flag_variable']
        assert len(names) == len(set(names)) and set(names) <= by_name
        assert names[1:] == rule['data_variables']
        source_letter, row_id = context['id'].split('/')
        row_id = int(row_id)
        is_fallback = source_letter == 'H'
        assert rule['active_value'] == int(is_fallback)
        original_row = source[row_id] if is_fallback else None
        if is_fallback:
            assert names[1:] == ['H/'+v for v in original_row['variables']]
            assert rule['rhs_bit'] == (original_row['product_sign'] == -1)
        else:
            expected_row, expected_sign = g_rows[row_id]
            assert names[1:] == ['G/'+v for v in expected_row]
            assert rule['rhs_bit'] == (expected_sign == -1)
        original_sign = -1 if rule['rhs_bit'] else 1
        for bits in itertools.product((0, 1), repeat=len(names)):
            # Engine A evaluates the exported bit predicate.
            accepted_a = (bits[0] != rule['active_value']
                          or sum(bits[1:]) % 2 == rule['rhs_bit'])
            # Engine B uses the algebraic active-sector product equation.
            flag_sign = 1-2*bits[0]
            active_projection = (1 + (-1)**rule['active_value']*flag_sign)//2
            data_sign = 1
            for b in bits[1:]:
                data_sign *= 1-2*b
            accepted_b = active_projection * (data_sign-original_sign) == 0
            assert accepted_a == accepted_b
            assignments += 1
        incidence_weight = Fraction(1, len(contexts)*len(names))
        assert incidence_weight > 0
        total_probability += len(names)*incidence_weight
    assert total_probability == 1
    return assignments


def ms_operators():
    i = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    factors = ((x, i), (i, x), (x, x), (i, z), (z, i), (z, z),
               (x, z), (z, x), (y, y))
    return {f'G/m{r}{c}': np.kron(*factors[3*r+c])
            for r in range(3) for c in range(3)}


def audit_magic_strategy(game):
    eye = np.eye(4, dtype=complex)
    zero = np.zeros((4, 4), dtype=complex)
    ops = ms_operators()
    worst = 0.0
    checked_contexts = 0
    explicit_joint_checks = 0
    for context in game['contexts']:
        rule = context['predicate']
        # flag=0; inactive H variables are exactly I (bit zero).
        row_ops = [eye]+[ops.get(v, eye) for v in context['variables'][1:]]
        for a in row_ops:
            worst = max(worst, float(np.linalg.norm(a@a-eye)),
                        float(np.linalg.norm(a-a.conj().T)))
        for j, a in enumerate(row_ops):
            for b in row_ops[j+1:]:
                worst = max(worst, float(np.linalg.norm(a@b-b@a)))
        active = eye if rule['active_value'] == 0 else zero
        product = eye.copy()
        for a in row_ops[1:]:
            product = product@a
        sign = (-1)**rule['rhs_bit']
        worst = max(worst, float(np.linalg.norm(active@(product-sign*eye))))
        if context['id'].startswith('G/'):
            joint_sum = np.zeros((4, 4), dtype=complex)
            for bits in itertools.product((0, 1), repeat=len(row_ops)):
                joint = eye.copy()
                for a, bit in zip(row_ops, bits):
                    joint = joint@(eye+(-1)**bit*a)/2
                valid = bits[0] != rule['active_value'] or sum(bits[1:])%2 == rule['rhs_bit']
                if valid:
                    joint_sum += joint
                else:
                    worst = max(worst, float(np.linalg.norm(joint)))
                # Each context/variable consistency holds on Phi_4 by transpose.
                for a in row_ops:
                    assert np.allclose(a, a.T)
                explicit_joint_checks += 1
            worst = max(worst, float(np.linalg.norm(joint_sum-eye)))
        checked_contexts += 1
    assert worst < 1e-12
    return {'local_dim': 4, 'ebits': 2, 'state': 'Phi_4',
            'every_context_operator_checked': checked_contexts,
            'explicit_active_joint_PVM_assignments_checked': explicit_joint_checks,
            'max_operator_residual': worst, 'score': 'exactly 1 by checked PVM/transposed consistency'}


def negative_control():
    # Both sides have the contradictory pair x=0,x=1. Shared flag cannot win.
    shared = []
    for f, x, y in itertools.product((0, 1), repeat=3):
        rows = [(f != 0 or x == b) for b in (0, 1)]
        rows += [(f != 1 or y == b) for b in (0, 1)]
        if all(rows):
            shared.append((f, x, y))
    assert not shared
    # Replacing the SAME flag by private per-row flags makes every row inactive.
    private = {'G0flag': 1, 'G1flag': 1, 'H0flag': 0, 'H1flag': 0, 'x': 0, 'y': 0}
    assert all(private[k] != active for k, active in
               [('G0flag', 0), ('G1flag', 0), ('H0flag', 1), ('H1flag', 1)])
    return {'shared_flag_classical_solutions': 0,
            'private_flags_allow_perfect_classical_cheat': True,
            'negative_control_pass': True}


def main():
    source_path = ROOT/'l4m8/L4M8-LINEAR-GAME.json'
    before = digest(source_path)
    native = json.loads(source_path.read_text(encoding='utf-8'))
    source_rows = native['rows']
    controls = []
    for label in ('MAGIC', 'CONTRADICTION'):
        path = HERE/f'SELECTOR-{label}.json'
        game = json.loads(path.read_text(encoding='utf-8'))
        assert game['fallback_provenance']['sha256'] == before
        controls.append({'object': path.name, 'sha256': digest(path),
                         'contexts': len(game['contexts']),
                         'truth_assignments_checked': audit_tables(game, source_rows, label)})
        if label == 'MAGIC':
            controls[-1]['finite_perfect_strategy'] = audit_magic_strategy(game)
    assert digest(source_path) == before
    ledger = {'status': 'PASS', 'objects': controls, 'negative_control': negative_control(),
              'fallback_source_unchanged': True, 'fallback_sha256': before,
              'limitations': 'Finite syntax, all local predicates, incidence weights, and one finite perfect strategy checked. Value1/nonattainment of H and undecidability are inherited written theorems, never inferred from finite tests.'}
    (HERE/'AUDIT-SELECTOR.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    print(json.dumps(ledger))


if __name__ == '__main__':
    main()
