"""Build two native conditional-parity BCS selector games; no search jobs."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FALLBACK = ROOT/'l4m8/L4M8-LINEAR-GAME.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def magic_square():
    names = [[f'm{r}{c}' for c in range(3)] for r in range(3)]
    return {'name': 'Magic Square', 'variables': sum(names, []),
            'rows': ([{'variables': row, 'product_sign': 1} for row in names]
                     + [{'variables': [names[r][c] for r in range(3)],
                         'product_sign': -1 if c == 2 else 1} for c in range(3)])}


def contradiction():
    return {'name': 'x=0 and x=1', 'variables': ['x'],
            'rows': [{'variables': ['x'], 'product_sign': 1},
                     {'variables': ['x'], 'product_sign': -1}]}


def build(left, right):
    variables = ['flag']
    contexts = []
    for active, prefix, source in ((0, 'G/', left), (1, 'H/', right)):
        used = {v for row in source['rows'] for v in row['variables']}
        variables += [prefix+v for v in source['variables'] if v in used]
        for row_id, row in enumerate(source['rows']):
            data = [prefix+v for v in row['variables']]
            assert len(set(data)) == len(data)
            contexts.append({'id': f'{prefix}{row_id}',
                             'variables': ['flag']+data,
                             'predicate': {'kind': 'conditional-parity',
                                           'flag_variable': 'flag',
                                           'active_value': active,
                                           'data_variables': data,
                                           'rhs_bit': int(row['product_sign'] == -1)}})
    return {'format': 'native-binary-constraint-variable-game/v1',
            'sources': {'G': left['name'], 'H': right['name']},
            'variables': variables, 'contexts': contexts,
            'question_distribution': {'kind': 'uniform-context-then-uniform-incidence',
                                      'context_count': len(contexts)},
            'answer_rule': 'Alice returns a satisfying full context assignment; Bob returns the queried variable bit; accept iff the bits agree.',
            'scope': 'General Boolean constraint-system game; conditional parity is not a linear parity equation.',
            'fallback_provenance': {'path': FALLBACK.relative_to(ROOT).as_posix(), 'sha256': digest(FALLBACK),
                                    'quantum_value_and_nonattainment': 'Inherited from the source/compiler theorem cited in the paper; not established by this generator.'}}


def main():
    original_hash = digest(FALLBACK)
    native = json.loads(FALLBACK.read_text(encoding='utf-8'))
    fallback = {'name': 'L_(4,8) one-third-rate fixed linear game',
                'variables': list(native['variables']), 'rows': native['rows']}
    counts = []
    for name, left in (('MAGIC', magic_square()), ('CONTRADICTION', contradiction())):
        game = build(left, fallback)
        path = HERE/f'SELECTOR-{name}.json'
        path.write_text(json.dumps(game, separators=(',', ':')), encoding='utf-8')
        counts.append({'path': path.name, 'sha256': digest(path),
                       'variables': len(game['variables']), 'contexts': len(game['contexts']),
                       'incidences': sum(len(c['variables']) for c in game['contexts']),
                       'max_context_arity': max(len(c['variables']) for c in game['contexts'])})
    assert digest(FALLBACK) == original_hash
    ledger = {'status': 'BUILT', 'fallback': {'path': FALLBACK.relative_to(ROOT).as_posix(),
                'sha256': original_hash, 'unchanged_during_build': True}, 'objects': counts}
    (HERE/'BUILD-SELECTOR.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    print(json.dumps(ledger))


if __name__ == '__main__':
    main()
