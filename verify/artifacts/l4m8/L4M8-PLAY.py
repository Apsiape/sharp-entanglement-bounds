"""Complete native linear-game play without storing huge game matrices.

Independent arithmetic: source words are tensor-permutations of q-level digits.
Small dense matrices independently audit the trace engine. Every game context
and outcome is evaluated, for four actual source sizes, without optimization.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import itertools
import json
import math
import random
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
START = time.monotonic()
Q,M = 4,8
LENGTHS = (1,2,3,4)


def local_identity(n):
    return tuple(range(n)),tuple(np.eye(Q,dtype=complex) for _ in range(n))


def local_mul(A,B):
    p,a = A; r,b = B
    return tuple(r[j] for j in p),tuple(a[i]@b[j] for i,j in enumerate(p))


def local_star(A):
    p,a = A
    inv = tuple(p.index(j) for j in range(len(p)))
    return inv,tuple(a[inv[j]].conj().T for j in range(len(p)))


def local_trace(A):
    p,a = A
    visited = set(); result = 1.+0j
    for start in range(len(p)):
        if start in visited: continue
        j = start; B = np.eye(Q,dtype=complex)
        while j not in visited:
            visited.add(j); B = B@a[j]; j = p[j]
        result *= np.trace(B)
    return result / Q**len(p)


def generators(n):
    I = local_identity(n)
    out = {}
    for j in range(Q):
        v = np.exp(2j*np.pi*j*np.arange(Q)/Q)/math.sqrt(Q)
        Z = np.eye(Q)-2*np.outer(v,v.conj())
        mats = list(I[1]); mats[0] = Z
        out[j+1] = (I[0],tuple(mats))
    mats = list(I[1]); mats[0] = -mats[0]
    out[5] = (I[0],tuple(mats))
    den = M**n-1
    out[6] = (I[0],tuple(np.diag(np.exp(2j*np.pi*(M//Q)*M**(n-1-j)*np.arange(Q)/den))
                         for j in range(n)))
    out[7] = (tuple((j-1)%n for j in range(n)),I[1])
    out.update({-i:local_star(A) for i,A in list(out.items())})
    return out


GEN = {n:generators(n) for n in LENGTHS}


def inverse(word):
    return tuple(-i for i in reversed(word))


def eval_word(word,n):
    A = local_identity(n)
    for letter in word: A = local_mul(A,GEN[n][letter])
    return A


TRACE = {}


def word_trace(word):
    # Only exact matrix identities are used here: c=-1 and involutive lamps.
    sign = 1; stack = []
    for letter in word:
        if abs(letter) == 5:
            sign = -sign; continue
        if abs(letter) <= Q: letter = abs(letter)
        if stack and (stack[-1] == -letter or (1<=letter<=Q and stack[-1]==letter)):
            stack.pop()
        else:
            stack.append(letter)
    key = tuple(stack)
    if key not in TRACE:
        TRACE[key] = np.array([local_trace(eval_word(key,n)) for n in LENGTHS])
    return sign*TRACE[key]


def free_reduce(word):
    out = []
    for letter in word:
        if out and out[-1] == -letter: out.pop()
        else: out.append(letter)
    return tuple(out)


def mono_mul(A,B):
    p,a = A; r,b = B
    return tuple(r[j] for j in p),tuple((a[i][0]*b[j][0],free_reduce(a[i][1]+b[j][1]))
                                       for i,j in enumerate(p))


def inner(A,B):
    p,a = A; r,b = B
    value = np.zeros(len(LENGTHS),complex)
    for i in range(len(p)):
        if p[i] == r[i]:
            s,u = a[i]; t,v = b[i]
            value += s*t*word_trace(inverse(u)+v)
    return value/len(p)


def dense_controls():
    rng = random.Random(62106)
    count = 0; largest = 0.
    for n in (1,2):
        d = Q**n; digits = list(itertools.product(range(Q),repeat=n))
        dense = {}
        for j in range(Q):
            v = np.exp(2j*np.pi*j*np.arange(Q)/Q)/math.sqrt(Q)
            dense[j+1] = np.kron(np.eye(Q)-2*np.outer(v,v.conj()),np.eye(Q**(n-1)))
        dense[5] = -np.eye(d,dtype=complex)
        dense[6] = np.diag([np.exp(2j*np.pi*(M//Q)*sum(b*M**(n-1-j) for j,b in enumerate(bs))/(M**n-1))
                            for bs in digits])
        dense[7] = np.zeros((d,d),complex)
        index = {bs:i for i,bs in enumerate(digits)}
        for i,bs in enumerate(digits): dense[7][i,index[bs[1:]+bs[:1]]] = 1
        dense.update({-i:A.conj().T for i,A in list(dense.items())})
        words = [(-6,-6),(-7,-7),(-6,-6,1,6,6,2),(-7,-7,6,7,7,-6)]
        words += [tuple(rng.choice(tuple(dense)) for _ in range(rng.randrange(19)))
                  for _ in range(100)]
        for word in words:
            A = np.eye(d,dtype=complex)
            for letter in word: A = A@dense[letter]
            expected = np.trace(A)/d
            actual = local_trace(eval_word(word,n))
            largest = max(largest,float(abs(actual-expected)))
            assert abs(actual-expected)<2e-12
            normalized = word_trace(word)[LENGTHS.index(n)]
            largest = max(largest,float(abs(normalized-expected)))
            assert abs(normalized-expected)<2e-12
            count += 1
    return {'independent_dense_word_controls':count,'maximum_trace_difference':largest}


def main():
    controls = dense_controls()
    data = json.loads((HERE/'L4M8-LINEAR-GAME.json').read_text())
    amp = data['amplification']
    mats = {name:(tuple(raw['row_to_column']),tuple((entry['sign'],tuple(entry['word']))
                    for entry in raw['entries'])) for name,raw in data['variables'].items()}
    I = (tuple(range(amp)),tuple((1,()) for _ in range(amp)))
    max_square = np.zeros(len(LENGTHS))
    for name,A in mats.items():
        square = mono_mul(A,A)
        error = np.abs(1-inner(I,square))
        max_square = np.maximum(max_square,error)
        assert np.max(error)<2e-12,('physical square',name,error)
    loss = np.zeros(len(LENGTHS)); max_mass = np.zeros(len(LENGTHS))
    max_imag = np.zeros(len(LENGTHS)); max_bad = np.zeros(len(LENGTHS))
    branches = 0
    for context,C in enumerate(data['rows']):
        row,target = C['variables'],C['product_sign']
        k = len(row); Bs = [mats[name] for name in row]
        terms = [I]; subsets = [()]
        for j,B in enumerate(Bs):
            terms += [mono_mul(A,B) for A in terms]
            subsets += [s+(j,) for s in subsets]
        G = np.array([[inner(A,B) for B in terms] for A in terms])
        H = []
        for B in Bs:
            left = [mono_mul(B,A) for A in terms]
            H.append(np.array([[inner(A,D) for D in left] for A in terms]))
        allowed = [a for a in itertools.product((-1,1),repeat=k) if math.prod(a)==target]
        repair = allowed[0]
        mass = np.zeros(len(LENGTHS)); context_loss = np.zeros(len(LENGTHS)); bad = np.zeros(len(LENGTHS))
        for alpha in itertools.product((-1,1),repeat=k):
            coeff = np.array([math.prod(alpha[j] for j in s) for s in subsets])/2**k
            p = np.einsum('i,ijl,j->l',coeff,G,coeff)
            max_imag = np.maximum(max_imag,np.abs(p.imag))
            assert p.real.min()>-2e-12
            mass += p.real
            if math.prod(alpha)!=target: bad += p.real; reported = repair
            else: reported = alpha
            for j,sign in enumerate(reported):
                corr = np.einsum('i,ijl,j->l',coeff,H[j],coeff)
                max_imag = np.maximum(max_imag,np.abs(corr.imag))
                incident = (p.real-sign*corr.real)/2
                assert incident.min()>-2e-12 and np.max(incident-p.real)<2e-12
                context_loss += incident/k
            branches += 1
        loss += context_loss/len(data['rows'])
        max_mass = np.maximum(max_mass,np.abs(mass-1))
        max_bad = np.maximum(max_bad,bad)
        if (context+1)%1000 == 0:
            print(json.dumps({'contexts_done':context+1,'seconds':time.monotonic()-START,
                              'source_trace_cache':len(TRACE)}),flush=True)
    print(json.dumps({'complete_losses':loss.tolist(),'maximum_mass_errors':max_mass.tolist(),
                      'complete_square_errors':max_square.tolist()}),flush=True)
    records = []
    for i,n in enumerate(LENGTHS):
        rels = [(6,j+1,-6,-((j+1)%Q+1)) for j in range(Q)]
        e2 = max(0.,max(2-2*local_trace(eval_word(r,n)).real for r in rels))
        print(json.dumps({'validation_length':n,'source_error_squared':e2,
                          'game_loss':float(loss[i]),'mass_error':float(max_mass[i])}),flush=True)
        assert e2>0 and loss[i]>0 and max_mass[i]<2e-12
        row = {'word_length':n,'source_dimension':Q**n,'game_schmidt_rank':amp*Q**n,
               'source_precision':math.sqrt(e2),'game_entropy':math.log2(amp*Q**n),
               'all_contexts':len(data['rows']),'all_raw_branches':branches,
               'actual_complete_game_loss':float(loss[i]),'loss_over_source_error_squared':float(loss[i]/e2),
               'maximum_normalized_square_trace_defect':float(max_square[i]),
               'maximum_probability_mass_error':float(max_mass[i]),
               'maximum_imaginary_probability_or_correlation':float(max_imag[i]),
               'maximum_context_forbidden_probability':float(max_bad[i])}
        records.append(row); print(json.dumps(row),flush=True)
    slopes = [math.log(b['game_schmidt_rank']/a['game_schmidt_rank'])/
              math.log(a['actual_complete_game_loss']/b['actual_complete_game_loss'])
              for a,b in zip(records,records[1:])]
    result = {'status':'PASS','arithmetic':'complex floating point, tensor-permutation trace; independently dense-audited',
              'q':Q,'m':M,'records':records,'empirical_rank_exponents':slopes,
              'dense_engine_controls':controls,'source_trace_cache':len(TRACE),
              'seconds':time.monotonic()-START,
              'limits':'Complete actual strategies, not optimization or a lower-bound proof. Source and retained-state compiler proofs give the matching one-third rate.'}
    (HERE/'L4M8-PLAY-CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    main()
