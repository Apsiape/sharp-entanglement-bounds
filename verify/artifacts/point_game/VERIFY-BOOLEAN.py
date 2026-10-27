"""PROPOSED independent exact audit of exported BOOLEAN-GAME.json.

No producer imports. Signed free words and terminating shortening rules are
derived directly from the exported source relators. A successful reduction
is a finite defining-relator certificate, never a word-problem oracle.
An incomplete reduction may give a false failure, never an unjustified PASS.
Root alone executes; all output next to this file. Single-threaded.
"""
import os
for task_env in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[task_env] = "1"
import itertools
import json
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVOL = {3,4,5}


def inverse(w):
    return tuple(-x for x in reversed(w))


def clean(sign,w):
    out = []
    for x in w:
        if abs(x) == 6:
            sign = -sign
            continue
        if abs(x) in INVOL:
            x = abs(x)
        if out and (out[-1] == -x or (abs(x) in INVOL and out[-1] == x)):
            out.pop()
        else:
            out.append(x)
    return sign,tuple(out)


def rule_table(relators):
    rules = {}
    for rel in relators:
        sign,w = clean(1,rel)
        if not w:
            assert sign == 1
            continue
        for base in (w,clean(1,inverse(w))[1]):
            for rotate in range(len(base)):
                cyc = base[rotate:]+base[:rotate]
                for n in range(len(cyc)//2+1,len(cyc)+1):
                    lhs = cyc[:n]
                    rhs_sign,rhs = clean(sign,inverse(cyc[n:]))
                    if len(rhs) < len(lhs):
                        value = (rhs_sign,rhs)
                        if lhs not in rules or (len(rhs),rhs_sign,rhs) < (len(rules[lhs][1]),rules[lhs][0],rules[lhs][1]):
                            rules[lhs] = value
    return sorted(rules.items(),key=lambda p:(-len(p[0]),p[0]))


class Matrix:
    # Independent row-monomial representation, no producer arithmetic.
    def __init__(self,p,entries):
        self.p = tuple(p)
        self.e = tuple((int(s),tuple(w)) for s,w in entries)
        self.n = len(self.p)
        assert sorted(self.p) == list(range(self.n))
        assert len(self.e) == self.n
    def __matmul__(self,b):
        assert self.n == b.n
        return Matrix(tuple(b.p[self.p[i]] for i in range(self.n)),
                      [(self.e[i][0]*b.e[self.p[i]][0],self.e[i][1]+b.e[self.p[i]][1]) for i in range(self.n)])
    def star(self):
        p,e = [None]*self.n,[None]*self.n
        for i,j in enumerate(self.p):
            p[j],e[j] = i,(self.e[i][0],inverse(self.e[i][1]))
        return Matrix(p,e)


def identity(n=12):
    return Matrix(range(n),[(1,())]*n)


def load_matrix(d):
    return Matrix(d["row_to_column"],[(c["sign"],c["word"]) for c in d["entries"]])


def main():
    start = time.monotonic()
    data = json.loads((HERE/"BOOLEAN-GAME.json").read_text(encoding="utf-8"))
    rels = data["source_relators"]
    rules = rule_table(rels)
    @lru_cache(maxsize=300000)
    def nf(sign,w):
        sign,w = clean(sign,w)
        while True:
            changed = False
            for lhs,(s,rhs) in rules:
                for i in range(len(w)-len(lhs)+1):
                    if w[i:i+len(lhs)] == lhs:
                        sign,w = clean(sign*s,w[:i]+rhs+w[i+len(lhs):])
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                return sign,w
    def zero(terms,label):
        cells = Counter()
        for c,M in terms:
            for i,(s,w) in enumerate(M.e):
                s,w = nf(s,w)
                cells[i,M.p[i],w] += c*s
        bad = [(k,c) for k,c in cells.items() if c]
        if bad:
            raise AssertionError((label,bad[:3]))
    def isid(M,label):
        zero([(1,M),(-1,identity())],label)
    matrices = {n:load_matrix(d) for n,d in data["variables"].items()}
    counts = Counter()
    for n,M in matrices.items():
        assert M.n == 12
        isid(M@M,("square",n))
        counts["variable_squares"] += 1
    for i,c in enumerate(data["constraints"]):
        row,allowed = c["variables"],{tuple(a) for a in c["allowed"]}
        assert len(set(row)) == len(row) and 1 <= len(row) <= 3
        assert allowed and all(len(a)==len(row) and set(a)<={-1,1} for a in allowed)
        for a,b in itertools.combinations(row,2):
            isid(matrices[a]@matrices[b]@matrices[a].star()@matrices[b].star(),("commutator",i,a,b))
            counts["context_commutators"] += 1
        for answer in itertools.product((-1,1),repeat=len(row)):
            counts["raw_branches"] += 1
            if answer in allowed:
                continue
            # Ordered forbidden-assignment product; omitted denominator 2^arity.
            terms = [(1,identity())]
            for s,n in zip(answer,row):
                terms = terms+[(c0*s,Q@matrices[n]) for c0,Q in terms]
            zero(terms,("forbidden",i,answer))
            counts["forbidden_assignments"] += 1
    # Verify upper/lower composition without the source normalizer's help.
    # Four times P1 U P1, then its A/T conjugates, equals 4 I_12 tensor source s.
    E,A,T = (matrices[n] for n in ("e1","A","T"))
    for index,name in enumerate(data["source_generators"],1):
        U = matrices["V_"+name]@A
        base = [(1,U),(-1,E@U),(-1,U@E),(1,E@U@E)]
        pulled = base+[(c,A@M@A) for c,M in base]+[(c,T@M@T) for c,M in base]
        expected = Matrix(range(12),[(1,(index,))]*12)
        zero(pulled+[(-4,expected)],("lower_after_upper",name))
        counts["whole_state_recovery_generators"] += 1
    # Negative central mark is indispensable in its retained parity row.
    mark_row = next(c for c in data["constraints"] if c["variables"] == ["V_J","A","e3"])
    assert all(a[0]*a[1]*a[2] == -1 for a in mark_row["allowed"])
    # Wrong onsite scalar J=+1 contradicts the source anticommutator rule:
    # J=-1 gave a signed x z x z=-1 relation. Its phase must be retained.
    assert nf(1,(4,5,4,5)) == (-1,())
    assert nf(1,(4,5,4,5)) != (1,())
    counts["negative_scalar_control"] += 1
    result = dict(status="PASS",counts=dict(counts),variables=len(matrices),contexts=len(data["constraints"]),
                  shortening_rules=len(rules),cache=nf.cache_info()._asdict(),seconds=time.monotonic()-start,
                  scope="Every variable square, context commutator, forbidden branch and six whole-state recovery maps; all word reductions use finite valid source relations.")
    (HERE/"VERIFY-BOOLEAN.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
