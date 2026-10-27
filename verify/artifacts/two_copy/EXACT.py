"""Independent full final-object algebra audit. Imports no producer/checker.

Arithmetic: exact signed monomials over two commuting negative binary HNN
factors.  All declared squares/row products/row commutators are visited.
Only the root agent executes this prepared computation.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import ctypes
import hashlib
import itertools
import json
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

HERE=Path(__file__).resolve().parent
START=time.monotonic()
FI=(((0,0),),())
EI=(1,FI,FI)


def peak_bytes():
    if os.name=='nt':
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        c=Counters();c.cb=ctypes.sizeof(c)
        ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        h=ctypes.windll.kernel32.GetCurrentProcess()
        assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(h),ctypes.byref(c),c.cb)
        return int(max(c.PeakWorkingSetSize,c.PeakPagefileUsage))
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def guard(deadline):
    assert time.monotonic()-START<deadline,'Deadline reached; no complete PASS claimed'
    assert peak_bytes()<5_500_000_000,'Memory guard reached; no complete PASS claimed'


def base_mul(a,b):
    p,n=a;q,m=b
    return (-1 if (n*q)%2 else 1),(p^q,n+m)


def reduce_factor(bases,stable):
    bases=list(bases);stable=list(stable);sign=1
    while True:
        for i in range(len(stable)-1):
            a,n=bases[i+1]
            if a or stable[i]==stable[i+1]:continue
            if stable[i]==1:middle=2*n
            elif n%2==0:middle=n//2
            else:continue
            s,v=base_mul(bases[i],(0,middle));sign*=s
            s,v=base_mul(v,bases[i+2]);sign*=s
            bases[i:i+3]=[v];stable[i:i+2]=[]
            break
        else:break
    return sign,(tuple(bases),tuple(stable))


@lru_cache(None)
def fmul(a,b):
    ba,xa=a;bb,xb=b
    sign,v=base_mul(ba[-1],bb[0])
    s,nf=reduce_factor(ba[:-1]+(v,)+bb[1:],xa+xb)
    return sign*s,nf


@lru_cache(None)
def finv(a):
    bases,stable=a
    out=FI;sign=1
    for i in reversed(range(len(bases))):
        p,n=bases[i]
        sign*=(-1 if (p*n)%2 else 1)
        s,out=fmul(out,(((p,-n),),()));sign*=s
        if i:
            s,out=fmul(out,(((0,0),(0,0)),(-stable[i-1],)));sign*=s
    return sign,out


@lru_cache(None)
def emul(a,b):
    s0,v0=fmul(a[1],b[1]);s1,v1=fmul(a[2],b[2])
    return a[0]*b[0]*s0*s1,v0,v1


@lru_cache(None)
def einv(a):
    s0,v0=finv(a[1]);s1,v1=finv(a[2])
    return a[0]*s0*s1,v0,v1


@lru_cache(None)
def source_entry(word):
    out=EI
    for letter in word:
        index=abs(letter);direction=1 if letter>0 else -1
        if index==5:
            out=(-out[0],out[1],out[2]);continue
        if index in (1,2,3,4):
            copy=0 if index<=2 else 1
            sign=-1 if index in (2,4) else 1
            nf=(((1,0),),())
        elif index in (6,7):
            copy=index-6;sign=1;nf=(((0,direction),),())
        elif index in (8,9):
            copy=index-8;sign=1;nf=(((0,0),(0,0)),(direction,))
        else:raise ValueError(letter)
        term=(sign,nf,FI) if copy==0 else (sign,FI,nf)
        out=emul(out,term)
    return out


class Matrices:
    def __init__(self):
        self.values=[];self.ids={};self.mul_cache={};self.identity_cache={};self.unit_cache={}

    def put(self,p,w):
        key=tuple(p),tuple(w)
        if key not in self.ids:
            assert len(key[0])==len(key[1]) and sorted(key[0])==list(range(len(key[0])))
            self.ids[key]=len(self.values);self.values.append(key)
        return self.ids[key]

    def identity(self,n):
        if n not in self.identity_cache:self.identity_cache[n]=self.put(range(n),[EI]*n)
        return self.identity_cache[n]

    def mul(self,a,b):
        key=a,b
        if key not in self.mul_cache:
            p,w=self.values[a];q,v=self.values[b];assert len(p)==len(q)
            self.mul_cache[key]=self.put([q[j] for j in p],[emul(w[i],v[j]) for i,j in enumerate(p)])
        return self.mul_cache[key]

    def join(self,tag,a,b):
        p,w=self.values[a];q,v=self.values[b];n=len(p);assert n==len(q)
        if tag=='D':return self.put(p+tuple(n+j for j in q),w+v)
        return self.put(tuple(n+j for j in p)+q,w+v)

    def unit(self,a,sign=1):
        key=a,sign
        if key not in self.unit_cache:
            p,w=self.values[a]
            self.unit_cache[key]=p==tuple(range(len(p))) and all(e==(sign,FI,FI) for e in w)
        return self.unit_cache[key]


def resolve(data,m):
    nodes=[]
    for i,node in enumerate(data['nodes']):
        tag=node[0]
        if tag=='I':image=m.identity(node[1])
        elif tag=='L':
            image=m.put(node[1],[emul((s,FI,FI),source_entry(tuple(w))) for s,w in node[2]])
        else:
            assert tag in ('D','A','M') and 0<=node[1]<i and 0<=node[2]<i
            a,b=nodes[node[1]],nodes[node[2]]
            image=m.mul(a,b) if tag=='M' else m.join(tag,a,b)
        nodes.append(image)
    return nodes


def controls():
    assert source_entry((8,6,-8,-6,-6))==EI
    assert source_entry((-8,6,6,8,-6))==EI
    assert source_entry((-8,6,8))!=(1,FI,FI)
    assert source_entry((6,1,-6,1))==(-1,FI,FI)
    assert source_entry((8,7,-8,-7))==EI
    assert source_entry((8,6,-8,-6))!=EI
    return {'positive_source_controls':4,'negative_source_controls':2}


def check_split(data,m,ops):
    names=data['variable_names'];by_name=dict(zip(names,ops))
    rec=data['lower_recovery'];cover=rec['cover']
    half=cover['half'];paths={int(i):w for i,w in cover['paths'].items()}
    W=[m.values[by_name[name]] for name in cover['W']]
    parity=m.values[by_name[cover['parity']]]
    flip=m.values[by_name[cover['flip']]]
    amp=data['counts']['amplification']
    I=m.identity(amp)

    def word(ws):
        out=I
        for name in ws:out=m.mul(out,by_name[name])
        return out

    assert parity[0]==tuple(range(amp))
    assert all(v[0]==tuple(range(amp)) for v in W)
    assert all(e[1:]==(FI,FI) for v in W for e in v[1])
    assert all(e[1:]==(FI,FI) for e in parity[1])
    copies={}
    for s,original in rec['original_word_map'].items():
        for i in half:
            g=paths[i]
            copies[i,s]=m.values[word(g+original+list(reversed(g))) ]
    count=0
    for r in range(amp):
        active=r if parity[1][r][0]==-1 else flip[0][r]
        assert parity[1][active][0]==-1
        chosen=next((i for i in half if W[i][1][active][0]==-1),None)
        assert chosen is not None
        for s,index in data['source_symbols'].items():
            p,w=copies[chosen,s]
            assert p[active]==active
            entry=w[active]
            if active!=r:
                # The last nested swap has identity entries on every block.
                assert flip[1][r]==EI and flip[1][active]==EI
            assert emul(entry,einv(source_entry((index,))))==EI,('split',r,s)
            count+=1
    assert count==576
    return count


def main():
    p=argparse.ArgumentParser();p.add_argument('--rows',type=int,default=None)
    p.add_argument('--deadline',type=float,default=550)
    args=p.parse_args()
    path=HERE/'TWO-COPY-FINAL-GAME.json';raw=path.read_bytes();data=json.loads(raw)
    assert data['format']=='explicit-parity-game-with-monomial-recipes-v1'
    control=controls();m=Matrices();nodes=resolve(data,m)
    names=data['variable_names'];assert len(names)==len(set(names))==199129
    ops=[nodes[i] for i in data['variable_operators']]
    assert len(ops)==len(names) and all(len(m.values[a][0])==64 for a in ops)
    count=Counter();arities=Counter();used=set();visited=hashlib.sha256()
    for i,A in enumerate(ops):
        assert m.unit(m.mul(A,A)),('square',i,names[i])
        count['squares']+=1
        if i%10000==0:guard(args.deadline)
    split=check_split(data,m,ops)
    limit=len(data['rows']) if args.rows is None else min(args.rows,len(data['rows']))
    for i,row in enumerate(data['rows'][:limit]):
        target=row[0];vs=row[1:]
        assert target in (-1,1) and len(vs)==len(set(vs))
        assert all(0<=v<len(names) for v in vs)
        used.update(vs);arities[len(vs)]+=1
        A=m.identity(64)
        for v in vs:A=m.mul(A,ops[v])
        assert m.unit(A,target),('row',i)
        count['rows']+=1
        for u,v in itertools.combinations(vs,2):
            A,B=ops[u],ops[v]
            comm=m.mul(m.mul(m.mul(A,B),A),B)
            assert m.unit(comm),('commutator',i,u,v)
            count['row_commutators']+=1
        visited.update(i.to_bytes(4,'little'))
        if i%5000==0:
            guard(args.deadline)
            print(json.dumps({'rows_done':i,'unique_exact_matrices':len(m.values),'seconds':time.monotonic()-START}),flush=True)
    # A counterfactual sign mutation must fail on the first declared parity row.
    first=data['rows'][0];A=m.identity(64)
    for v in first[1:]:A=m.mul(A,ops[v])
    assert not m.unit(A,-first[0]);control['wrong_row_sign_rejected']=True
    complete=limit==len(data['rows'])
    if complete:
        assert len(data['rows'])==164608 and used==set(range(len(names)))
        assert count=={'squares':199129,'rows':164608,'row_commutators':493855}
        assert dict(arities)=={3:164606,9:1,2:1}
    result={'status':'PASS' if complete else 'PROBE','artifact_sha256':hashlib.sha256(raw).hexdigest(),
            'counts':dict(count),'all_declared_variables_used':complete and len(used)==len(names),
            'rows_covered':[0,limit],'row_visit_hash':visited.hexdigest(),
            'split_source_block_checks':split,'controls':control,'arity_histogram':dict(arities),
            'unique_exact_matrices':len(m.values),'monomial_product_cache':len(m.mul_cache),
            'source_word_cache':source_entry.cache_info().currsize,
            'factor_product_cache':fmul.cache_info().currsize,
            'peak_process_bytes':peak_bytes(),'seconds':time.monotonic()-START,
            'limits':'Exact audit in the full negative product-HNN algebra, including all final rows. Finite identities do not replace the written arbitrary-state theorem.'}
    out=HERE/('EXACT-CHECKS.json' if complete else 'EXACT-PROBE.json')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
