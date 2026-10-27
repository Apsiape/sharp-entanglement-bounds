"""Complete physical sequential play of the exported FINAL two-copy game.

No source relator/Britton reduction is used here. The physical model is the
tensor product of two actual binary digit models with flat state. Every final
row and raw assignment, including all 512 cover outcomes, is visited. Repeated
Born data are cached only after exact operator-recipe equality. The independent
EXACT.py engine is not imported. Root alone executes this prepared computation.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import cmath
import ctypes
import hashlib
import itertools
import json
import math
import random
import time
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
START=time.monotonic()
LENGTHS=()
IDENTITY2=(1.+0j,0j,0j,1.+0j)
FLIP2=(0j,1.+0j,1.+0j,0j)
PEI=(1,(),())


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


@lru_cache(None)
def physical_free(word):
    # Only exact free-group identities and a=a* are used. In particular yy,
    # y^-1 y^-1, and x^-1 x^-1 DO NOT cancel.
    out=[]
    for letter in word:
        letter=1 if abs(letter)==1 else letter
        if out and (out[-1]==-letter or (letter==1 and out[-1]==1)):out.pop()
        else:out.append(letter)
    return tuple(out)


def winverse(word):
    return tuple(1 if abs(t)==1 else -t for t in reversed(word))


@lru_cache(None)
def parse_source(word):
    sign=1;out=[[],[]]
    for letter in word:
        i=abs(letter);direction=1 if letter>0 else -1
        if i==5:sign=-sign
        elif i in (1,2,3,4):
            copy=0 if i<=2 else 1
            if i in (2,4):sign=-sign
            out[copy].append(1)
        elif i in (6,7):out[i-6].append(2*direction)
        elif i in (8,9):out[i-8].append(3*direction)
        else:raise ValueError(letter)
    # Cross-copy separation is exact in these physical tensor-product models.
    return sign,physical_free(tuple(out[0])),physical_free(tuple(out[1]))


@lru_cache(None)
def epmul(a,b):
    return a[0]*b[0],physical_free(a[1]+b[1]),physical_free(a[2]+b[2])


def two_mul(a,b):
    w,x,y,z=a;r,s,t,u=b
    return w*r+x*t,w*s+x*u,y*r+z*t,y*s+z*u


def two_star(a):
    w,x,y,z=a
    return w.conjugate(),y.conjugate(),x.conjugate(),z.conjugate()


def local_identity(n):
    return tuple(range(n)),(IDENTITY2,)*n


def local_mul(a,b):
    p,w=a;q,v=b
    return tuple(q[j] for j in p),tuple(two_mul(w[i],v[j]) for i,j in enumerate(p))


def local_star(a):
    p,w=a;inv=tuple(p.index(i) for i in range(len(p)))
    return inv,tuple(two_star(w[inv[i]]) for i in range(len(p)))


def local_trace(a):
    p,w=a;seen=set();out=1.+0j
    for i in range(len(p)):
        if i in seen:continue
        cur=i;v=IDENTITY2
        while cur not in seen:
            seen.add(cur);v=two_mul(v,w[cur]);cur=p[cur]
        out*=v[0]+v[3]
    return out/(2**len(p))


@lru_cache(None)
def local_generators(n):
    I=local_identity(n);mats=list(I[1]);mats[0]=FLIP2
    A=(I[0],tuple(mats))
    Y=(I[0],tuple((1.+0j,0j,0j,cmath.exp(2j*math.pi*2**(n-1-j)/(2**n-1))) for j in range(n)))
    X=(tuple((j-1)%n for j in range(n)),I[1])
    out={1:A,2:Y,3:X}
    out.update({-i:local_star(v) for i,v in list(out.items())})
    return out


def local_word(word,n):
    out=local_identity(n);g=local_generators(n)
    for letter in word:out=local_mul(out,g[letter])
    return out


@lru_cache(None)
def factor_trace(word):
    return np.array([local_trace(local_word(word,n)) for n in LENGTHS],complex)


@lru_cache(None)
def entry_inner(a,b):
    return a[0]*b[0]*factor_trace(physical_free(winverse(a[1])+b[1]))*factor_trace(physical_free(winverse(a[2])+b[2]))


class PhysicalMatrices:
    def __init__(self):
        self.values=[];self.ids={};self.products={};self.inner_cache={};self.identities={}

    def put(self,p,w):
        key=tuple(p),tuple(w)
        if key not in self.ids:
            assert len(key[0])==len(key[1]) and sorted(key[0])==list(range(len(key[0])))
            self.ids[key]=len(self.values);self.values.append(key)
        return self.ids[key]

    def identity(self,n):
        if n not in self.identities:self.identities[n]=self.put(range(n),[PEI]*n)
        return self.identities[n]

    def mul(self,a,b):
        key=a,b
        if key not in self.products:
            p,w=self.values[a];q,v=self.values[b];assert len(p)==len(q)
            self.products[key]=self.put([q[j] for j in p],[epmul(w[i],v[j]) for i,j in enumerate(p)])
        return self.products[key]

    def join(self,tag,a,b):
        p,w=self.values[a];q,v=self.values[b];n=len(p);assert n==len(q)
        return self.put(p+tuple(n+j for j in q),w+v) if tag=='D' else self.put(tuple(n+j for j in p)+q,w+v)

    def inner(self,a,b):
        key=a,b
        if key not in self.inner_cache:
            p,w=self.values[a];q,v=self.values[b];assert len(p)==len(q)
            out=np.zeros(len(LENGTHS),complex)
            for i in range(len(p)):
                if p[i]==q[i]:out+=entry_inner(w[i],v[i])
            self.inner_cache[key]=out/len(p)
        return self.inner_cache[key]


def resolve(data,m):
    nodes=[]
    for i,node in enumerate(data['nodes']):
        tag=node[0]
        if tag=='I':A=m.identity(node[1])
        elif tag=='L':A=m.put(node[1],[(s*parse_source(tuple(w))[0],parse_source(tuple(w))[1],parse_source(tuple(w))[2]) for s,w in node[2]])
        else:
            assert tag in ('D','A','M') and 0<=node[1]<i and 0<=node[2]<i
            a,b=nodes[node[1]],nodes[node[2]]
            A=m.mul(a,b) if tag=='M' else m.join(tag,a,b)
        nodes.append(A)
    return nodes


def dense_factor_generators(n):
    d=2**n;digits=list(itertools.product((0,1),repeat=n));index={v:i for i,v in enumerate(digits)}
    A=np.zeros((d,d),complex);X=np.zeros((d,d),complex)
    for i,v in enumerate(digits):
        A[i,index[(1-v[0],)+v[1:]]]=1
        X[i,index[v[1:]+v[:1]]]=1
    Y=np.diag(np.exp(2j*np.pi*np.arange(d)/(d-1)))
    out={1:A,2:Y,3:X};out.update({-i:v.conj().T for i,v in list(out.items())})
    return out


def dense_word(word,gen):
    d=next(iter(gen.values())).shape[0];A=np.eye(d,dtype=complex)
    for letter in word:A=A@gen[letter]
    return A


def dense_entry(e,gen):
    return e[0]*np.kron(dense_word(e[1],gen),dense_word(e[2],gen))


def dense_monomial(A,gen):
    p,w=A;d=next(iter(gen.values())).shape[0]**2;out=np.zeros((len(p)*d,len(p)*d),complex)
    for i,j in enumerate(p):out[i*d:(i+1)*d,j*d:(j+1)*d]=dense_entry(w[i],gen)
    return out


def controls(m,nodes,data):
    rng=random.Random(26102512);largest=0.;count=0
    for n in (2,3):
        gen=dense_factor_generators(n)
        words=[(-2,-2),(-3,-3),(3,2,-3,-2,-2),(-3,2,2,3,-2),(2,1,-2,1)]
        words += [tuple(rng.choice((1,2,3,-1,-2,-3)) for _ in range(rng.randrange(21))) for _ in range(100)]
        for w in words:
            expected=np.trace(dense_word(w,gen))/(2**n)
            actual=local_trace(local_word(w,n));reduced=local_trace(local_word(physical_free(w),n))
            diff=max(abs(actual-expected),abs(reduced-expected));largest=max(largest,float(diff))
            assert diff<3e-12,('dense factor',n,w,actual,expected)
            count+=1
    # Repeated inverse letters are a deliberate regression control.
    assert physical_free((-2,-2))==(-2,-2) and physical_free((-3,-3))==(-3,-3)
    assert physical_free((-2,2))==()
    # Independently build small dense matrices for actual core recipes (16 blocks,
    # two 1-bit source factors). Audit multiplication and normalized inner product.
    gen=dense_factor_generators(1)
    leaf_ids=[nodes[i] for i,node in enumerate(data['nodes']) if node[0]=='L']
    chosen=[]
    for mid in leaf_ids:
        if any(e[1] or e[2] for e in m.values[mid][1]):chosen.append(mid)
        if len(chosen)==8:break
    assert chosen
    dcontrols=0
    for A,B in zip(chosen,chosen[1:]+chosen[:1]):
        DA=dense_monomial(m.values[A],gen);DB=dense_monomial(m.values[B],gen)
        DC=dense_monomial(m.values[m.mul(A,B)],gen)
        diff=float(np.linalg.norm(DA@DB-DC));assert diff<2e-11
        largest=max(largest,diff);dcontrols+=1
        expected=np.trace(DA.conj().T@DB)/len(DA)
        p,w=m.values[A];q,v=m.values[B]
        actual=sum(e[0]*f[0]*local_trace(local_word(physical_free(winverse(e[1])+f[1]),1))*local_trace(local_word(physical_free(winverse(e[2])+f[2]),1))
                   for i,(e,f) in enumerate(zip(w,v)) if p[i]==q[i])/len(p)
        assert abs(actual-expected)<2e-12
        dcontrols+=1
    return {'dense_factor_word_controls':count,'dense_core_monomial_controls':dcontrols,
            'maximum_dense_difference':largest,'repeated_inverse_negative_controls':'PASS'}


def branch_data(m,ops,target):
    r=len(ops);amp=len(m.values[ops[0]][0]);alphas=list(itertools.product((-1,1),repeat=r))
    repair=next(alpha for alpha in alphas if math.prod(alpha)==target)
    prob=[];incident=[];max_imag=0.
    scalar=True;signs=[]
    for mid in ops:
        p,w=m.values[mid]
        if p!=tuple(range(amp)) or any(e[1] or e[2] for e in w):scalar=False;break
        signs.append(np.array([e[0] for e in w]))
    if scalar:
        # Exact Born calculation on diagonal ancillary reflections; in particular
        # the arity-nine cover needs no 512-by-512 Gram matrix.
        for alpha in alphas:
            weights=np.ones(amp)
            for a,s in zip(alpha,signs):weights*=((1+a*s)/2)
            p=float(np.sum(weights)/amp);reported=alpha if math.prod(alpha)==target else repair
            vals=np.array([(p-a*float(np.dot(weights,s)/amp))/2 for a,s in zip(reported,signs)])
            prob.append(np.full(len(LENGTHS),p))
            incident.append(np.repeat(vals[:,None],len(LENGTHS),axis=1))
    else:
        assert r<=3,('unexpected non-scalar high arity',r)
        I=m.identity(amp);terms=[I];subsets=[()]
        for j,B in enumerate(ops):
            terms += [m.mul(A,B) for A in terms]
            subsets += [s+(j,) for s in subsets]
        G=np.array([[m.inner(A,B) for B in terms] for A in terms])
        H=[]
        for B in ops:
            left=[m.mul(B,A) for A in terms]
            H.append(np.array([[m.inner(A,D) for D in left] for A in terms]))
        for alpha in alphas:
            coeff=np.array([math.prod(alpha[j] for j in s) for s in subsets])/2**r
            p=np.einsum('i,ijl,j->l',coeff,G,coeff)
            max_imag=max(max_imag,float(np.max(np.abs(p.imag))))
            reported=alpha if math.prod(alpha)==target else repair
            vals=[]
            for j,a in enumerate(reported):
                corr=np.einsum('i,ijl,j->l',coeff,H[j],coeff)
                max_imag=max(max_imag,float(np.max(np.abs(corr.imag))))
                vals.append((p.real-a*corr.real)/2)
            prob.append(p.real);incident.append(np.array(vals))
    prob=np.array(prob);incident=np.array(incident)
    assert np.min(prob)>-3e-12 and np.max(np.abs(np.sum(prob,axis=0)-1))<3e-12
    assert np.min(incident)>-3e-12 and np.max(incident-prob[:,None,:])<3e-12
    assert max_imag<3e-12
    bad=np.sum([prob[i] for i,a in enumerate(alphas) if math.prod(a)!=target],axis=0)
    return prob,incident,bad,max_imag,scalar


def main():
    global LENGTHS
    p=argparse.ArgumentParser();p.add_argument('--lengths',default='2,3')
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=None)
    p.add_argument('--deadline',type=float,default=550)
    args=p.parse_args();LENGTHS=tuple(int(v) for v in args.lengths.split(','))
    assert LENGTHS and len(set(LENGTHS))==len(LENGTHS) and all(1<=n<=12 for n in LENGTHS)
    path=HERE/'TWO-COPY-FINAL-GAME.json';raw=path.read_bytes();data=json.loads(raw)
    assert data['format']=='explicit-parity-game-with-monomial-recipes-v1'
    m=PhysicalMatrices();nodes=resolve(data,m)
    ops=[nodes[i] for i in data['variable_operators']];assert len(ops)==199129
    control=controls(m,nodes,data);guard(args.deadline)
    I=m.identity(64);squares=np.zeros(len(LENGTHS));square_count=0
    for A in ops:
        defect=np.abs(1-m.inner(I,m.mul(A,A)))
        squares=np.maximum(squares,defect);assert np.max(defect)<3e-12
        square_count+=1
    print(json.dumps({'physical_squares_done':square_count,'unique_physical_matrices':len(m.values),
                      'seconds':time.monotonic()-START,'peak_bytes':peak_bytes()}),flush=True)
    start=args.start;stop=len(data['rows']) if args.stop is None else min(args.stop,len(data['rows']))
    assert 0<=start<stop<=len(data['rows'])
    cache={};loss=np.zeros(len(LENGTHS));max_mass=np.zeros(len(LENGTHS));max_bad=np.zeros(len(LENGTHS))
    max_imag=0.;min_probability=0.;min_incident=0.;branches=0;incidences=0;contexts=0;scalar_contexts=0
    visited=hashlib.sha256();cover_rows=[]
    for index in range(start,stop):
        row=data['rows'][index];target=row[0];vs=row[1:];r=len(vs);row_ops=tuple(ops[v] for v in vs)
        key=row_ops,target
        if key not in cache:cache[key]=branch_data(m,row_ops,target)
        probs,inc,bad,imag,scalar=cache[key]
        mass=np.zeros(len(LENGTHS));local=np.zeros(len(LENGTHS))
        # Actual coverage loop: every final context/outcome/incidence is visited.
        # Cached probabilities are the same exact monomial expression, not a
        # representative-only substitution for an unvisited set of contexts.
        for branch in range(2**r):
            mass+=probs[branch]
            for j in range(r):
                local+=inc[branch,j]/r;incidences+=1
            branches+=1
        loss+=local/len(data['rows']);max_mass=np.maximum(max_mass,np.abs(mass-1))
        max_bad=np.maximum(max_bad,bad);max_imag=max(max_imag,imag)
        min_probability=min(min_probability,float(np.min(probs)));min_incident=min(min_incident,float(np.min(inc)))
        contexts+=1;scalar_contexts+=int(scalar);visited.update(index.to_bytes(4,'little'))
        if r==9:cover_rows.append({'row':index,'all_raw_outcomes':2**r,'scalar_Born_evaluation':scalar})
        if contexts%2000==0:
            guard(args.deadline)
            print(json.dumps({'contexts_done':contexts,'last_row':index,'unique_context_Born_data':len(cache),
                              'unique_physical_matrices':len(m.values),'seconds':time.monotonic()-START,
                              'peak_bytes':peak_bytes()}),flush=True)
    complete=start==0 and stop==len(data['rows'])
    if complete:
        assert contexts==164608 and branches==1317364 and incidences==3955160
        assert len(cover_rows)==1 and cover_rows[0]['all_raw_outcomes']==512
    # Same source errors in both factors; every other source relation is exact.
    records=[]
    for j,n in enumerate(LENGTHS):
        e=2*math.sin(math.pi/(2*(2**n-1)))
        records.append({'binary_word_length':n,'source_dimension':2**(2*n),
                        'game_schmidt_rank':64*2**(2*n),'game_entropy_bits':6+2*n,
                        'source_precision':e,'source_error_squared':e*e,
                        'actual_game_loss':float(loss[j]),'loss_over_source_error_squared':float(loss[j]/(e*e)),
                        'maximum_probability_mass_error':float(max_mass[j]),
                        'maximum_physical_square_trace_error':float(squares[j]),
                        'maximum_forbidden_context_probability':float(max_bad[j])})
    if complete:assert all(r['actual_game_loss']>0 for r in records)
    result={'status':'PASS' if complete else 'SHARD','artifact_sha256':hashlib.sha256(raw).hexdigest(),
            'lengths':LENGTHS,'rows_covered':[start,stop],'row_visit_hash':visited.hexdigest(),
            'all_final_contexts_visited':complete,'contexts':contexts,'raw_outcome_visits':branches,
            'incidence_outcome_visits':incidences,'physical_square_visits':square_count,
            'cover_rows':cover_rows,'records':records,'dense_controls':control,
            'unique_context_Born_data':len(cache),'scalar_contexts_visited':scalar_contexts,
            'unique_physical_matrices':len(m.values),'physical_monomial_product_cache':len(m.products),
            'matrix_inner_cache':len(m.inner_cache),'factor_trace_cache':factor_trace.cache_info().currsize,
            'maximum_imaginary_probability_or_correlation':max_imag,
            'minimum_raw_probability':min_probability,'minimum_incident_error_probability':min_incident,
            'peak_process_bytes':peak_bytes(),'seconds':time.monotonic()-START,
            'arithmetic':'complex floating point; physical tensor-permutation traces independently dense-audited; no source-relator simplification',
            'limits':'Complete actual strategy evaluation for the exported finite game; no optimization, lower-bound proof, or asymptotic matching inference from these finite sizes.'}
    suffix='CHECKS' if complete else str(start)+'-'+str(stop)
    out=HERE/('PLAY-'+suffix+'.json');out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
