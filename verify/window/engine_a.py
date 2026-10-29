"""Proof-carrying equalities for the six-type filling construction.
Only stdlib. Each cell is (relation, sign, conjugating word).
Run this file before engine_b.py. No unavailable archive is used.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json, math, sys
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path

OUT=Path(__file__).resolve().parent
def red(w):
    z=[]
    for x in w:
        if z and z[-1]==-x: z.pop()
        else: z.append(x)
    return tuple(z)
def inv(w): return tuple(-x for x in reversed(w))
def mul(*ws): return red(x for w in ws for x in w)
def power(w,n): return mul(*([w]*n)) if n>=0 else power(inv(w),-n)
a,b,s=(1,),(2,),(3,)
def conj(p,w): return mul(p,w,inv(p))
def sigma(n): return conj(power(a,n),s)
def comm(x,y): return mul(x,y,inv(x),inv(y))
t=mul(inv(b),a); q=sigma(-2)
ORIG={'S2':power(s,2),'B3':power(mul(s,sigma(1)),3),
      'F2':comm(s,sigma(2)), 'H':mul(comm(a,b),inv(s)),
      'J0':mul(b,s,inv(b),inv(sigma(1)))}
AT={**ORIG,'N2':comm(sigma(-2),b),'N3':comm(sigma(-3),b),
    'J1':mul(b,sigma(1),inv(b),inv(sigma(2)))}
RETAIN=['F2','H','J0','N2','N3','J1']
def cc(c,p): return [(r,e,mul(p,w)) for r,e,w in c]
def ci(c): return [(r,-e,w) for r,e,w in reversed(c)]
class Eq:
    def __init__(self,l,r,c=()): self.l,self.r,self.c=red(l),red(r),list(c)
    def then(self,f):
        assert self.r==f.l, ('trans',self.r,f.l)
        return Eq(self.l,f.r,self.c+f.c)
    def rev(self): return Eq(self.r,self.l,ci(self.c))
    def times(self,f):
        return Eq(mul(self.l,f.l),mul(self.r,f.r),self.c+cc(f.c,self.r))
    def under(self,p): return Eq(conj(p,self.l),conj(p,self.r),cc(self.c,p))
    def inverse(self): return Eq(inv(self.l),inv(self.r),cc(ci(self.c),inv(self.l)))
def ident(w): return Eq(w,w)
def prod(*es):
    z=ident(())
    for e in es: z=z.times(e)
    return z
def wrap(e,l=(),r=()): return prod(ident(l),e,ident(r))
def cyclic(w):
    w=red(w); p=[]
    while len(w)>1 and w[0]==-w[-1]: p.append(w[0]); w=w[1:-1]
    return tuple(p),w
def atom(l,r,name):
    """Recognize a freely conjugate +/- cell, without using group relations."""
    p,c=cyclic(mul(l,inv(r))); z,v=cyclic(AT[name])
    for sign in (1,-1):
        h=v if sign==1 else inv(v)
        for k in range(len(h)):
            if c==h[k:]+h[:k]:
                return Eq(l,r,[(name,sign,mul(p,inv(h[:k]),inv(z)))])
    raise AssertionError(('not one cell',name,l,r,c,v))
def as_comm(e,x,y):
    assert e.l==mul(x,y) and e.r==mul(y,x)
    return wrap(e,r=mul(inv(x),inv(y)))
def comm_from(f,x,y):
    e=atom(mul(x,y),mul(y,x),f)
    return e
DERIVED={}; RECORD=[]
def remember(name,e):
    if e.r==(): AT[name]=e.l
    else: AT[name]=mul(e.l,inv(e.r))
    DERIVED[name]=e
    RECORD.append((name,e))
    return e
def lemma(l,r,name):
    z=atom(l,r,name)
    cells=[]
    for _,sign,w in z.c:
        c=DERIVED[name].c
        cells+=cc(c if sign==1 else ci(c),w)
    return Eq(l,r,cells)
def using(name,x,y): return lemma(mul(x,y),mul(y,x),name)
def far(x,y,d):
    return comm_from('F2',x,y) if d==2 else using('F'+str(d),x,y)
def comm_word(x,ys,one):
    # x y1...yk -> y1 x y2...yk -> ... -> y1...yk x
    z=ident(mul(x,*ys)); before=()
    for j,y in enumerate(ys):
        z=z.then(wrap(one(x,y),before,mul(*ys[j+1:])))
        before=mul(before,y)
    return z

# Three fixed expansions into the original five Johnson relators.
v=sigma(1)
aba=atom(conj(a,b),mul(s,b),'H')
aa=aba.under(a).then(wrap(aba,v)) # a^2 b a^-2 = sigma1 s b
braid=atom(mul(s,v,s),inv(mul(v,s,v)),'B3')
braid=braid.then(prod(atom(inv(v),v,'S2'),atom(inv(s),s,'S2'),atom(inv(v),v,'S2')))
j0=atom(mul(b,s),mul(v,b),'J0')
W=conj(power(a,2),b)
commW=wrap(aa,s).then(wrap(braid,r=b)).then(wrap(j0.rev(),mul(v,s))).then(wrap(aa.rev(),r=s))
remember('N2_original',as_comm(commW,s,W).under(power(a,-2)))
assert DERIVED['N2_original'].l==AT['N2']

ba=atom(mul(b,a),mul(inv(s),a,b),'H')
def build_J(i):
    prev='J0' if i==1 else 'J1' if i==2 else 'J'+str(i-1)
    mid=sigma(i-1)
    z=prod(ba,ident(mid),ba.inverse())
    j=atom(conj(b,mid),sigma(i),'J0') if i==1 else atom(conj(b,mid),sigma(i),'J1') if i==2 else lemma(conj(b,mid),sigma(i),prev)
    z=z.then(wrap(j, mul(inv(s),a),mul(inv(a),s)))
    z=z.then(wrap(far(sigma(i+1),s,i+1),l=inv(s)))
    return z
# J1 must use F2 only, and is expanded at the final step of recursion.
J1original=build_J(1)
remember('J1_original',J1original)
def build_N(m):
    x=sigma(-m); y=sigma(-1); z=conj(inv(a),b)
    beq=atom(b,mul(y,z),'H')
    f=far(x,y,m-1)
    n=atom(mul(x,z),mul(z,x),'N'+str(m-1)) if m<=4 else using('N'+str(m-1),x,z)
    eq=wrap(beq,x).then(wrap(f,r=z)).then(wrap(n,l=y)).then(wrap(beq.rev(),r=x))
    return as_comm(eq,x,b)
N3original=build_N(3)
# N2 atomic in this derivation, expanded below.
remember('N3_original',N3original)

def expand(c,original=False):
    out=[]
    for r,e,w in c:
        base={'N2':'N2_original','N3':'N3_original','J1':'J1_original'}.get(r)
        if original and base:
            sub=expand(DERIVED[base].c,True)
            out+=cc(sub if e==1 else ci(sub),w)
        else: out.append((r,e,w))
    return out

# Elementary fillings, in a noncircular dependency order.
def C(m):
    if m==0: return ident(s)
    prev=C(m-1).under(b)
    ji=atom(conj(b,sigma(m-1)),sigma(m),'J0') if m==1 else atom(conj(b,sigma(m-1)),sigma(m),'J1') if m==2 else lemma(conj(b,sigma(m-1)),sigma(m),'J'+str(m-1))
    return ji.rev().then(prev)
def build_F(k,j):
    x=sigma(-k); y=sigma(j); ce=C(j)
    def one(xx,yy):
        if yy in (b,inv(b)):
            return comm_from('N'+str(k),xx,yy) if k<=3 else using('N'+str(k),xx,yy)
        return far(xx,yy,k)
    z=wrap(ce,x).then(comm_word(x,[b]*j+[s]+[inv(b)]*j,one)).then(wrap(ce.rev(),r=x))
    return as_comm(z,x,y).under(power(a,k))
remember('F3',build_F(2,1))
remember('J2',build_J(2))
remember('F4',build_F(3,1))
remember('F5',build_F(3,2))
remember('F6',build_F(3,3))
for i in (3,4,5): remember('J'+str(i),build_J(i))
for i in (1,2,3): remember('C'+str(i),C(i))
for m in (4,5,6): remember('N'+str(m),build_N(m))

T=comm(power(a,2),power(b,2)); Pi=mul(v,sigma(2),s,v)
raw=mul(conj(a,comm(a,b)),conj(mul(a,b),comm(a,b)),comm(a,b),conj(b,comm(a,b)))
remember('T_free_expansion',Eq(T,raw))
hs=atom(comm(a,b),s,'H')
cv=prod(hs.under(a),hs.under(mul(a,b)),hs,hs.under(b))
cv=cv.then(prod(ident(v),wrap(j0,r=inv(b)).under(a),ident(s),wrap(j0,r=inv(b))))
remember('T_to_Pi',cv)
def D(w): return mul(*( {1:power(a,2),2:power(b,2),3:T}[x] if x>0 else inv({1:power(a,2),2:power(b,2),3:T}[-x]) for x in w))
def double_transfer(i):
    def j(r):
        return atom(conj(b,sigma(r)),sigma(r+1),'J'+str(r)) if r<2 else lemma(conj(b,sigma(r)),sigma(r+1),'J'+str(r))
    return j(i).under(b).then(j(i+1))
inds=[1,2,0,1]
# D(F2): move all four left factors across all four right factors.
left=[sigma(i) for i in inds]; right=[sigma(i+4) for i in inds]
def commute_lists(xs,ys,one):
    z=ident(mul(*xs,*ys))
    for k in range(len(xs)-1,-1,-1):
        z=z.then(wrap(comm_word(xs[k],ys,one),mul(*xs[:k]),mul(*xs[k+1:])))
    return z
flookup={sigma(i):i for i in range(-8,10)}
def sigfar(x,y): return far(x,y,abs(flookup[x]-flookup[y]))
cp=commute_lists(left,right,sigfar)
convert=prod(cv,cv.under(power(a,4)))
eq=convert.then(cp).then(prod(cv.under(power(a,4)).rev(),cv.rev()))
remember('D_F2',as_comm(eq,T,conj(power(a,4),T)))
remember('D_H',Eq(D(AT['H']),()))
for name,shift in [('J0',0),('J1',2)]:
    conv=cv.under(power(a,shift))
    eq=conv.under(power(b,2))
    eq=eq.then(prod(*(double_transfer(i+shift) for i in inds)))
    eq=eq.then(cv.under(power(a,shift+2)).rev())
    remember('D_'+name,eq)
for k in (2,3):
    shift=-2*k; xs=[sigma(i+shift) for i in inds]
    conv=cv.under(power(a,shift))
    def nb(x,y):
        m=-flookup[x]
        return comm_from('N'+str(m),x,y) if m<4 else using('N'+str(m),x,y)
    eq=wrap(conv,r=power(b,2)).then(commute_lists(xs,[b,b],nb)).then(wrap(conv.rev(),power(b,2)))
    remember('D_N'+str(k),as_comm(eq,D(sigma(-k)),power(b,2)))

u1=conj(inv(t),q); u2=conj(power(t,-2),q)
uq=atom(sigma(-3),u1,'N2')
remember('sigma_minus3',uq)
commu=wrap(uq.rev(),r=b).then(comm_from('N3',sigma(-3),b)).then(wrap(uq,b))
# sigma_-4 -> a^-1 u1 a = t^-1 b^-1 u1 b t -> t^-1 u1 t
eq=uq.under(inv(a))
eq=eq.then(wrap(commu, mul(inv(t),inv(b)), t))
remember('sigma_minus4',eq)
dt=atom(D(t),mul(t,t,q),'H')
dq=cv.under(power(a,-4)).then(prod(uq,ident(q),eq,uq))
remember('d_t',dt); remember('d_q',dq)
et=atom(conj(inv(b),t),mul(t,t,q,inv(t)),'H')
eqb=atom(conj(inv(b),q),q,'N2')
remember('E_t',et); remember('E_q',eqb)
formal_d={4:(4,4,5),5:mul((-4,5,4),(5,),(-4,-4,5,4,4),(-4,5,4))}
formal_e={4:(4,4,5,-4),5:(5,)}
def formal_sub(w,dic): return mul(*(dic[x] if x>0 else inv(dic[-x]) for x in w))
def evalformal(w): return mul(*(t if x==4 else q if x==5 else inv(t) if x==-4 else inv(q) for x in w))
for label,rule in [('t',dt),('q',dq)]:
    word=formal_d[4 if label=='t' else 5]
    subs=prod(*(et if x==4 else eqb if x==5 else et.inverse() if x==-4 else eqb.inverse() for x in word))
    odd=rule.under(inv(b)).then(subs)
    remember('o_'+label,odd)
remember('comm_s_t',as_comm(atom(mul(s,t),mul(t,s),'J0'),s,t))
remember('comm_s_q',as_comm(atom(mul(s,q),mul(q,s),'F2'),s,q))
for parity in (0,1):
    m=3; hm=mul(power(b,-m),power(a,m))
    h=mul(power(b,-(2*m+parity)),power(a,2*m+parity))
    rhs=D(hm) if not parity else mul(inv(b),D(hm),b,t)
    remember('h_even_free' if not parity else 'h_odd_free',Eq(h,rhs))
for m in (1,2,3,7,16,31):
    hm=mul(power(b,-m),power(a,m))
    remember('transfer_free_'+str(m),Eq(mul(sigma(m),inv(conj(power(b,m),s))),conj(power(b,m),comm(hm,s))))

def verify(e,c,rels):
    expected=mul(e.l,inv(e.r))
    actual=mul(*(conj(w,rels[r] if sign==1 else inv(rels[r])) for r,sign,w in c))
    assert expected==actual
def counts(c,order):
    z=Counter(r for r,_,_ in c)
    assert not (set(z)-set(order)),set(z)-set(order)
    return [z[r] for r in order]
records=[]
for name,e in RECORD:
    verify(e,e.c,AT)
    full=expand(e.c,True); verify(e,full,ORIG)
    records.append(dict(name=name,lhs=e.l,rhs=e.r,cells=e.c,original_cells=full,
                        original_counts=counts(full,list(ORIG))))
    print(name,'aux',dict(Counter(r for r,_,_ in e.c)), 'original',records[-1]['original_counts'])
M=[counts(DERIVED['D_'+r].c,RETAIN) for r in RETAIN]
claimed=[[18,20,64,34,34,12],[0]*6,[5,18,17,10,2,7],[2,12,8,6,6,0],[16,40,52,32,28,4],[22,52,72,44,36,20]]
assert M==claimed,(M,claimed)
def mm(A,B): return [[sum(x*y for x,y in zip(row,col)) for col in zip(*B)] for row in A]
I=[0,2,3,4,5]; B=[[M[i][j] for j in I] for i in I]
assert all(x>0 for row in mm(B,B) for x in row)
# Faddeev-LeVerrier, entirely integer arithmetic.
def charpoly(A):
    n=len(A); C=[[int(i==j) for j in range(n)] for i in range(n)]; out=[1]
    for k in range(1,n+1):
        C=mm(A,C); tr=sum(C[i][i] for i in range(n)); assert tr%k==0
        c=-tr//k; out.append(c)
        C=[[x+c*int(i==j) for j,x in enumerate(row)] for i,row in enumerate(C)]
    return out
coeff=charpoly(M); assert coeff==[1,-89,824,-3416,-6672,19648,0]
def polynomial(x):
    z=0
    for c in coeff[:-1]: z=z*x+c
    return z
w=[200,1,53,24,151,248]; Mw=[sum(x*y for x,y in zip(row,w)) for row in M]
assert Mw==[15938,0,4197,1886,11984,19720]
assert all(x<=80*y for x,y in zip(Mw,w))
assert polynomial(79)==-5063370 and polynomial(80)==30871488
with localcontext() as c:
    c.prec=75; lo,hi=Decimal(79),Decimal(80)
    for _ in range(210):
        mid=(lo+hi)/2
        if polynomial(mid)<0: lo=mid
        else: hi=mid
    lam=(lo+hi)/2; p=lam.ln()/Decimal(2).ln(); beta=1/(2*p+1)
    spectral=dict(interval=[str(lo),str(hi)],lambda_=str(lam),p=str(p),beta=str(beta))
print('M =',M); print('B =',B); print('charpoly =',coeff); print('Mw =',Mw)
print('SPECTRAL',json.dumps(spectral))
for r,cost in [('N2_original',9),('N3_original',12),('J1_original',4)]:
    assert len(expand(DERIVED[r].c,True))==cost
for r in ('d_t','d_q','o_t','o_q'):
    print('RULE',r,counts(DERIVED[r].c,RETAIN))
formal_o={x:formal_sub(w,formal_e) for x,w in formal_d.items()}
assert [len(formal_d[4]),len(formal_d[5]),len(formal_o[4]),len(formal_o[5])]==[3,10,7,26]
def W(n):
    if n==1: return (4,)
    return mul(formal_sub(W(n//2),formal_d if n%2==0 else formal_o),(4,) if n%2 else ())
for n in range(1,32):
    z=W(n); assert sum(1 if x==4 else -1 if x==-4 else 0 for x in z)==n
print('Formal rewriting: n=1..31; t-exponents and lengths checked; max length',max(len(W(n)) for n in range(1,32)))
data=dict(description='Proof cells reconstructed from the six-type filling construction',
          original_relators=ORIG,atomic_relators={r:AT[r] for r in RETAIN},
          retained_order=RETAIN,records=records,M=M,B=B,charpoly=coeff,weight=w,Mw=Mw,spectral=spectral)
(OUT/'certificates.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
(OUT/'engine_a_results.json').write_text(json.dumps({k:v for k,v in data.items() if k!='records'},indent=2),encoding='utf-8')
print('PASS:',len(records),'finite certificates checked both atomically and fully expanded; matrix and spectral certificates verified.')
