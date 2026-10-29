"""Independent exact SL(2,Z) verifier, count derivation, and determinant engine.
Does not import engine_a and never freely reduces words. Certificates are input,
but targets, atomic/original relators, arithmetic and count recurrences are rebuilt.
The embedding is faithful (Sanov ping-pong plus a free normal-closure basis).
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json, itertools, math
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
OUT=Path(__file__).resolve().parent
data=json.loads((OUT/'certificates.json').read_text(encoding='utf-8'))
I=(1,0,0,1)
def mm(A,B):
    a,b,c,d=A; e,f,g,h=B
    return (a*e+b*g,a*f+b*h,c*e+d*g,c*f+d*h)
def mi(A):
    a,b,c,d=A; assert a*d-b*c==1
    return (d,-b,-c,a)
P=(1,2,0,1); Q=(1,0,2,1)
G={1:Q,2:mm(mm(P,Q),mi(P)),3:mm(mm(mm(P,P),Q),mi(mm(P,P)))}
G.update({-k:mi(v) for k,v in list(G.items())})
def val(w):
    z=I
    for x in w: z=mm(z,G[x])
    return z
def iv(w): return [-x for x in w[::-1]]
def pw(w,n): return w*n if n>=0 else iv(w)*(-n)
def cj(p,w): return p+w+iv(p)
def cm(x,y): return x+y+iv(x)+iv(y)
a,b,s=[1],[2],[3]
def sig(i): return cj(pw(a,i),s)
t=iv(b)+a; q=sig(-2)
rels={'S2':s*2,'B3':(s+sig(1))*3,'F2':cm(s,sig(2)),
      'H':cm(a,b)+iv(s),'J0':cj(b,s)+iv(sig(1)),
      'N2':cm(sig(-2),b),'N3':cm(sig(-3),b),
      'J1':cj(b,sig(1))+iv(sig(2))}
orig=['S2','B3','F2','H','J0']; order=['F2','H','J0','N2','N3','J1']
for r in orig: assert val(data['original_relators'][r])==val(rels[r])
for r in order: assert val(data['atomic_relators'][r])==val(rels[r])
T=cm(a*2,b*2)
def D(w):
    z=[]; mapping={1:a*2,2:b*2,3:T}
    for c in w: z+=mapping[c] if c>0 else iv(mapping[-c])
    return z
def fsub(w,mapping):
    z=[]
    for c in w: z+=mapping[c] if c>0 else iv(mapping[-c])
    return z
u1=cj(iv(t),q); u2=cj(pw(t,-2),q)
Pi=sig(1)+sig(2)+s+sig(1)
targets={'N2_original':(rels['N2'],[]),'N3_original':(rels['N3'],[]),
         'J1_original':(cj(b,sig(1)),sig(2)),
         'T_free_expansion':(T,cj(a,cm(a,b))+cj(a+b,cm(a,b))+cm(a,b)+cj(b,cm(a,b))),
         'T_to_Pi':(T,Pi),'sigma_minus3':(sig(-3),u1),'sigma_minus4':(sig(-4),u2),
         'd_t':(D(t),t*2+q),'d_q':(D(q),u1+q+u2+u1),
         'E_t':(cj(iv(b),t),t*2+q+iv(t)),'E_q':(cj(iv(b),q),q),
         'comm_s_t':(cm(s,t),[]),'comm_s_q':(cm(s,q),[])}
for j in range(2,6): targets['J'+str(j)]=(cj(b,sig(j)),sig(j+1))
for j in range(1,4): targets['C'+str(j)]=(sig(j),cj(pw(b,j),s))
for j in range(3,7): targets['F'+str(j)]=(cm(s,sig(j)),[])
for j in range(4,7): targets['N'+str(j)]=(cm(sig(-j),b),[])
for r in order:
    if r in ['J0','J1']:
        n=0 if r=='J0' else 1
        targets['D_'+r]=(cj(b*2,cj(a*(2*n),T)),cj(a*(2*n+2),T))
    else: targets['D_'+r]=(D(rels[r]),[])
formal_d={4:[4,4,5],5:[-4,5,4,5,-4,-4,5,4,5,4]}
formal_e={4:[4,4,5,-4],5:[5]}
def text_reduce(w):
    # Different design from engine A's stack: eliminate inverse adjacent pairs
    # from a formal string until a fixed point is reached.
    old=None
    while old!=w:
        old=w
        for pair in ['tT','Tt','qQ','Qq']: w=w.replace(pair,'')
    return w
def formal_string(w):
    return ''.join({4:'t',-4:'T',5:'q',-5:'Q'}[x] for x in w)
assert [len(text_reduce(formal_string(w))) for w in
        [formal_d[4],formal_d[5],fsub(formal_d[4],formal_e),fsub(formal_d[5],formal_e)]]==[3,10,7,26]
for name,x in [('t',4),('q',5)]:
    targets['o_'+name]=(cj(iv(b),D(t if x==4 else q)),fsub(fsub(formal_d[x],formal_e),{4:t,5:q}))
for parity in [0,1]:
    m=3; h=pw(b,-m)+pw(a,m)
    targets['h_even_free' if parity==0 else 'h_odd_free']=(pw(b,-(2*m+parity))+pw(a,2*m+parity),D(h) if parity==0 else iv(b)+D(h)+b+t)
for m in [1,2,3,7,16,31]:
    targets['transfer_free_'+str(m)]=(sig(m)+iv(cj(pw(b,m),s)),cj(pw(b,m),cm(pw(b,-m)+pw(a,m),s)))

def product_cells(cells):
    z=I
    for r,e,w in cells:
        C=val(w); R=val(rels[r]); R=R if e==1 else mi(R)
        z=mm(z,mm(mm(C,R),mi(C)))
    return z
assert set(targets)=={r['name'] for r in data['records']}
total_aux=total_original=0
for r in data['records']:
    name=r['name']; left,right=targets[name]
    expected=mm(val(left),mi(val(right)))
    assert expected==mm(val(r['lhs']),mi(val(r['rhs']))),('target',name)
    assert product_cells(r['cells'])==expected,('atomic',name)
    assert all(c[0] in orig for c in r['original_cells'])
    assert product_cells(r['original_cells'])==expected,('expanded',name)
    assert [sum(c[0]==k for c in r['original_cells']) for k in orig]==r['original_counts']
    total_aux+=len(r['cells']); total_original+=len(r['original_cells'])
    print('SL2Z PASS',name,len(r['cells']),len(r['original_cells']))

# Independently propagate multisets from equations (1)-(5), rather than
# counting words or importing the first program's arithmetic routines.
E={x:Counter({x:1}) for x in order}
f={2:E['F2']}; n={2:E['N2'],3:E['N3']}; j={0:E['J0'],1:E['J1']}; c={0:Counter()}
c[1]=j[0]; c[2]=j[0]+j[1]
def times(C,k): return Counter({a:b*k for a,b in C.items()})
f[3]=f[2]+times(n[2],2)+times(c[1],2)
j[2]=j[1]+times(E['H'],2)+f[3]
c[3]=c[2]+j[2]
for d,J in [(4,1),(5,2),(6,3)]: f[d]=f[3]+times(n[3],2*J)+times(c[J],2)
for d in [4,5,6]: n[d]=n[d-1]+f[d-1]+times(E['H'],2)
for d in [3,4,5]: j[d]=j[d-1]+times(E['H'],2)+f[d+1]
v=times(E['H'],4)+times(E['J0'],2)
rows={
 'F2':times(v,4)+sum((times(f[d],k) for d,k in zip(range(2,7),[1,4,6,4,1])),Counter()),
 'H':Counter(),
 'J0':times(v,2)+sum((times(j[d],k) for d,k in enumerate([1,3,3,1])),Counter()),
 'N2':times(v,2)+times(times(n[3],2)+n[2]+n[4],2),
 'N3':times(v,2)+times(times(n[5],2)+n[4]+n[6],2),
 'J1':times(v,2)+sum((times(j[d],k) for d,k in zip(range(2,6),[1,3,3,1])),Counter())}
M=[[rows[r][k] for k in order] for r in order]
assert M==data['M']
lookup={r['name']:r for r in data['records']}
for name,vec in [(f'F{x}',f[x]) for x in range(3,7)]+[(f'N{x}',n[x]) for x in range(4,7)]+[(f'J{x}',j[x]) for x in range(2,6)]+[(f'C{x}',c[x]) for x in range(1,4)]:
    assert Counter(cell[0] for cell in lookup[name]['cells'])==vec,('hist',name)
for name in order: assert Counter(cell[0] for cell in lookup['D_'+name]['cells'])==rows[name]
assert Counter(abs(x) for x in formal_d[5])==Counter({4:6,5:4})
for name,expected in [('d_t',[0,1,0,0,0,0]),('d_q',[0,4,2,5,1,0]),('o_t',[0,3,0,1,0,0]),('o_q',[0,10,2,9,1,0])]:
    assert [sum(cell[0]==r for cell in lookup[name]['cells']) for r in order]==expected

# Characteristic polynomial by the Leibniz determinant, polynomial convolution.
def polymul(u,v):
    z=[0]*(len(u)+len(v)-1)
    for i,x in enumerate(u):
        for k,y in enumerate(v): z[i+k]+=x*y
    return z
def determinant_poly(A):
    n=len(A); out=[0]*(n+1)
    for p in itertools.permutations(range(n)):
        sign=(-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))
        term=[sign]
        for i,k in enumerate(p): term=polymul(term,[-A[i][k],1] if i==k else [-A[i][k]])
        for i,x in enumerate(term): out[i]+=x
    return out[::-1]
poly=determinant_poly(M); assert poly==[1,-89,824,-3416,-6672,19648,0]
idx=[0,2,3,4,5]; B=[[M[i][j] for j in idx] for i in idx]
B2=[[sum(B[i][k]*B[k][j] for k in range(5)) for j in range(5)] for i in range(5)]
assert min(x for row in B2 for x in row)>0
assert determinant_poly(B)==poly[:-1]
w=[200,1,53,24,151,248]; Mw=[sum(M[i][j]*w[j] for j in range(6)) for i in range(6)]
assert Mw==[15938,0,4197,1886,11984,19720] and all(Mw[i]<=80*w[i] for i in range(6))
with localcontext() as ctx:
    ctx.prec=75
    def fun(x): return x**5-89*x**4+824*x**3-3416*x**2-6672*x+19648
    def deriv(x): return 5*x**4-356*x**3+2472*x**2-6832*x-6672
    assert fun(Decimal(79))==-5063370 and fun(Decimal(80))==30871488
    x=Decimal(80)
    for _ in range(24): x-=fun(x)/deriv(x)
    lo=Decimal('79.14730128683524170696832102134646654642031865175177993139190550')
    hi=lo+Decimal('1e-62')
    assert fun(lo)<0<fun(hi)
    assert fun(Fraction(lo))<0<fun(Fraction(hi)) # exact rational signs
    assert 5*79**4-356*80**3+2472*79**2-6832*80-6672>0
    assert abs(x-Decimal(data['spectral']['lambda_']))<Decimal('1e-60')
    p=x.ln()/Decimal(2).ln(); beta=1/(2*p+1)
    spectral={'lambda':str(x),'p':str(p),'beta':str(beta),'interval':[str(lo),str(hi)]}
print('M =',M); print('B^2 =',B2); print('charpoly =',poly); print('Mw =',Mw)
print('SPECTRAL',json.dumps(spectral))
results={'finite_certificates':len(targets),'atomic_cells_total':total_aux,'original_cells_total':total_original,
         'matrix':M,'B_squared':B2,'charpoly':poly,'spectral':spectral,'method':'exact faithful SL(2,Z); independently specified targets and recurrences; Leibniz determinant'}
(OUT/'engine_b_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('PASS:',len(targets),'certificates in faithful exact matrices, both atomic and original; independent counts and spectrum agree.')

# Numerical diagnostics of the exact analytic proof of (10)-(12).
# These tests do not replace the state-by-state proof in VERIFY.md.
import numpy as np
rng=np.random.default_rng(20261028)
def unitary(d):
    Z=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
    return np.linalg.qr(Z)[0]
def energy(F,rho):
    return sum(np.trace((u-np.eye(len(u))).conj().T@(u-np.eye(len(u)))@rho).real for u in F)/len(F)
holonomy_cases=0
max_telescope_residual=0.; max_root_gap=0.; min_slack10=float('inf'); min_slack11=float('inf')
for A in [1,2,3,5,8]:
    for trial in range(12):
        theta=rng.uniform(-np.pi,np.pi,size=4)
        if trial==0: theta=np.array([0.,np.pi,-np.pi,0.7])
        V=np.diag(np.exp(1j*theta)); weights=rng.random(4); weights/=sum(weights)
        rho=np.diag(weights)
        F=[unitary(4) for _ in range(A-1)]
        product=np.eye(4,dtype=complex)
        for u in F: product=u@product
        F.append(V@product.conj().T)
        suffix=np.eye(4,dtype=complex); telescoped=np.zeros((4,4),dtype=complex)
        for u in F[::-1]:
            telescoped+=suffix@(u-np.eye(4)); suffix=suffix@u
        residual=np.linalg.norm(telescoped-(V-np.eye(4)))
        max_telescope_residual=max(max_telescope_residual,float(residual)); assert residual<2e-13
        bound10=4*sum(weights*np.sin(theta/(2*A))**2)
        slack=energy(F,rho)-bound10; min_slack10=min(min_slack10,float(slack)); assert slack>-2e-13
        roots=[np.diag(np.exp(1j*theta/A))]*A
        gap=abs(energy(roots,rho)-bound10); max_root_gap=max(max_root_gap,float(gap)); assert gap<2e-14
        R=unitary(4); rhogen=R@rho@R.conj().T
        bound11=np.trace((V-np.eye(4)).conj().T@(V-np.eye(4))@rhogen).real/A**2
        slack11=energy(F,rhogen)-bound11; min_slack11=min(min_slack11,float(slack11)); assert slack11>-2e-13
        # Twisted cycle energy for all vector paths, including A=1 and A=2.
        phase=float(theta[trial%4]); diff=np.zeros((A,A),dtype=complex)
        for r in range(A-1): diff[r,r]=-1; diff[r,r+1]=1
        diff[A-1,0]+=np.exp(-1j*phase); diff[A-1,A-1]-=1
        spectrum=np.linalg.eigvalsh(diff.conj().T@diff)
        formula=sorted(4*np.sin((-phase+2*np.pi*np.arange(A))/(2*A))**2)
        assert np.max(np.abs(spectrum-formula))<2e-13
        holonomy_cases+=1
results['holonomy_diagnostics']={'cases':holonomy_cases,'min_slack10':min_slack10,
 'min_slack11':min_slack11,'max_equal_root_gap':max_root_gap,'max_telescope_residual':max_telescope_residual}
(OUT/'engine_b_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('HOLONOMY DIAGNOSTICS',json.dumps(results['holonomy_diagnostics']))
