"""Deterministic finite checks of spectrum, leakage and budgets; not a numerical E_D oracle.
All matrix work is single threaded. Largest dense matrix has order 256.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import itertools, json, math, pathlib, ctypes, time
from fractions import Fraction
import numpy as np

OUT=pathlib.Path(__file__).resolve().parent/'recorded'
OUT.mkdir(exist_ok=True)
RNG=np.random.default_rng(20261028)
RESULTS={}
TOL=3e-9
def op(a): return float(np.linalg.norm(a,2))
def hs(a): return float(np.linalg.norm(a,'fro'))
def dagger(a): return a.conj().T
def ent(a):
    x=np.linalg.eigvalsh((a+dagger(a))/2)
    x=x[x>1e-14]
    return float(-np.sum(x*np.log2(x)))
def shannon(x):
    x=np.asarray(x); x=x[x>1e-15]
    return float(-np.sum(x*np.log2(x)))
def h2(x):
    if x<=0 or x>=1:return 0.
    return -x*math.log2(x)-(1-x)*math.log2(1-x)
def g(x):return (1+x)*h2(x/(1+x)) if x else 0.
def root(a):
    d,v=np.linalg.eigh(a)
    return (v*np.sqrt(np.maximum(d,0)))@dagger(v)
def state(n):
    a=RNG.normal(size=(n,n))+1j*RNG.normal(size=(n,n))
    a=a@dagger(a);return a/np.trace(a)
def unitary(n):
    a=RNG.normal(size=(n,n))+1j*RNG.normal(size=(n,n))
    q,r=np.linalg.qr(a);return q@np.diag(np.diag(r)/np.abs(np.diag(r)))
def expi(a,t):
    d,v=np.linalg.eigh(a);return (v*np.exp(1j*t*d))@dagger(v)
def bob(a,da,db):return np.trace(a.reshape(da,db,da,db),axis1=0,axis2=2)
def alice(a,da,db):return np.trace(a.reshape(da,db,da,db),axis1=1,axis2=3)
def td(a,b):return float(np.sum(np.abs(np.linalg.eigvalsh(a-b)))/2)
def phi(d):return np.eye(d,dtype=complex).reshape(-1)/math.sqrt(d)
def pure(v):return np.outer(v,v.conj())
X=np.array([[0,1],[1,0]],complex);Z=np.diag([1.,-1.]);I2=np.eye(2)
def record(name,**data):RESULTS[name]=data;print(name, json.dumps(data),flush=True)

start=time.perf_counter()
# Fourier equal-loss decomposition: noncommuting, full-rank mixed inputs.
max_loss=0.;max_rec=0.;min_entropy_slack=1e9
for da,db in [(2,2),(2,3),(3,3),(3,4)]:
    n=da*db
    for _ in range(60):
        rho=state(n);u=unitary(n);loss=u@np.diag(RNG.random(n))@dagger(u)
        ell=float(np.trace(rho@loss).real);s=root(rho)
        _,v=np.linalg.eigh(s@(loss-ell*np.eye(n))@s)
        fourier=np.exp(2j*np.pi*np.outer(np.arange(n),np.arange(n))/n)/math.sqrt(n)
        cols=s@v@fourier
        rec=cols@dagger(cols);max_rec=max(max_rec,op(rec-rho))
        avg=0.
        for col in cols.T:
            q=float(np.vdot(col,col).real);psi=col/math.sqrt(q)
            max_loss=max(max_loss,abs(float(np.vdot(psi,loss@psi).real)-ell))
            avg+=q*ent(bob(pure(psi),da,db))
        min_entropy_slack=min(min_entropy_slack,ent(bob(rho,da,db))-avg)
assert max_loss<TOL and max_rec<TOL and min_entropy_slack>-TOL
record('equal_loss_decomposition',instances=240,max_loss_error=max_loss,max_reconstruction_error=max_rec,min_entropy_slack=min_entropy_slack)

# The entropy-tail and normalization estimates, including highly uneven spectra.
max_tail=0.;max_ent=0.
for _ in range(400):
    n=int(RNG.integers(4,100));lam=np.sort(RNG.dirichlet(np.full(n,0.15)))[::-1]
    r=int(RNG.integers(2,n));tail=float(lam[r:].sum());H=shannon(lam)
    max_tail=max(max_tail,tail*math.log2(r)-H)
    trunc=lam[:r]/(1-tail)
    max_ent=max(max_ent,shannon(trunc)-H/(1-tail))
assert max_tail<TOL and max_ent<TOL
record('schmidt_truncation',instances=400,max_tail_bound_violation=max_tail,max_entropy_bound_violation=max_ent)

# Flag randomization LP: enumerate all one- and two-point feasible vertices.
max_overhead=0.;max_increase=0.
for _ in range(300):
    ell=RNG.random(9);S=8*RNG.random(9);p=RNG.dirichlet(np.ones(9));eps=float(p@ell)
    candidates=[(float(S[i]),[1.]) for i in range(9) if ell[i]<=eps]
    for i in range(9):
        for j in range(i):
            if (ell[i]-eps)*(ell[j]-eps)<0:
                q=(eps-ell[j])/(ell[i]-ell[j]);candidates.append((float(q*S[i]+(1-q)*S[j]),[q,1-q]))
    best,q=min(candidates,key=lambda c:c[0]);max_increase=max(max_increase,best-float(p@S))
    max_overhead=max(max_overhead,shannon(q))
assert max_increase<TOL and max_overhead<=1+TOL
record('flag_two_branch_rounding',instances=300,max_cost_increase=max_increase,max_coherent_flag_overhead=max_overhead)

# Positive loss projection, residual row, and conditional-entropy estimates.
max_proj=-1.;max_row=-1.;max_cond=-1.
for _ in range(200):
    n=12;rho=state(n);u=unitary(n);k=int(RNG.integers(1,n))
    P=u[:,:k]@dagger(u[:,:k]);q=float(np.trace(rho@(np.eye(n)-P)).real)
    sigma=P@rho@P/(1-q);L=state(n);L=L/op(L);eps=float(np.trace(rho@L).real)
    max_proj=max(max_proj,float(np.trace(sigma@L).real)-(math.sqrt(eps)+math.sqrt(q))**2/(1-q))
    assert td(rho,sigma)<=math.sqrt(q)+TOL
    residuals=[(RNG.normal(size=(n,n))+1j*RNG.normal(size=(n,n)))/6 for _ in range(3)]
    Q=sum(float(np.trace(rho@dagger(r)@r).real) for r in residuals)
    K=math.sqrt(sum(op(r)**2 for r in residuals))
    actual=sum(float(np.trace(sigma@dagger(r)@r).real) for r in residuals)
    max_row=max(max_row,actual-(math.sqrt(Q)+K*math.sqrt(q))**2/(1-q))
    da,db=3,4;a=state(n);b=(1-RNG.random())*a+RNG.random()*state(n);b=b/np.trace(b)
    t=td(a,b);cond=lambda s:ent(s)-ent(bob(s,da,db))
    max_cond=max(max_cond,abs(cond(a)-cond(b))-2*t*math.log2(da)-g(t))
assert max(max_proj,max_row,max_cond)<TOL
record('projection_and_entropy',instances=200,max_loss_bound_violation=max_proj,max_residual_bound_violation=max_row,max_conditional_entropy_violation=max_cond)

# Explicit one-way subset conversion, completeness and normalized branch states.
max_complete=0.;max_branch=0.;conversion_count=0
for d in range(2,10):
    for m in range(2,d+1):
        accum=np.zeros((d,d));den=math.comb(d-1,m-1)
        for subset in itertools.combinations(range(d),m):
            K=np.zeros((m,d));K[np.arange(m),subset]=1/math.sqrt(den)
            accum+=K.T@K
            # Bob's branch reindexing has the same indicated matrix entries.
            reindex=np.zeros((m,d));reindex[np.arange(m),subset]=1
            branch=np.kron(K,reindex)@phi(d);branch/=np.linalg.norm(branch)
            max_branch=max(max_branch,float(np.linalg.norm(branch-phi(m))))
        max_complete=max(max_complete,op(accum-np.eye(d)));conversion_count+=1
assert max_complete<TOL and max_branch<TOL
record('subset_conversion',dimension_pairs=conversion_count,max_completeness_error=max_complete,max_branch_error=max_branch)

# Exact synchrony with coherence and multiplicity, then a tilted no-kernel pair.
# Two inequivalent irreducible sectors: dimensions 1 and 2; second has multiplicity 2.
da=5;Pbig=np.zeros((da*da,5),complex);Pbig[0,0]=1
for a,b in itertools.product(range(2),repeat=2):
    for r in range(2):Pbig[(1+2*r+a)*da+(1+2*r+b),1+2*a+b]=1/math.sqrt(2)
rho=Pbig@state(5)@dagger(Pbig)
sector=np.diag([0,1,1,1,1]);Qz=np.diag([1,1,1,-1,-1]);Qx=np.eye(5,dtype=complex);Qx[1:,1:]=np.kron(X,I2)
H=sum(dagger(np.kron(Q.T,np.eye(da))-np.kron(np.eye(da),Q))@(np.kron(Q.T,np.eye(da))-np.kron(np.eye(da),Q)) for Q in [sector,Qz,Qx])
assert abs(np.trace(rho@H))<TOL
w=float(np.trace(rho@np.kron(sector,sector)).real)
record('exact_sync_multiplicity',joint_dimension=25,synchrony_expectation=float(np.trace(rho@H).real),entangled_sector_weight=w)
tilt=[]
for theta in [.001,.01,.1,.5,1.]:
    D1=np.kron(Z,I2)-np.kron(I2,Z);D2=np.kron(X,I2)-np.kron(I2,math.cos(theta)*X+math.sin(theta)*Z)
    H=D1@D1+D2@D2;mineig=float(np.linalg.eigvalsh(H)[0]);value=float(np.vdot(phi(2),H@phi(2)).real)
    assert mineig>0 and abs(value-2*(1-math.cos(theta)))<TOL
    tilt.append({'theta':theta,'min_eigenvalue':mineig,'mirror_energy':value})
record('tilted_mirrors',instances=tilt)

# Extendibility singlet bound: exact spectral norm, not a rate computation.
extend=[]
for d in range(2,7):
    V=np.zeros((d**3,d));W=np.zeros_like(V)
    for a,b,c in itertools.product(range(d),repeat=3):
        if a==b:V[(a*d+b)*d+c,c]=1/math.sqrt(d)
        if a==c:W[(a*d+b)*d+c,b]=1/math.sqrt(d)
    top=float(np.linalg.eigvalsh(V@V.T+W@W.T)[-1])
    assert abs(top-(1+1/d))<TOL
    extend.append({'d':d,'norm_P_plus_Q':top})
record('extendible_singlet_bound',instances=extend)
erasures=[]
for d in range(2,8):
    v=np.zeros(d*(d+1));v[[i*(d+1)+i for i in range(d)]]=1/math.sqrt(d)
    rho=.5*pure(v)
    for i in range(d):rho[i*(d+1)+d,i*(d+1)+d]+=.5/d
    joint=ent(rho);bs=ent(bob(rho,d,d+1));as_=ent(alice(rho,d,d+1))
    assert abs(joint-1-.5*math.log2(d))<TOL and abs(joint-bs)<TOL and abs(as_-math.log2(d))<TOL
    erasures.append({'d':d,'S_AB':joint,'S_B':bs,'S_A':as_,'displayed_formation_cost':.5*math.log2(d)})
record('half_erasure',instances=erasures)

# Thin BS tuple: modular arithmetic avoids large phase cancellation.
thin=[];R=np.array([[1,1],[1,-1]])/math.sqrt(2);omega=np.exp(2j*np.pi/3)
for n in [3,5,7,9,11]:
    q=3**n-2**n;h=(q-1)//3;a=[3**j*2**(n-1-j) for j in range(n)]
    c=[1,1+(h*pow(a[0],-1,q))%q];dim=2*n
    U=np.diag([np.exp(2j*np.pi*((c[r]*a[j])%q)/q) for j in range(n) for r in range(2)])
    V0=np.zeros((dim,dim))
    for j in range(n):V0[2*((j-1)%n):2*((j-1)%n)+2,2*j:2*j+2]=I2
    RH=np.eye(dim);RH[:2,:2]=R;V=RH@V0
    F=V@U@U@dagger(V)@np.linalg.matrix_power(dagger(U),3);eta=op(F-np.eye(dim))
    A=V@U@dagger(V);W=U@A@dagger(U)@dagger(A)
    D=np.eye(dim,dtype=complex);D[:2,:2]=np.diag([omega.conjugate(),omega])
    theta=2*math.asin(math.sqrt(2)*math.sin(math.pi/q)/2)
    assert abs(eta-math.sqrt(2)*math.sin(math.pi/q))<TOL
    assert theta>=2*math.sqrt(2)/q-TOL and op(W-D)<=10*math.pi/(3*q)+TOL
    G=[];B=[]
    for j in range(n):
        Vj=np.linalg.matrix_power(V,j);Vk=Vj@V
        G.append(Vj@D@dagger(Vj));B.append(Vk@U@dagger(Vk))
    triangular=max([op(B[j]@G[l]-G[l]@B[j]) for j in range(n) for l in range(j+1,n)] or [0])
    flips=max(op(dagger(B[j])@G[j]@B[j]-dagger(G[j])) for j in range(n))
    assert triangular<TOL and flips<=10*eta+TOL
    thin.append({'n':n,'eta_BS':eta,'W_to_D':op(W-D),'triangular_error':triangular,'flip_over_eta':flips/eta})
record('thin_BS_geometry',instances=thin)

# Highest weight no-wrap assertion, sampled over high weights and small angles.
checked=0
for theta in np.geomspace(1e-5,.78,80):
    for ell in list(range(1,80))+[100,300,1000,10000]:
        weights=np.arange(-ell,ell+1,2);err=float(np.max(np.abs(np.exp(1j*theta*weights)-1)))
        if ell*theta>=math.pi/2:assert err>=math.sqrt(2)-TOL
        if err<.5:assert ell<=math.pi*err/(2*theta)+TOL
        checked+=1
record('SU2_highest_weights',cases=checked)

# Negative-marker polar compression, with noncommuting perturbations.
max_compression=0.;max_invariance=0.
for _ in range(100):
    n=8;k=3;P=np.diag([1.]*k+[0.]*(n-k));H=np.eye(n)-2*P
    J=H@np.diag(np.exp(1j*.002*RNG.uniform(-1,1,n)))
    block=np.zeros((n,n),complex);block[:k,:k]=unitary(k);block[k:,k:]=unitary(n-k)
    A=state(n);U=expi(A,.02)@block
    eta=max(op(J@J-np.eye(n)),op(J@U-U@J))
    assert op(H-J)<=eta+TOL and op(P@U-U@P)<=1.5*eta+TOL
    Q=np.zeros_like(U)
    for sl in [slice(0,k),slice(k,n)]:
        l,_,r=np.linalg.svd(U[sl,sl]);Q[sl,sl]=l@r
    max_compression=max(max_compression,op(Q-U)-3*eta)
    max_invariance=max(max_invariance,op(Q@P-P@Q))
assert max_compression<TOL and max_invariance<TOL
record('marker_compression',instances=100,max_three_eta_bound_violation=max_compression,max_Q_commutator=max_invariance)

# Near-invariant-support pinching and the ternary conditional flip inequality.
max_leak=0.;max_pinched_entropy=0.;min_cond_slack=1e9
for _ in range(60):
    m=3;n=3**m;k=7
    coords=list(itertools.product(range(3),repeat=m))
    G=[np.diag([omega**x[j] for x in coords]) for j in range(m)]
    P0=np.diag([1.]*k+[0.]*(n-k));v=expi(state(n),.1);P=v@P0@dagger(v);rho=P/k
    pinched=np.diag(np.diag(rho));alpha=float(np.trace((np.eye(n)-P)@pinched).real)
    Gamma=sum(op(P@g_-g_@P) for g_ in G)
    max_leak=max(max_leak,alpha-Gamma**2)
    bound=math.log2(k)+h2(alpha)+alpha*m*math.log2(3)
    max_pinched_entropy=max(max_pinched_entropy,ent(pinched)-bound)
    law=RNG.dirichlet(np.ones(n)).reshape((3,)*m)
    for j in range(m):
        joint=np.sum(law,axis=tuple(range(j))) if j else law
        flip=joint[[0,2,1],...];t=float(np.sum(np.abs(joint-flip))/2)
        tail=joint.sum(axis=0);cond=shannon(joint.flatten())-shannon(tail.flatten());active=float(joint[1:].sum())
        min_cond_slack=min(min_cond_slack,cond-active+h2(t/2)+t/2)
assert max_leak<TOL and max_pinched_entropy<TOL and min_cond_slack>-TOL
record('pinching_and_conditional_flips',instances=60,max_leakage_bound_violation=max_leak,max_entropy_bound_violation=max_pinched_entropy,min_conditional_entropy_slack=min_cond_slack)

# Slow-mode mass constraint, with weight not merely an unused small eigenvalue.
min_slow_slack=1.
for _ in range(300):
    eig=np.r_[np.zeros(3),10**RNG.uniform(-10,0,15)];weight=RNG.dirichlet(np.ones(18))
    q=float(weight[eig>0].sum());mean=float(weight@eig);threshold=2*mean/q
    mass=float(weight[(eig>0)&(eig<=threshold)].sum());min_slow_slack=min(min_slow_slack,mass-q/2)
assert min_slow_slack>-TOL
record('populated_slow_modes',instances=300,min_mass_slack=min_slow_slack)

# Pairwise upper family: enumerate flags, calculate actual matrix defects.
collisions=[]
for M,K in [(2,1),(3,2),(4,2),(4,3),(3,4)]:
    dim=2**K
    def pauli(j,Q):
        ans=np.ones((1,1))
        for k in range(K):ans=np.kron(ans,Q if j==k else I2)
        return ans
    xs=[pauli(j,X) for j in range(K)];zs=[pauli(j,Z) for j in range(K)]
    total=0.;flags=K**M
    for f in itertools.product(range(K),repeat=M):
        j,l=0,1;C=xs[f[j]]@zs[f[l]]-zs[f[l]]@xs[f[j]]
        total+=hs(C)**2/dim/flags
        assert hs(xs[f[j]]@zs[f[j]]+zs[f[j]]@xs[f[j]])<TOL
        assert np.linalg.norm((np.kron(xs[f[j]].T,np.eye(dim))-np.kron(np.eye(dim),xs[f[j]]))@phi(dim))<TOL
    assert abs(total-4/K)<TOL
    collisions.append({'M':M,'K':K,'flags':flags,'cross_defect_squared':total,'coherent_information':float(K),'pure_ensemble_cost':float(K)})
record('pairwise_collision_family',instances=collisions)
record('large_error_product_strategy',M=1,kappa=2.0,local_dimension=1,reflections='all identities',same_pair_signed_defect=2.0,mirror_defect=0.0,original_state_rate=0.0,min_M_kappa_inverse_squared=.25)

# Exact Laurent-polynomial attack on the algebraic freeness lemma for F.9.
# B(z) has a common denominator 2; track integer numerators only.
def matmul2(a,b):
    return (a[0]*b[0]+a[1]*b[2],a[0]*b[1]+a[1]*b[3],a[2]*b[0]+a[3]*b[2],a[2]*b[1]+a[3]*b[3])
letters={
    'a':{1:(1,0,0,0),-1:(0,0,0,1)},
    'A':{-1:(1,0,0,0),1:(0,0,0,1)},
    'b':{1:(1,1,1,1),-1:(1,-1,-1,1)},
    'B':{-1:(1,1,1,1),1:(1,-1,-1,1)},
}
inverse={'a':'A','A':'a','b':'B','B':'b'}
laurent_count=0
def walk(word,poly):
    global laurent_count
    if word:
        trace={e:v[0]+v[3] for e,v in poly.items()}
        assert any(e!=0 and c!=0 for e,c in trace.items()),word
        laurent_count+=1
    if len(word)==8:return
    for letter,gen in letters.items():
        if word and letter==inverse[word[-1]]:continue
        nxt={}
        for e,v in poly.items():
            for f,w in gen.items():
                z=matmul2(v,w);old=nxt.get(e+f,(0,0,0,0));nxt[e+f]=tuple(x+y for x,y in zip(old,z))
        walk(word+letter,nxt)
walk('',{0:(1,0,0,1)})
assert laurent_count==13120
record('polynomial_augmentation_Laurent_freeness',nontrivial_reduced_words=laurent_count,max_word_length=8,arithmetic='exact integers with implicit common powers-of-two denominators')

# Check a rigorous universal small-error lower constant from [P] Theorem 6.4.
min_lower_ratio=1.
for kappa in np.geomspace(1e-6,.01,100):
    for M in [1,2,3,10,100,10000,10**10]:
        N=min(M,math.floor(1/(10000*kappa*kappa)))
        p=50*N*kappa*kappa;rate=N*(1-2*h2(p))
        ratio=rate/min(M,1/(kappa*kappa));min_lower_ratio=min(min_lower_ratio,ratio)
        assert ratio>=1/40000-TOL
record('small_error_pairwise_budget_lower_constant',cases=700,kappa_max=.01,c=1/40000,min_verified_rate_ratio=min_lower_ratio)

# Decidable abstract array with delayed false witnesses: exact rational arithmetic.
qs=[Fraction(1,4),Fraction(9,10),Fraction(1,2),Fraction(4,5),Fraction(3,5),Fraction(11,20)]
cutoffs=[None,101,None,503,None,None]
def a(n,D,N):
    return n*max([Fraction(0)]+[q for i,(q,cut) in enumerate(zip(qs,cutoffs)) if i<n and (cut is None or D<cut) and N>=D+i])
abstract_checks=0
for n in range(1,7):
    Ds=sorted(set([2**n,100,101,502,503,1024]) & set(range(2**n,1025)))
    expected=n*max([Fraction(0)]+[q for i,q in enumerate(qs[:n]) if cutoffs[i] is None])
    assert min(a(n,D,4096) for D in Ds)==expected
    for D in Ds:
        for N in [1,10,100,500,2000]:
            assert a(n,D,N)<=a(n,D,N+1)
            if D+1>=2**n:assert a(n,D+1,N)<=a(n,D,N)
            if n<6 and D>=2**(n+1):assert a(n,D,N)<=a(n+1,D,N)
            for m in [1,100,1000]:assert (N+m)*a(n,D,N+m)>=N*a(n,D,N)+m*a(n,D,m)
            abstract_checks+=1
record('abstract_arithmetic_array',grid_points=abstract_checks,delayed_false_witnesses=[101,503],all_exact_rational_checks_passed=True)

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
pmc=PROCESS_MEMORY_COUNTERS();pmc.cb=ctypes.sizeof(pmc)
get_process=ctypes.windll.kernel32.GetCurrentProcess;get_process.restype=ctypes.c_void_p
get_memory=ctypes.windll.psapi.GetProcessMemoryInfo
get_memory.argtypes=[ctypes.c_void_p,ctypes.POINTER(PROCESS_MEMORY_COUNTERS),ctypes.c_ulong]
get_memory.restype=ctypes.c_int
assert get_memory(get_process(),ctypes.byref(pmc),pmc.cb)!=0
RESULTS['run']={'seed':20261028,'single_thread_environment':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'elapsed_seconds':time.perf_counter()-start,'peak_working_set_bytes':int(pmc.PeakWorkingSetSize),'numpy_version':np.__version__,'test_groups':len(RESULTS)}
assert 0<pmc.PeakWorkingSetSize<1_400_000_000
(OUT/'refinements-results.json').write_text(json.dumps(RESULTS,indent=2),encoding='utf-8')
print('ALL FINITE CHECKS PASSED',flush=True)
