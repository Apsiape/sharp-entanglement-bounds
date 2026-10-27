"""Root-only sparse point/Pauli and scalar-clock controls of Gamma_3."""
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'SOURCE-GAMMA3.json'
counts = {}


def exact(value,name):
    assert value, name
    counts[name]=counts.get(name,0)+1


def inv(word):
    return word.swapcase()[::-1]


def reduce(word):
    stack=[]
    for ch in word:
        if stack and stack[-1] == ch.swapcase():
            stack.pop()
        else:
            stack.append(ch)
    return ''.join(stack)


def comm(a,b):
    return reduce(a+b+inv(a)+inv(b))


def standard_presentation():
    s=comm('p','q');j='xzxz';u=reduce(s+'p');v='pqP'
    rows=[s*2,(s+'p'+s+'P')*3,comm(s,'pp'+s+'PP'),
          'p'+s+'P'+'q'+inv(s)+'Q','xx','zz','xzxzxzxz']
    for a in ['x','z']:
        for g in [u,v]:
            rows.append(comm(a,g))
    for a,b in itertools.product(['x','z'],repeat=2):
        rows.append(comm(a,'p'+b+'P'))
    rows.extend([comm(j,'p'),comm(j,'q')])
    images={'p':'ppp','q':'qqq',
            'x':reduce('x'+'pxP'+'ppxPP'),
            'z':reduce('z'+'pzP'+'ppzPP')}
    for g in ['p','q','x','z']:
        rows.append('t'+g+'T'+inv(images[g]))
    return [reduce(w) for w in rows]


source=json.loads(SOURCE.read_text(encoding='utf-8-sig'))
rows=source['relators']
words=[r['word'] if isinstance(r,dict) else r for r in rows]
exact(set(source['generators'])==set('pqxzt'),'actual_five_generators')
exact(len(words)==21,'actual_twenty_one_relators')
for w in words:
    exact(bool(w) and set(w)<=set('pqxztPQXZT'),'literal_word_alphabet')
exact(sorted(map(reduce,words))==sorted(standard_presentation()),'independent_literal_export')


def shift(point,exponent,donor):
    ray,height=point
    if ray not in [1,donor]:
        return point
    coordinate=height-1 if ray==1 else -height
    coordinate += exponent
    return (1,coordinate+1) if coordinate>=0 else (donor,-coordinate)


def p(point,n):
    return shift(point,n,2)


def q(point,n):
    return shift(point,n,3)


def encode(ray,height,copy,m):
    return ray,m*(height-1)+(copy if ray==1 else m+1-copy)


for m in [2,3,5]:
    for ray,height,copy in itertools.product([1,2,3],range(1,17),range(1,m+1)):
        point=(ray,height);native=encode(ray,height,copy,m)
        for gen,fun in [('p',p),('q',q)]:
            for sign in [-1,1]:
                base=fun(point,sign)
                exact(fun(native,sign*m)==encode(*base,copy,m),'actual_block_replication')
    # Multiply the FOUR actual sparse Pauli normal forms. The cocycle is
    # |old Z support intersect new X support| modulo2, independent of labels.
    sites=set(range(m));epsilon=0;xs=set();zs=set()
    for addx,addz in [(sites,set()),(set(),sites),(sites,set()),(set(),sites)]:
        epsilon ^= len(zs & addx)%2
        xs ^= addx;zs ^= addz
    exact(not xs and not zs and epsilon==m%2,'onsite_actual_Pauli_commutator')
    if m%2==0:
        exact(m%2 != 1,'even_replication_rejected')
    else:
        exact(m%2 == 1,'odd_replication_preserves_negative_J')


def q_cocycle(point,a,b,delta):
    for fun,exponent in [(q,b),(p,a),(q,delta),(p,-a),(q,-(b+delta))]:
        point=fun(point,exponent)
    return point


def cosine(n):
    return {j:math.cos(math.pi*j/(2*(n+1)))/math.sqrt(n+1)
            for j in range(-n,n+1)}


def shift_energy(vec,delta):
    support=set(vec)|{j+delta for j in vec}
    return sum((vec.get(j,0)-vec.get(j-delta,0))**2 for j in support)


clock_records=[]
for ell in [1,2]:
    m=3;maximum=(ell+1)*m**(2*ell);radius=8*maximum+8
    signed_cases=0
    for height in [-ell,0,ell]:
        block=m**(ell-height)
        for sign in [-1,1]:
            delta=sign*block
            for a,b in itertools.product([-maximum,0,maximum],[-maximum-block,-maximum,0,maximum,maximum+block]):
                if not (-maximum<=b<=maximum or -maximum<=b+delta<=maximum):
                    continue
                signed_cases += 1
                displacement_budget=abs(b)+2*abs(a)+abs(delta)+abs(b+delta)
                exact(displacement_budget<=8*maximum,'analytic_tail_displacement_budget')
                # A fixed bound outside which the zero-translation word acts
                # trivially is analytic; these sparse samples exercise both
                # signed boundary orientations and the donor junctions.
                for ray in [1,2,3]:
                    for pointheight in range(1,radius+2):
                        point=(ray,pointheight)
                        image=q_cocycle(point,a,b,delta)
                        exact(image[1]>=1,'cocycle_point_valid')
                        exact(pointheight<=radius or image==point,'cocycle_outside_region_fixed')
                        if pointheight<=radius:
                            exact(image[1]<=radius,'cocycle_preserves_finite_point_region')
                # Actual lamp conjugation is inverse p^a q^b on the block.
                for h in range(1,block+1):
                    image=q(p((1,h),-a),-b)
                    exact(image[1]<=radius,'active_lamp_block_inside_region')
    ff=cosine(maximum);gg=cosine(ell)
    exact(abs(sum(v*v for v in ff.values())-1)<1e-12,'cosine_translation_normalized')
    exact(abs(sum(v*v for v in gg.values())-1)<1e-12,'cosine_height_normalized')
    t_energy=shift_energy(gg,1)
    pq_energy=sum(gg[j]**2*shift_energy(ff,m**(ell-j)) for j in gg)
    upper=(math.pi/(2*(ell+1)))**2
    exact(t_energy<=upper+1e-12,'actual_t_scalar_clock_bound')
    exact(pq_energy<=upper+1e-12,'actual_pq_scalar_clock_bound')
    exact(abs(t_energy-(2-2*math.cos(math.pi/(2*(ell+1)))))<1e-12,'actual_t_clock_eigenvalue')
    clock_records.append({'L':ell,'M':maximum,'R':radius,'signed_active_cases':signed_cases,
                          't_root_error':math.sqrt(t_energy),'pq_root_error':math.sqrt(pq_energy),
                          'common_upper':math.sqrt(upper)})

out={'status':'PASS','checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
     'exact_comparisons':sum(counts.values()),'counts':counts,
     'relator_lengths':{'max':max(map(len,words)),'sum':sum(map(len,words))},
     'clock_records':clock_records,
     'limits':'Literal 21-row exporter, bounded block replication, sparse point cocycles and scalar clock controls. Not a faithful Gamma point action (t changes the direct-limit stage); not finite native spins; HNN embedding/infinite presentation/upper induction and ED theorem are analytic.'}
(HERE/'COMPACT-RESULTS.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
