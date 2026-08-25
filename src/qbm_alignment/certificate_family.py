from __future__ import annotations
import itertools, json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import networkx as nx
import numpy as np
N=16; PLANTED_WEIGHT=6; TARGET_WIDTHS=(3,4,5,6); INSTANCES_PER_WIDTH=5; GENERATION_SEED=20260824

def all_bits(n:int)->np.ndarray:
    x=np.arange(1<<n,dtype=np.uint64)[:,None]; s=np.arange(n-1,-1,-1,dtype=np.uint64)
    return ((x>>s)&1).astype(np.int8)

def formula_edges(clauses:Sequence[tuple[int,int,int]])->tuple[tuple[int,int],...]:
    e=set()
    for c in clauses:
        for a,b in itertools.combinations(c,2): e.add((a,b) if a<b else (b,a))
    return tuple(sorted(e))

def min_fill_order(n:int,edges:Sequence[tuple[int,int]])->tuple[int,...]:
    adj={i:set() for i in range(n)}
    for a,b in edges: adj[a].add(b);adj[b].add(a)
    rem=set(range(n)); order=[]
    while rem:
        opts=[]
        for v in rem:
            nb=sorted(adj[v]&rem); fill=sum(b not in adj[a] for i,a in enumerate(nb) for b in nb[i+1:])
            opts.append((fill,len(nb),v))
        _,_,v=min(opts); nb=list(adj[v]&rem)
        for i,a in enumerate(nb):
            for b in nb[i+1:]: adj[a].add(b);adj[b].add(a)
        rem.remove(v); order.append(v)
    return tuple(order)

def induced_width(n:int,edges:Sequence[tuple[int,int]],order:Sequence[int])->tuple[int,tuple[int,...]]:
    adj={i:set() for i in range(n)}
    for a,b in edges: adj[a].add(b);adj[b].add(a)
    rem=set(range(n)); width=0; deg=[]
    for v in order:
        nb=list(adj[v]&rem); width=max(width,len(nb));deg.append(len(nb))
        for i,a in enumerate(nb):
            for b in nb[i+1:]:adj[a].add(b);adj[b].add(a)
        rem.remove(v)
    return width,tuple(deg)

def exact_min_width_order(n:int,edges:Sequence[tuple[int,int]])->tuple[int,...]:
    adj=[0]*n
    for a,b in edges:adj[a]|=1<<b;adj[b]|=1<<a
    full=(1<<n)-1; size=1<<n
    bw=np.full(size,n+1,dtype=np.int16); bc=np.full(size,np.iinfo(np.int64).max,dtype=np.int64)
    pv=np.full(size,-1,dtype=np.int16); ps=np.full(size,-1,dtype=np.int32); bw[0]=0;bc[0]=0
    cache={}
    def deg(S,v):
        k=(S,v)
        if k in cache:return cache[k]
        rem=full^S; reach=adj[v]&S; front=reach
        while front:
            bit=front&-front;front-=bit;u=bit.bit_length()-1;new=adj[u]&S&~reach;reach|=new;front|=new
        nb=adj[v]&rem;r=reach
        while r:
            bit=r&-r;r-=bit;u=bit.bit_length()-1;nb|=adj[u]&rem
        nb&=~(1<<v);cache[k]=nb.bit_count();return cache[k]
    for S in range(size):
        if bw[S]>n:continue
        r=full^S
        while r:
            bit=r&-r;r-=bit;v=bit.bit_length()-1;d=deg(S,v);T=S|bit;w=max(int(bw[S]),d);c=int(bc[S])+(1<<(d+1))
            if w<bw[T] or (w==bw[T] and c<bc[T]):bw[T]=w;bc[T]=c;pv[T]=v;ps[T]=S
    rev=[];S=full
    while S:
        v=int(pv[S]);rev.append(v);S=int(ps[S])
    return tuple(reversed(rev))

class Space:
    def __init__(self):
        self.n=N;self.planted=tuple([1]*PLANTED_WEIGHT+[0]*(N-PLANTED_WEIGHT));self.pidx=int(''.join(map(str,self.planted)),2)
        self.bits=all_bits(N);zeros=range(PLANTED_WEIGHT,N)
        self.candidates=tuple((t,a,b) for t in range(PLANTED_WEIGHT) for a,b in itertools.combinations(zeros,2))
        self.masks=np.asarray([self.bits[:,c].sum(1)==1 for c in self.candidates],bool)
    def sols(self,inds):
        if not inds:return np.arange(1<<N)
        return np.flatnonzero(np.logical_and.reduce(self.masks[list(inds)],axis=0))
    def unique(self,inds):
        s=self.sols(inds);return len(s)==1 and int(s[0])==self.pidx

def greedy_core(sp:Space,rng:random.Random):
    order=list(range(len(sp.candidates)));rng.shuffle(order);chosen=[];valid=np.ones(1<<N,bool)
    for i in order:
        chosen.append(i);valid&=sp.masks[i]
        if valid.sum()==1:break
    if valid.sum()!=1 or not valid[sp.pidx]:return None
    rest=[i for i in order if i not in chosen];chosen+=rest[:rng.randint(0,min(48,len(rest)))]
    changed=True
    while changed:
        changed=False;o=chosen[:];rng.shuffle(o)
        for i in o:
            tr=[j for j in chosen if j!=i]
            if sp.unique(tr):chosen=tr;changed=True
    return tuple(sorted(chosen))

def inc_graph(clauses,planted):
    g=nx.Graph()
    for v,b in enumerate(planted):g.add_node(('v',v),kind=f'v{b}')
    for k,c in enumerate(clauses):
        g.add_node(('c',k),kind='c')
        for v in c:g.add_edge(('c',k),('v',v))
    return g

@dataclass(frozen=True)
class Instance:
    instance_id:str;n:int;planted:tuple[int,...];clauses:tuple[tuple[int,int,int],...];width:int;order:tuple[int,...];witnesses:tuple[tuple[int,...],...]
    @property
    def edges(self):return formula_edges(self.clauses)

def generate_family(out:Path)->list[Instance]:
    out.mkdir(parents=True,exist_ok=True);mf=out/'manifest.json'
    if mf.exists():
        data=json.loads(mf.read_text());return [Instance(x['instance_id'],x['n'],tuple(x['planted']),tuple(tuple(c) for c in x['clauses']),x['width'],tuple(x['order']),tuple(tuple(w) for w in x['witnesses'])) for x in data]
    rng=random.Random(GENERATION_SEED);sp=Space();acc={w:[] for w in TARGET_WIDTHS};graphs={w:[] for w in TARGET_WIDTHS};seen=set();trial=0
    match=nx.algorithms.isomorphism.categorical_node_match('kind',None)
    while any(len(acc[w])<INSTANCES_PER_WIDTH for w in TARGET_WIDTHS):
        trial+=1;core=greedy_core(sp,rng)
        if core is None or core in seen:continue
        seen.add(core);clauses=tuple(sp.candidates[i] for i in core);edges=formula_edges(clauses);ho=min_fill_order(N,edges);uw,_=induced_width(N,edges,ho)
        if uw not in TARGET_WIDTHS or len(acc[uw])>=INSTANCES_PER_WIDTH:continue
        eo=exact_min_width_order(N,edges);w,_=induced_width(N,edges,eo)
        if w!=uw or len(acc[w])>=INSTANCES_PER_WIDTH:continue
        g=inc_graph(clauses,sp.planted)
        if any(nx.is_isomorphic(g,q,node_match=match) for q in graphs[w]):continue
        witnesses=[]
        for rem in core:
            alt=[int(x) for x in sp.sols([i for i in core if i!=rem]) if int(x)!=sp.pidx]
            if not alt:raise RuntimeError('nonessential clause')
            witnesses.append(tuple(int(v) for v in sp.bits[alt[0]]))
        iid=f'ct_w{w}_i{len(acc[w])+1}';inst=Instance(iid,N,sp.planted,clauses,w,eo,tuple(witnesses));acc[w].append(inst);graphs[w].append(g)
        print('accepted',iid,'m',len(clauses),'edges',len(edges),'trial',trial,flush=True)
    fam=[x for w in TARGET_WIDTHS for x in acc[w]]
    mf.write_text(json.dumps([{'instance_id':x.instance_id,'n':x.n,'planted':x.planted,'clauses':x.clauses,'width':x.width,'order':x.order,'witnesses':x.witnesses} for x in fam],indent=2)+'\n')
    for x in fam:
        lines=[f'{x.n} {len(x.clauses)} {sum(x.planted)}',' '.join(map(str,x.planted)),*[' '.join(str(v+1) for v in c) for c in x.clauses]];(out/f'{x.instance_id}.txt').write_text('\n'.join(lines)+'\n')
    return fam

def state_features(n,edges):
    bits=all_bits(n);z=(1-2*bits).astype(float);pair=np.column_stack([z[:,i]*z[:,j] for i,j in edges]) if edges else np.empty((len(bits),0));return bits,np.column_stack([z,pair])

def coefficients(n,clauses,edges):
    h=np.zeros(n);J={}
    for c in clauses:
        for i in c:h[i]-=.5
        for a,b in itertools.combinations(c,2):e=(a,b) if a<b else (b,a);J[e]=J.get(e,0)+.5
    return np.r_[h,[J[e] for e in edges]]

def costs(bits,clauses):
    idx=np.asarray(clauses);occ=bits[:,idx].sum(2);return (((occ-1)**2)-1).sum(1).astype(float)

def evaluate(theta,F,C,want_fisher=False):
    s=F@theta;lw=-s;lw-=lw.max();w=np.exp(np.clip(lw,-745,0));p=w/w.sum();E=float(p@C);mu=p@F;mcf=(p*C)@F;g=-(mcf-E*mu)
    I=None
    if want_fisher:
        X=F-mu;I=X.T@(X*p[:,None])
    return E,g,p,I

def eff_condition(I,cut=1e-12):
    vals=np.linalg.eigvalsh((I+I.T)/2);mx=max(float(vals[-1]),0.)
    keep=vals[vals>cut*mx]
    if mx<=0 or len(keep)==0:return math.inf,0,0.,mx
    return float(mx/keep[0]),len(keep),float(keep[0]),mx

def local_min(bits,C,index):
    n=bits.shape[1];x=bits[index];e=C[index]
    for i in range(n):
        y=x.copy();y[i]^=1;j=int(''.join(map(str,y.tolist())),2)
        if C[j]<e-1e-12:return False
    return True
