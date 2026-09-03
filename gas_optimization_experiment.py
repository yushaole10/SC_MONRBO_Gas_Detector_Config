from pathlib import Path
import json, math, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiment';OUT.mkdir(exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
GASES=['O2-def','H2S','CO','CO2','NH3','NOx','CH4','HCN','SO2','HCHO']
D=14
LB=np.array([-5,.0]+[.72]*10+[1,1.],float)
UB=np.array([1,.98]+[1.28]*10+[5,6.],float)

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def make_data(seed=20260814):
    rng=np.random.default_rng(seed)
    S=np.eye(10)
    pairs={(1,2):.18,(2,1):.11,(4,8):.22,(8,4):.14,(5,8):.17,(8,5):.10,(7,1):.12,(9,4):.16,(6,2):.08}
    for (i,j),v in pairs.items():S[i,j]=v
    S+=rng.uniform(0,.025,(10,10))*(1-np.eye(10))
    env=rng.normal(0,.045,(10,2));drift=rng.normal(0,.025,10)
    n=5000;y=rng.beta(1.2,3.0,(n,10))*2.5
    hot=rng.random(n)<.28;y[hot,rng.integers(0,10,hot.sum())]+=rng.uniform(.8,1.8,hot.sum());y=np.clip(y,0,3)
    e=rng.uniform(-1,1,(n,2));age=rng.uniform(0,1,(n,1))
    x=y@S.T+e@env.T+age*drift+rng.normal(0,.035,(n,10))
    X=np.c_[x,e];
    # time-series incidents at 1 s resolution
    ns,nt=36,120;Y=np.zeros((ns,nt,10));E=np.zeros((ns,nt,2));A=np.zeros((ns,nt,1))
    t=np.arange(nt)
    for s in range(ns):
        Y[s]=rng.uniform(0.03,.28,(nt,10));
        E[s,:,0]=.25*np.sin(t/22+rng.uniform(0,6))+rng.normal(0,.04,nt)
        E[s,:,1]=.35*np.sin(t/31+rng.uniform(0,6))+rng.normal(0,.05,nt)
        A[s,:,0]=np.linspace(0,.8,nt)
        for _ in range(rng.integers(1,4)):
            g=rng.integers(0,10);on=rng.integers(12,75);amp=rng.uniform(1.05,2.6);rise=rng.uniform(3,12)
            Y[s,:,g]+=amp*sigmoid((t-on)/rise)
    base=Y@S.T+E@env.T+A*drift+rng.normal(0,.04,(ns,nt,10))
    scenarios=[]
    for name,hm,dm,nm in [('nominal',1,1,1),('humid',1.9,1.1,1.25),('drift',1.1,2.3,1.35)]:
        xx=Y@S.T+E@(env*hm).T+A*(drift*dm)+rng.normal(0,.04*nm,(ns,nt,10))
        scenarios.append((name,np.concatenate([xx,E],axis=2),Y))
    return X,y,scenarios,S,env,drift

XTR,YTR,SCENARIOS,S_TRUE,ENV_TRUE,DRIFT_TRUE=make_data()
XT=XTR.T@XTR;XTY=XTR.T@YTR
W_CACHE={}
def ridge(loglam):
    key=round(float(loglam),3)
    if key not in W_CACHE:
        lam=10**key;W_CACHE[key]=np.linalg.solve(XT+lam*np.eye(XT.shape[0]),XTY)
    return W_CACHE[key]

def decode(z):
    z=np.clip(z,LB,UB);return z[0],z[1],z[2:12],int(np.rint(z[12])),int(np.rint(z[13]))

def metrics(z,robust=True,firmware=False):
    loglam,alpha,thr,k,tau=decode(z);vals=[]
    W=None if firmware else ridge(loglam)
    use=SCENARIOS if robust else SCENARIOS[:1]
    for _,X,Y in use:
        Xs=X[:,::tau];Ys=Y[:,::tau]
        est=Xs[...,:10] if firmware else np.maximum(0,Xs@W)
        if alpha>0:
            for j in range(1,est.shape[1]):est[:,j]=alpha*est[:,j-1]+(1-alpha)*est[:,j]
        truth=Ys>=1.;raw=est>=thr
        alarm=np.zeros_like(raw)
        run=np.zeros((raw.shape[0],raw.shape[2]),int)
        for j in range(raw.shape[1]):
            run=np.where(raw[:,j],run+1,0);alarm[:,j]=run>=k
        fn=np.logical_and(truth,~alarm).sum()/max(1,truth.sum())
        fp=np.logical_and(~truth,alarm).sum()/max(1,(~truth).sum())
        delays=[]
        for s in range(truth.shape[0]):
            for g in range(10):
                idx=np.flatnonzero(truth[s,:,g])
                if idx.size:
                    a=np.flatnonzero(alarm[s,idx[0]:,g]);delays.append((a[0] if a.size else truth.shape[1]-idx[0])*tau)
        delay=np.mean(delays) if delays else 0
        rmse=np.sqrt(np.mean((est-Ys)**2))
        vals.append((rmse,fn,fp,delay))
    a=np.array(vals);worst=a.max(axis=0)
    risk=.65*worst[1]+.20*worst[2]+.15*min(worst[3]/30,1)
    energy=(1/tau)*(1+.12*(1-alpha)+.03*k)
    obj=np.array([worst[0],risk,energy])
    violation=max(0,worst[1]-.10)+max(0,worst[2]-.10)+max(0,worst[3]-18)/30
    return obj,violation,{'rmse':worst[0],'fnr':worst[1],'fpr':worst[2],'delay_s':worst[3],'energy':energy}

def dominates(a,b,va=0,vb=0,safe=True):
    if safe:
        if va<=1e-12<vb:return True
        if vb<=1e-12<va:return False
        if va>1e-12 and vb>1e-12:return va<vb
    return np.all(a<=b) and np.any(a<b)

def fronts(F,V,safe=True):
    n=len(F);S=[[] for _ in range(n)];cnt=np.zeros(n,int);fr=[[]]
    for i in range(n):
        for j in range(n):
            if i==j:continue
            if dominates(F[i],F[j],V[i],V[j],safe):S[i].append(j)
            elif dominates(F[j],F[i],V[j],V[i],safe):cnt[i]+=1
        if cnt[i]==0:fr[0].append(i)
    k=0
    while fr[k]:
        q=[]
        for i in fr[k]:
            for j in S[i]:cnt[j]-=1;q.append(j) if cnt[j]==0 else None
        k+=1;fr.append(q)
    return fr[:-1]

def crowd(F,idx):
    n=len(idx);d=np.zeros(n)
    if n<=2:return np.full(n,np.inf)
    A=F[idx]
    for m in range(F.shape[1]):
        o=np.argsort(A[:,m]);d[o[[0,-1]]]=np.inf;span=A[o[-1],m]-A[o[0],m]
        if span>1e-12:d[o[1:-1]]+=(A[o[2:],m]-A[o[:-2],m])/span
    return d

def decd(X,F,V,maxn,safe=True):
    keep=[]
    for fr in fronts(F,V,safe):
        if len(keep)+len(fr)<=maxn:keep+=fr;continue
        pool=list(fr)
        while len(keep)+len(pool)>maxn:
            cd=crowd(F,pool);pool.pop(int(np.argmin(cd)))
        keep+=pool;break
    return X[keep],F[keep],V[keep]

def evaluate(X,robust=True):
    res=[metrics(x,robust)[:2] for x in X]
    return np.array([r[0] for r in res]),np.array([r[1] for r in res])

def nsga2(seed,pop=24,gen=35,safe=True):
    rng=np.random.default_rng(seed);X=rng.uniform(LB,UB,(pop,D));F,V=evaluate(X)
    for it in range(gen):
        fr=fronts(F,V,safe);rank=np.empty(pop,int);cdall=np.zeros(pop)
        for k,ix in enumerate(fr):rank[ix]=k;cdall[ix]=crowd(F,ix)
        parents=[]
        for _ in range(pop):
            a,b=rng.integers(0,pop,2);parents.append(a if (rank[a]<rank[b] or (rank[a]==rank[b] and cdall[a]>cdall[b])) else b)
        P=X[parents];C=np.empty_like(P)
        for i in range(0,pop,2):
            a,b=P[i],P[(i+1)%pop];u=rng.random(D);beta=np.where(u<=.5,(2*u)**(1/3),(1/(2*(1-u)))**(1/3))
            C[i]=.5*((1+beta)*a+(1-beta)*b);C[(i+1)%pop]=.5*((1-beta)*a+(1+beta)*b)
        mut=rng.random(C.shape)<1/D;C+=mut*rng.normal(0,.08,(pop,D))*(UB-LB);C=np.clip(C,LB,UB)
        FC,VC=evaluate(C);X,F,V=decd(np.r_[X,C],np.r_[F,FC],np.r_[V,VC],pop,safe)
    return decd(X,F,V,pop,safe)

def monrbo(seed,pop=24,gen=35,safe=True,use_decd=True,use_tao=True):
    rng=np.random.default_rng(seed);X=rng.uniform(LB,UB,(pop,D));F,V=evaluate(X)
    for it in range(1,gen+1):
        nd=fronts(F,V,safe)[0];A=X[nd];AF=F[nd];AV=V[nd];cd=crowd(AF,list(range(len(A))))
        if np.any(np.isfinite(cd)):
            finite=np.where(np.isfinite(cd),cd,2*np.max(cd[np.isfinite(cd)]))
        else:
            finite=np.ones_like(cd)
        prob=np.nan_to_num(finite,nan=1.0,posinf=2.0)+1e-6;prob/=prob.sum()
        worst=X[np.argmax(F.sum(axis=1)+10*V)];delta=(1-2*it/gen)**5
        C=[]
        for n,x in enumerate(X):
            xb=A[rng.choice(len(A),p=prob)];r1,r2=rng.choice(pop,2,replace=False);dx=rng.random(D)*np.abs(xb-x)/max(it,1)
            den=2*(worst+xb-2*x);nrsr=rng.normal(size=D)*(worst-xb)*dx/(den+np.sign(den)*1e-9+(den==0)*1e-9)
            rho=rng.random()*(xb-x)+rng.random()*(X[r1]-X[r2]);x1=x-nrsr+rho;x2=xb-nrsr+rho;x3=x-delta*(x2-x1);q=rng.random();new=q*(q*x1+(1-q)*x2)+(1-q)*x3
            if use_tao and rng.random()<.45:new+=rng.uniform(-1,1,D)*(xb-rng.random()*x)+delta*rng.normal(0,.03,D)*(UB-LB)
            C.append(np.clip(new,LB,UB))
        C=np.array(C);FC,VC=evaluate(C);XX=np.r_[X,C];FF=np.r_[F,FC];VV=np.r_[V,VC]
        if use_decd:X,F,V=decd(XX,FF,VV,pop,safe)
        else:
            keep=[]
            for fr in fronts(FF,VV,safe):
                if len(keep)+len(fr)<=pop:keep+=fr
                else:
                    cd=crowd(FF,fr);keep+=list(np.array(fr)[np.argsort(-cd)[:pop-len(keep)]]);break
            X,F,V=XX[keep],FF[keep],VV[keep]
    return decd(X,F,V,pop,safe)

def norm_metrics(F,ref):
    lo=ref.min(0);hi=ref.max(0);return (F-lo)/(hi-lo+1e-12)
def hv_mc(F,refall,seed=1,n=40000):
    N=norm_metrics(F,refall);rng=np.random.default_rng(seed);P=rng.random((n,3))*1.1
    return np.mean(np.any(np.all(N[:,None,:]<=P[None,:,:],axis=2),axis=0))*1.1**3
def spacing(F):
    if len(F)<3:return np.nan
    Dm=np.abs(F[:,None]-F[None]).sum(2);Dm[Dm==0]=np.inf;d=Dm.min(1);return np.std(d)

def best_compromise(X,F,V):
    feas=V<=1e-12;idx=np.where(feas)[0] if feas.any() else np.arange(len(X));A=F[idx];N=(A-A.min(0))/(A.max(0)-A.min(0)+1e-12);return idx[np.argmin(N.sum(1))]

def run_all():
    algs={
      'NSGA-II':lambda s:nsga2(s,safe=False),
      'MONRBO':lambda s:monrbo(s,safe=False,use_decd=True,use_tao=True),
      'SC-MONRBO':lambda s:monrbo(s,safe=True,use_decd=True,use_tao=True),
      'Abl-no-safety':lambda s:monrbo(s,safe=False,use_decd=True,use_tao=True),
      'Abl-static-CD':lambda s:monrbo(s,safe=True,use_decd=False,use_tao=True),
      'Abl-no-TAO':lambda s:monrbo(s,safe=True,use_decd=True,use_tao=False),
    }
    # MONRBO and SC share operators; SC name denotes the explicit feasibility rule used in reporting.
    seeds=[11,37,71];runs=[];archives={}
    for name,fn in algs.items():
        for seed in seeds:
            t=time.perf_counter();X,F,V=fn(seed);rt=time.perf_counter()-t;archives[(name,seed)]=(X,F,V)
            bi=best_compromise(X,F,V);_,_,detail=metrics(X[bi]);runs.append({'algorithm':name,'seed':seed,'runtime_s':rt,'feasible_fraction':float(np.mean(V<=1e-12)),'best_rmse':F[bi,0],'best_risk':F[bi,1],'best_energy':F[bi,2],**detail})
            print(name,seed,round(rt,2),round(runs[-1]['feasible_fraction'],2))
    allF=np.vstack([v[1] for v in archives.values()]);
    for row in runs:
        F=archives[(row['algorithm'],row['seed'])][1];row['hypervolume_mc']=hv_mc(F,allF,row['seed']);row['spacing']=spacing(norm_metrics(F,allF))
    df=pd.DataFrame(runs);df.to_csv(OUT/'results.csv',index=False)
    # firmware baseline
    z=np.r_[[-5,0],np.ones(10),1,1];o,v,d=metrics(z,firmware=True);pd.DataFrame([{'method':'firmware fixed threshold',**d,'risk':o[1],'violation':v}]).to_csv(OUT/'firmware_baseline.csv',index=False)
    # representative SC archive
    X,F,V=archives[('SC-MONRBO',37)];np.savetxt(OUT/'pareto_archive.csv',np.c_[X,F,V],delimiter=',',header=','.join([f'x{i+1}' for i in range(D)]+['rmse','risk','energy','violation']),comments='')
    summary=df.groupby('algorithm').agg(['mean','std']).round(4);summary.to_csv(OUT/'summary.csv')
    # figures
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    fig=plt.figure(figsize=(9,6));ax=fig.add_subplot(111,projection='3d')
    colors={'NSGA-II':'#6b7780','MONRBO':'#e39c37','SC-MONRBO':'#174a7e'}
    for name in colors:
        F=np.vstack([archives[(name,s)][1] for s in seeds]);ax.scatter(F[:,0],F[:,1],F[:,2],s=12,alpha=.55,label=name,color=colors[name])
    ax.set_xlabel('Worst-case NRMSE');ax.set_ylabel('Safety-alarm loss');ax.set_zlabel('Energy proxy');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(FIG/'pareto_comparison.png',dpi=400);plt.close(fig)
    cols=['hypervolume_mc','feasible_fraction','best_rmse','fnr','fpr','delay_s','runtime_s'];m=df.groupby('algorithm')[cols].mean().loc[list(algs)]
    fig,axs=plt.subplots(2,3,figsize=(10,6));plotcols=['hypervolume_mc','feasible_fraction','best_rmse','fnr','fpr','delay_s']
    for ax,c in zip(axs.flat,plotcols):ax.barh(m.index,m[c],color=['#6b7780','#e39c37','#174a7e','#c84630','#26a6a1','#8e6bbd']);ax.set_xlabel(c.replace('_',' '));ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'algorithm_ablation.png',dpi=400);plt.close(fig)
    # cross sensitivity heat map
    fig,ax=plt.subplots(figsize=(7,5.7));im=ax.imshow(S_TRUE,cmap='viridis',vmin=0,vmax=1.05);ax.set_xticks(range(10),GASES,rotation=45,ha='right');ax.set_yticks(range(10),GASES);ax.set_xlabel('Interfering gas');ax.set_ylabel('Sensor channel');fig.colorbar(im,ax=ax,label='Normalized sensitivity');fig.tight_layout();fig.savefig(FIG/'synthetic_cross_sensitivity.png',dpi=400);plt.close(fig)
    config={'seed_data':20260814,'seeds_algorithm':seeds,'population':24,'generations':35,'objectives':['worst-case NRMSE','safety-alarm loss','energy proxy'],'constraints':{'FNR_max':.10,'FPR_max':.10,'delay_max_s':18},'data_status':'synthetic digital calibration chamber; not field validation'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return df,summary

if __name__=='__main__':run_all()
