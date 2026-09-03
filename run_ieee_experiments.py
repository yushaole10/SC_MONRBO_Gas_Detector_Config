from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json, os, time, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gas_optimization_experiment as g

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiment'; FIG=OUT/'figures'; OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEEDS=[11,19,23,31,37,43,47,53,59,61,67,71,73,79,83,89,97,101,107,109]
SENS_SEEDS=SEEDS[:10]
ALGS=['NSGA-II','MONRBO','SC-MONRBO','Abl-no-safety','Abl-static-CD','Abl-no-TAO']

def execute(task):
    kind,name,seed,pop,gen=task
    t=time.perf_counter()
    if name=='NSGA-II': X,F,V=g.nsga2(seed,pop=pop,gen=gen,safe=False)
    elif name in ('MONRBO','Abl-no-safety'): X,F,V=g.monrbo(seed,pop=pop,gen=gen,safe=False,use_decd=True,use_tao=True)
    elif name=='SC-MONRBO': X,F,V=g.monrbo(seed,pop=pop,gen=gen,safe=True,use_decd=True,use_tao=True)
    elif name=='Abl-static-CD': X,F,V=g.monrbo(seed,pop=pop,gen=gen,safe=True,use_decd=False,use_tao=True)
    elif name=='Abl-no-TAO': X,F,V=g.monrbo(seed,pop=pop,gen=gen,safe=True,use_decd=True,use_tao=False)
    else: raise ValueError(name)
    rt=time.perf_counter()-t; bi=g.best_compromise(X,F,V); _,_,d=g.metrics(X[bi])
    row={'kind':kind,'algorithm':name,'seed':seed,'population':pop,'generations':gen,'runtime_s':rt,
         'feasible_fraction':float(np.mean(V<=1e-12)),'rmse':d['rmse'],'fnr':d['fnr'],'fpr':d['fpr'],
         'delay_s':d['delay_s'],'energy':d['energy'],'risk':F[bi,1]}
    return row,X,F,V

def holm(pvals):
    p=np.asarray(pvals,float); order=np.argsort(p); out=np.empty_like(p); running=0
    for rank,idx in enumerate(order):
        val=min(1,(len(p)-rank)*p[idx]); running=max(running,val); out[idx]=running
    return out

def a12(a,b,higher=True):
    a=np.asarray(a);b=np.asarray(b); score=0
    for x in a:
        score+=np.sum(x>b) if higher else np.sum(x<b);score+=.5*np.sum(x==b)
    return score/(len(a)*len(b))

def average_ranks(x):
    x=np.asarray(x,float);order=np.argsort(x);r=np.empty(len(x),float);i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        r[order[i:j]]=(i+1+j)/2;i=j
    return r

def wilcoxon_normal(a,b):
    d=np.asarray(a,float)-np.asarray(b,float);d=d[np.abs(d)>1e-15]
    n=len(d)
    if n==0:return 0.0,1.0
    ranks=average_ranks(np.abs(d));wplus=float(ranks[d>0].sum());wminus=float(ranks[d<0].sum());w=min(wplus,wminus)
    mean=n*(n+1)/4;_,counts=np.unique(np.abs(d),return_counts=True)
    var=n*(n+1)*(2*n+1)/24-sum(c**3-c for c in counts)/48
    if var<=0:return w,1.0
    z=(abs(wplus-mean)-.5)/math.sqrt(var);p=math.erfc(max(0,z)/math.sqrt(2))
    return w,min(1.0,p)

def make_stats(df):
    tests=[]
    directions={'hypervolume_mc':True,'feasible_fraction':True,'rmse':False,'risk':False,'energy':False,'delay_s':False}
    sc=df[df.algorithm=='SC-MONRBO'].set_index('seed')
    for metric,higher in directions.items():
        for comp in ['NSGA-II','MONRBO']:
            bb=df[df.algorithm==comp].set_index('seed'); common=sc.index.intersection(bb.index)
            a=sc.loc[common,metric].to_numpy();b=bb.loc[common,metric].to_numpy()
            stat,p=wilcoxon_normal(a,b)
            tests.append({'metric':metric,'comparison':f'SC-MONRBO vs {comp}','n_pairs':len(common),
                          'wilcoxon_W':stat,'p_raw':p,'A12_preferred_direction':a12(a,b,higher),
                          'preferred':'higher' if higher else 'lower'})
    adj=holm([x['p_raw'] for x in tests])
    for r,p in zip(tests,adj):r['p_holm']=p
    return pd.DataFrame(tests)

def summarize(df,group):
    metrics=['runtime_s','feasible_fraction','rmse','fnr','fpr','delay_s','energy','risk','hypervolume_mc','spacing']
    return df.groupby(group)[metrics].agg(['mean','std','median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)]).round(6)

def plots(main,archives,sens):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8})
    colors={'NSGA-II':'#687883','MONRBO':'#e39c37','SC-MONRBO':'#174a7e'}
    fig=plt.figure(figsize=(8.8,6.2));ax=fig.add_subplot(111,projection='3d')
    for name in colors:
        sub=main[main.algorithm==name];med=sub.hypervolume_mc.median();seed=int(sub.iloc[np.argmin(np.abs(sub.hypervolume_mc-med))].seed)
        F=archives[(name,seed)][1];ax.scatter(F[:,0],F[:,1],F[:,2],s=20,alpha=.72,label=f'{name} (seed {seed})',color=colors[name])
    ax.set_xlabel('Worst-case NRMSE');ax.set_ylabel('Safety-alarm loss');ax.set_zlabel('Energy proxy');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(FIG/'pareto_comparison.png',dpi=500,bbox_inches='tight');plt.close(fig)
    order=ALGS; metrics=['hypervolume_mc','feasible_fraction','rmse','fnr','fpr','delay_s']
    fig,axs=plt.subplots(2,3,figsize=(10.6,6.5))
    for ax,m in zip(axs.flat,metrics):
        vals=[main[main.algorithm==a][m] for a in order];med=[v.median() for v in vals];lo=[np.quantile(v,.25) for v in vals];hi=[np.quantile(v,.75) for v in vals]
        ax.barh(order,med,color=['#687883','#e39c37','#174a7e','#c84630','#26a6a1','#8e6bbd'])
        ax.errorbar(med,range(len(order)),xerr=[np.array(med)-lo,hi-np.array(med)],fmt='none',ecolor='black',capsize=2,lw=.8)
        ax.set_xlabel(m.replace('_',' '));ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'algorithm_ablation.png',dpi=500,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,4.1))
    for ax,factor in zip(axs,['population','generations']):
        ss=sens[sens.factor==factor];grp=ss.groupby('value')
        x=np.array(sorted(ss.value.unique()));fmed=np.array([grp.get_group(v).feasible_fraction.median() for v in x]);
        elo=np.array([grp.get_group(v).energy.quantile(.25) for v in x]);ehi=np.array([grp.get_group(v).energy.quantile(.75) for v in x]);emed=np.array([grp.get_group(v).energy.median() for v in x])
        ax.plot(x,fmed,'o-',color='#174a7e',label='feasible fraction');ax.set_ylabel('Feasible fraction',color='#174a7e');ax.set_ylim(-.03,1.05);ax.set_xlabel(factor);ax.grid(alpha=.2)
        a2=ax.twinx();a2.plot(x,emed,'s--',color='#e39c37',label='energy proxy');a2.fill_between(x,elo,ehi,color='#e39c37',alpha=.18);a2.set_ylabel('Energy proxy',color='#e39c37')
    axs[0].text(.5,-.22,'(a) Population size',transform=axs[0].transAxes,ha='center');axs[1].text(.5,-.22,'(b) Generation budget',transform=axs[1].transAxes,ha='center')
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(FIG/'sensitivity.png',dpi=500,bbox_inches='tight');plt.close(fig)

def main():
    tasks=[('main',a,s,24,35) for a in ALGS for s in SEEDS]
    archives={};rows=[];workers=min(8,max(2,(os.cpu_count() or 4)-1))
    print(f'running {len(tasks)} main runs with {workers} workers',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(execute,t) for t in tasks]
        for n,f in enumerate(as_completed(futs),1):
            row,X,F,V=f.result();rows.append(row);archives[(row['algorithm'],row['seed'])]=(X,F,V)
            if n%10==0:print(f'main {n}/{len(tasks)}',flush=True)
    main_df=pd.DataFrame(rows).sort_values(['algorithm','seed']).reset_index(drop=True)
    allF=np.vstack([v[1] for v in archives.values()])
    for i,r in main_df.iterrows():
        F=archives[(r.algorithm,int(r.seed))][1];main_df.loc[i,'hypervolume_mc']=g.hv_mc(F,allF,int(r.seed),n=40000);main_df.loc[i,'spacing']=g.spacing(g.norm_metrics(F,allF))
    main_df.to_csv(OUT/'results_20seeds.csv',index=False);main_df.to_csv(OUT/'results.csv',index=False)
    summarize(main_df,'algorithm').to_csv(OUT/'summary_20seeds.csv');make_stats(main_df).to_csv(OUT/'statistical_tests.csv',index=False)
    # representative SC archive nearest median HV
    sc=main_df[main_df.algorithm=='SC-MONRBO'];med=sc.hypervolume_mc.median();seed=int(sc.iloc[np.argmin(np.abs(sc.hypervolume_mc-med))].seed)
    X,F,V=archives[('SC-MONRBO',seed)];np.savetxt(OUT/'pareto_archive.csv',np.c_[X,F,V],delimiter=',',header=','.join([f'x{i+1}' for i in range(g.D)]+['rmse','risk','energy','violation']),comments='')
    # firmware baseline unchanged but regenerated
    z=np.r_[[-5,0],np.ones(10),1,1];o,v,d=g.metrics(z,firmware=True);pd.DataFrame([{'method':'firmware fixed threshold',**d,'risk':o[1],'violation':v}]).to_csv(OUT/'firmware_baseline.csv',index=False)
    sens_cases=[('population',16,16,35),('population',24,24,35),('population',32,32,35),('generations',20,24,20),('generations',35,24,35),('generations',50,24,50)]
    stasks=[]
    for factor,value,pop,gen in sens_cases:
        for seed0 in SENS_SEEDS:stasks.append((factor,f'sens-{factor}-{value}',seed0,pop,gen))
    print(f'running {len(stasks)} sensitivity runs',flush=True);srows=[]
    def sens_task_tuple(t):return t
    # execute SC-MONRBO under a temporary name handled locally through direct submissions
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futmap={ex.submit(execute,('sensitivity','SC-MONRBO',seed,pop,gen)):(factor,value) for factor,_,seed,pop,gen in stasks for value in [int(_.split('-')[-1])]}
        for n,f in enumerate(as_completed(futmap),1):
            factor,value=futmap[f];row,_,_,_=f.result();row['factor']=factor;row['value']=value;srows.append(row)
            if n%10==0:print(f'sensitivity {n}/{len(stasks)}',flush=True)
    sens=pd.DataFrame(srows).sort_values(['factor','value','seed']);sens.to_csv(OUT/'sensitivity_results_10seeds.csv',index=False);sens.to_csv(OUT/'sensitivity.csv',index=False)
    sens.groupby(['factor','value'])[['feasible_fraction','rmse','fnr','fpr','delay_s','energy']].agg(['mean','std','median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)]).to_csv(OUT/'sensitivity_summary_10seeds.csv')
    plots(main_df,archives,sens)
    config={'seed_data':20260814,'main_seeds':SEEDS,'sensitivity_seeds':SENS_SEEDS,'population':24,'generations':35,'objectives':['worst-case NRMSE','safety-alarm loss','dimensionless energy proxy'],'constraints':{'FNR_max':.10,'FPR_max':.10,'delay_max_s':18},'optimization_acquisition_model':'ideal synchronized digital vectors','code_guided_stress_model':'11-command, 1-s inter-query schedule with zero-order-held channels; post hoc only','data_status':'synthetic digital calibration chamber; not physical validation','physical_power_status':'not measured'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('complete',flush=True)

if __name__=='__main__':
    import multiprocessing as mp
    mp.freeze_support();main()
