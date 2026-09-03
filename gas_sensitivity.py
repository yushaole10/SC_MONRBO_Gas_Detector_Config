from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gas_optimization_experiment as g

OUT=Path(__file__).resolve().parent/'experiment'; rows=[]
cases=[('population',16,16,35),('population',24,24,35),('population',32,32,35),
       ('generations',20,24,20),('generations',35,24,35),('generations',50,24,50)]
for factor,value,pop,gen in cases:
    X,F,V=g.monrbo(101,pop=pop,gen=gen,safe=True,use_decd=True,use_tao=True)
    i=g.best_compromise(X,F,V);_,_,d=g.metrics(X[i])
    rows.append({'factor':factor,'value':value,'population':pop,'generations':gen,
                 'feasible_fraction':np.mean(V<=1e-12),'rmse':d['rmse'],'fnr':d['fnr'],
                 'fpr':d['fpr'],'delay_s':d['delay_s'],'energy':d['energy']})
    print(rows[-1])
df=pd.DataFrame(rows);df.to_csv(OUT/'sensitivity.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(9,3.8))
for ax,(factor,sub) in zip(axs,df.groupby('factor',sort=False)):
    ax.plot(sub.value,sub.feasible_fraction,'o-',label='feasible fraction',color='#174a7e')
    ax2=ax.twinx();ax2.plot(sub.value,sub.energy,'s--',label='energy',color='#e39c37')
    ax.set_xlabel(factor);ax.set_ylabel('Feasible fraction');ax2.set_ylabel('Energy proxy')
    ax.grid(alpha=.2);ax.text(.5,-.22,'(a) Population size' if factor=='population' else '(b) Generation budget',transform=ax.transAxes,ha='center')
fig.tight_layout(rect=(0,.08,1,1));fig.savefig(OUT/'figures'/'sensitivity.png',dpi=400);plt.close(fig)
