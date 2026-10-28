import numpy as np, pandas as pd, sys
sys.argv=['x']; 
import SurfaceVolatility as sv
from scipy.stats import norm
rng=np.random.default_rng(1)
s=pd.read_csv('output/OEX_calibration_summary.csv')
H=0.01
def exp_iv(S0,r,V0,k,th,lam,rho,T,m):
    psi,a0,a1,a2,b0,b2=sv.compute_coefficients(V0,k,th,lam,rho,T)
    x=np.log(S0);K=m*S0;P=lambda xx: sv.BS_put(xx,psi,K,r,T)
    P0=P(x);h=H
    d1=(P(x+h)-P(x-h))/(2*h);d2=(P(x+h)-2*P0+P(x-h))/h**2
    d3=(P(x+2*h)-2*P(x+h)+2*P(x-h)-P(x-2*h))/(2*h**3)
    d4=(P(x+2*h)-4*P(x+h)+6*P0-4*P(x-h)+P(x-2*h))/h**4
    p=P0+a0*d1+a1*d2+a2*d3+b0*d2+b2*d4
    return sv.implied_vol_put(p,S0,K,r,T),p
def mc_put(S0,r,V0,k,th,lam,rho,T,m,N=400000,steps=None):
    steps=steps or int(252*T)*2; dt=T/steps
    lnS=np.zeros(N);V=np.full(N,V0)
    for _ in range(steps):
        z1=rng.standard_normal(N);z2=rho*z1+np.sqrt(1-rho**2)*rng.standard_normal(N)
        lnS+=(r-0.5*V**2)*dt+V*np.sqrt(dt)*z1
        V=np.abs(V+k*(th-V)*dt+lam*V*np.sqrt(dt)*z2)
    ST=S0*np.exp(lnS);disc=np.exp(-r*T);K=m*S0
    pay=np.maximum(K-ST,0)*disc  # put
    # control variate: discounted ST has mean S0
    c=np.cov(pay,ST*disc)[0,1]/np.var(ST*disc);pay2=pay-c*(ST*disc-S0)
    return pay2.mean(),pay2.std()/np.sqrt(N)
rows=[]
for _,q in s.iterrows():
    for T in [0.25,1.0]:
        for m in [1.0]:
            r=0.02
            ive,pe=exp_iv(q.S0,r,q.V0,q.kappa,q.theta,q.lam,q.rho,T,m)
            pm,se=mc_put(q.S0,r,q.V0,q.kappa,q.theta,q.lam,q.rho,T,m)
            ivm=sv.implied_vol_put(pm,q.S0,m*q.S0,r,T)
            ivh=sv.implied_vol_put(pm+1.96*se,q.S0,m*q.S0,r,T)-ivm
            rows.append((q.Period,T,round(ive,4),round(ivm,4),round(ive-ivm,4),round(ivh,4)))
res=pd.DataFrame(rows,columns=['Period','T','Expansion_ATM_IV','MC_ATM_IV','Diff','MC_95pct_halfwidth']); res.to_csv('output/OEX_expansion_vs_MC_check.csv',index=False)
print(pd.DataFrame(rows,columns=['Period','T','Expansion_ATM_IV','MC_ATM_IV','Diff','MC_95%_halfwidth']).to_string(index=False))
