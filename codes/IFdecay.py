
import math as math
import numpy as np
import sympy as sp
import scipy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as scln
from scipy.integrate import trapz, quad
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k
mpl.rcParams.update(mpl.rcParamsDefault)

kmax=600
N=500
wantdelt=0.1

beta=5
kondo=0.5
ga=kondo* np.pi / 2
No=20000
hbar=1
om=np.linspace(-100,100,No)
yy1=np.zeros(No,dtype = "complex")
yy2=np.zeros(No,dtype = "complex")
tarr = np.linspace(0,wantdelt*(N-1),num=N)
delt=(tarr[1]-tarr[0])
Hs=np.array([[0,1],[1,0]],dtype = "complex")
wcut = 7.5
JJ=om.copy()
ohmicity=0.5
wc=wcut
def J(xs):  # Spectral density (Ohmic bath with exponential cutoff)
    output = np.zeros_like(xs)
    output[xs > 0] = ga * (xs**ohmicity)[xs > 0] * np.exp(-xs[xs > 0] / wc)/(wc**(ohmicity-1))
    output[xs <= 0] = - ga * ((-xs)**ohmicity)[xs <= 0] * np.exp(xs[xs <= 0] / wc)/(wc**(ohmicity-1))
    #output=np.heaviside(xs,0.5)ga*(xs**ohmicity)/(wc**(1-ohmicity))*np.exp(-(xs)/wc)
    return output
for i in enumerate(om):
    JJ[i[0]]=J(i[1])

etakkd=np.zeros(kmax+1,dtype="complex")


delt2=delt*1
for iii in enumerate(om):   
    yy1[iii[0]]=1/(2*np.pi)*J(om[iii[0]])/(om[iii[0]]**2)*np.exp(beta*hbar*om[iii[0]]/2)/(math.sinh(beta*hbar*om[iii[0]]/2))*(1-np.exp(-1j*om[iii[0]]*delt2))
etanul=trapz(yy1,om)


onetwo=[1,-1]
def yy2f(x,diffk): 
    return 2/(1*np.pi)*J(x)/(x**2)*np.exp(beta*hbar*x/2)/(math.sinh(beta*hbar*x/2))*((math.sin(x*delt2/2))**2)*np.exp(-1j*x*delt2*(diffk))
for difk in np.linspace(1,kmax,kmax,dtype="int"):
    for i in enumerate(om):
        #print(om[i[0]])
        yy2[i[0]]=2/(1*np.pi)*J(om[i[0]])/(om[i[0]]**2)*np.exp(beta*hbar*om[i[0]]/2)/(math.sinh(beta*hbar*om[i[0]]/2))*((math.sin(om[i[0]]*delt2/2))**2)*np.exp(-1j*om[i[0]]*delt2*(difk))
    #aaa=quad(yy2f,-np.inf,np.inf,args=difk)
    etakkd[difk]=trapz(yy2,om)
def Influencediff(dx1,dx2,dx3,dx4,diff):
    x1=onetwo[dx1]
    x2=onetwo[dx2]
    x3=onetwo[dx3]
    x4=onetwo[dx4]
    Sum=-1/hbar*(x3-x4)*(etakkd[diff]*x1-np.conjugate(etakkd[diff])*x2)     # eq 12 line 1 Nancy quapi I           
    return np.exp(Sum)

def Influencenull(dx1,dx2): # eq 12 line 2 Nancy quapi I    
    x1=onetwo[dx1]
    
    x2=onetwo[dx2]
    Sum=-1/hbar*(x1-x2)*((etanul)*x1-np.conjugate(etanul)*x2)                
    return np.exp(Sum)
    

I0_val=np.array([Influencenull(0,0),Influencenull(0,1),Influencenull(1,0),Influencenull(1,1)], dtype = "complex")

I_val=np.zeros((4,4,kmax+1),dtype = "complex")  
A=np.zeros((4,len(tarr)),dtype = "complex")
A[0,0]=1
P_val=np.zeros((4,4),dtype = "complex")


for i in np.arange(0,kmax+1):
    I_val[0,0,i]=Influencediff(0,0,0,0,i)
    I_val[0,1,i]=Influencediff(1,0,0,0,i)
    I_val[0,2,i]=Influencediff(0,1,0,0,i)
    I_val[0,3,i]=Influencediff(1,1,0,0,i)

    I_val[1,0,i]=Influencediff(0,0,1,0,i)
    I_val[1,1,i]=Influencediff(1,0,1,0,i)
    I_val[1,2,i]=Influencediff(0,1,1,0,i)
    I_val[1,3,i]=Influencediff(1,1,1,0,i)

    I_val[2,0,i]=Influencediff(0,0,0,1,i)
    I_val[2,1,i]=Influencediff(1,0,0,1,i)
    I_val[2,2,i]=Influencediff(0,1,0,1,i)
    I_val[2,3,i]=Influencediff(1,1,0,1,i)

    I_val[3,0,i]=Influencediff(0,0,1,1,i)
    I_val[3,1,i]=Influencediff(1,0,1,1,i)
    I_val[3,2,i]=Influencediff(0,1,1,1,i)
    I_val[3,3,i]=Influencediff(1,1,1,1,i)

#plt.rc('text', usetex=True)
plt.rc("font", family="serif")
plt.rc("xtick")
plt.rc("ytick")
#plt.rc("text", usetex=True)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fnorm=np.zeros((kmax+1),dtype="complex")
matrixone=np.ones((4),dtype="complex")

for a in range(kmax+1):
    fnorm[a]=np.linalg.norm(I_val[:,:,a])/4

I0_valm=np.zeros((4,4),dtype="complex")
for v in range(4):
    I0_valm[v,v]=I0_val[v]  
#fnorm[0]=np.linalg.norm(I_val)
print(np.linalg.norm(I0_valm))
x=np.log(np.arange(1,kmax+1))
#print(sc.stats.linregress(x, y=np.log(fnorm[1:]), alternative="two-sided"))
ax.semilogy(range(1,kmax+1)*delt,np.abs(I_val[1,2,1:]-1),color="k", ls="solid",label="Re${I}_{12}^{\Delta k}$")
ax.semilogy(range(1,kmax+1)*delt,np.abs(I_val[1,3,1:]-1),color=(0.6,0.6,0.6), ls="solid",label="Re${I}_{13}^{\Delta k}$")
ax.semilogy(range(1,kmax+1)*delt,np.abs(I_val[0,0,1:]-1),color=(0.2,0.2,0.2), ls="solid",label="Re${I}_{00}^{\Delta k}$")

ax.semilogy(range(1,14)*delt,np.abs(I_val[1,2,1:14]-1),color="k", ls="none",marker='x')
ax.semilogy(range(1,14)*delt,np.abs(I_val[1,3,1:14]-1),color=(0.6,0.6,0.6), ls="none",marker='x')
ax.semilogy(range(1,14)*delt,np.abs(I_val[0,0,1:14]-1),color=(0.2,0.2,0.2), ls="none",marker='x')

ax.set_xlabel("Time")
ax.set_xlim([delt,delt*kmax])
#ax.set_ylim([0.99,1.01])
#plt.plot(np.log(abs(etakkd[1:])))
#plt.plot(2.5*1/x**0.5)
plt.legend()
#plt.savefig(f"plots/I{ohmicity}{kondo}.png",dpi=1000)

plt.show()