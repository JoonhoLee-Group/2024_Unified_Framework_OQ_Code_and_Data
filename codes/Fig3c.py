def onorm(M):
    MM = np.transpose(M.conj()) @ M
    eigenvalues, eigenvectors = np.linalg.eigh(MM)
    return np.sqrt(np.max(eigenvalues))







import math as math
import numpy as np
import sympy as sp
from mpmath import coth
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k
import h5py
import matplotlib.pyplot as plt
from pyeom.inverse.inverse import *
plt.rc("font", family="serif")
plt.rc("xtick")


plt.rc("ytick")
fig = plt.figure()  # width=8 inches, height=6 inches

ax = fig.add_subplot(1, 1, 1)
N=500
wantdelt=0.1
t=np.linspace(0,wantdelt*(N-1),num=N)

Idval=np.array([[1,0],[0,1]],dtype = 'complex')
Hs=np.array([[0,1],[1,0]],dtype = 'complex')
delt=wantdelt
G=scln.expm(np.kron(1j*Hs*delt/2,Idval)+np.kron(Idval,(-1j*Hs*delt/2)))
#filename = f'jobsdone\jobgqme0.50.5\dyn0.5beta5delt0.1ohmicity0.5kmax14.h5'
filename = "dyn0.5beta5delt0.1ohmicity0.5kmax13.h5"
h5 = h5py.File(filename,'r')
kmax=13
#MK=np.zeros((4,4,kmax+1),dtype='complex')
M = h5["M"]  # VSTOXX futures data
A = np.array(h5["10AT"])
print(list(A))
UU=np.zeros((4,4,N),dtype = 'complex')
UU[:,:,1]=M[:,:,1]
Mnorm=np.zeros((kmax+1),dtype = 'complex')
for k in range(2,kmax+1):
    tempM=np.zeros((4,4),dtype = 'complex')
    for j in range(1,k):
        tempM += np.matmul(M[:, :, j],UU[:, :, k - j])
    (UU[:,:,(k)])=M[:,:,(k)]+tempM
#for i in range(1,kmax+1):
#    UU[:,:,i]=np.matmul(G,np.matmul(UU[:,:,i],G))
MK=KfromU(UU,kmax)
#ax.plot(np.arange(0,14)*wantdelt,MK[1,2,:],color="k", ls="solid",label="Dyck/INFPI $\mathcal{K}_{12}^N$")
#ax.plot(np.arange(0,14)*wantdelt,MK[0,0,:],color=(0.6,0.6,0.6), ls="solid",label="Dyck/INFPI $\mathcal{K}_{00}^N$")
#ax.plot(np.arange(0,14)*wantdelt,MK[0,3,:],color=(0.2,0.2,0.2), ls="solid",label="Dyck/INFPI $\mathcal{K}_{03}^N$")
for i in range(kmax+1):
    Mnorm[i]=onorm(MK[:,:,i]/(delt**2))
#ax.plot(t,A[0,:],color="k")
ax.semilogy(np.arange(2,kmax+1)*wantdelt,Mnorm[2:],color="k", ls="none",marker='.',markersize=12,label=r'$\|\mathcal{K}_{N}\|$')

h5.close()

ax.set_facecolor((0.9,1,1))
plt.xlabel("Time",fontsize=15)
#ax.set_ylabel(r"$\sigma_z$")
#ax.set_ylim([-0.2,1])
#ax.set_xlim([0,4])

#plt.legend()
#plt.savefig("plots/K051.png",dpi=1000)
#plt.show()



























































import math as math
import numpy as np
import sympy as sp
from mpmath import coth
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k
import h5py
import matplotlib.pyplot as plt
from pyeom.inverse.inverse import *
filename = f'ttt/ttt/heomdynamicssubohmicpapertrueyes0.110.0/tmax10.0beta5alpha0.5s0.5initial1.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A1 = np.transpose(h5["rho"])

filename = f'ttt/ttt/heomdynamicssubohmicpapertrueyes0.110.0/tmax10.0beta5alpha0.5s0.5initial2.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A2 = np.transpose(h5["rho"])

filename = f'ttt/ttt/heomdynamicssubohmicpapertrueyes0.110.0/tmax10.0beta5alpha0.5s0.5initial3.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A3 = np.transpose(h5["rho"])

filename = 'ttt/ttt/heomdynamicssubohmicpapertrueyes0.110.0/tmax10.0beta5alpha0.5s0.5initial4.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A4 = np.transpose(h5["rho"])

Uex=UfromA(A1,A2,A3,A4)
A5=A1.copy()
for i in range(1,len(A1)):  
    A5[:,i]=np.matmul(Uex[:,:,i],[1,0,0,0])
mmax=99
Mex=KfromU(Uex,mmax)

t=np.arange(0,0.1*(mmax+1),0.1)
ddelt=0.1
Mnormex=np.zeros((mmax+1),dtype = 'complex')
for i in range(mmax+1):
    Mnormex[i]=onorm(Mex[:,:,i]/(ddelt**2))
ax.plot(t[2:],Mnormex[2:],color=(0,0,0),alpha=1,markersize=12, ls="solid")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.plot(t[0:500],A1[0,:500])

ax.set_xlim([0,4])
























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
beta=5
kmax=40
N=500
wantdelt=0.1
kondo=0.5
ga=kondo* np.pi / 2
No=10000
hbar=1
om=np.linspace(-150,150,No)
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

TI_val=np.zeros((4,4,kmax+1),dtype = "complex")  
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
for i in range(kmax+1):
    TI_val[:,:,i]=I_val[:,:,i]-np.ones(4)
#plt.rc('text', usetex=True)

#plt.rc("text", usetex=True)



I0_valm=np.zeros((4,4),dtype="complex")
for v in range(4):
    I0_valm[v,v]=I0_val[v]  
#fnorm[0]=np.linalg.norm(I_val)
print(np.linalg.norm(I0_valm))
x=np.log(np.arange(1,kmax+1))
for i in range(kmax):
    x[i]=onorm(TI_val[:,:,i])
#print(sc.stats.linregress(x, y=np.log(fnorm[1:]), alternative="two-sided"))
#ax.semilogy(range(1,kmax+1)*delt,np.abs(I_val[1,2,1:]-1),color="k", ls="solid",label="Re${I}_{12}^{\Delta k}$")
#ax.semilogy(range(1,kmax+1)*delt,np.abs(I_val[1,3,1:]-1),color=(0.6,0.6,0.6), ls="solid",label="Re${I}_{13}^{\Delta k}$")
#ax.semilogy(range(1,kmax+1)*delt,np.abs(I_val[0,0,1:]-1),color=(0.2,0.2,0.2), ls="solid",label="Re${I}_{00}^{\Delta k}$")

#ax.semilogy(range(1,14)*delt,np.abs(I_val[1,2,1:14]-1),color="k", ls="none",marker='x')
#ax.semilogy(range(1,14)*delt,np.abs(I_val[1,3,1:14]-1),color=(0.6,0.6,0.6), ls="none",marker='x')
#ax.semilogy(range(1,14)*delt,np.abs(I_val[0,0,1:14]-1),color=(0.2,0.2,0.2), ls="none",marker='x')

#ax.set_xlabel("Time")
#ax.set_xlim([delt,delt*kmax])
#ax.set_ylim([0.99,1.01])
#plt.plot(np.log(abs(etakkd[1:])))
#plt.plot(2.5*1/x**0.5
#ax.set_ylabel(r'$\|\mathcal{K}_{N \Delta t}\|_O$', color='k',fontsize=15)

ax.semilogy(range(2,kmax+1)*delt,x[1:], color='#5E091A')
ax.semilogy(range(2,14+1)*delt,x[1:14], color='#5E091A', ls="none",marker='x',markersize=12,label=r'$\|\tilde{I}_N\|$')
#ax.set_ylabel(, color='#5E091A',fontsize=15)
ax.tick_params('y', color='#5E091A')
#fnorm=np.zeros((kmax+1),dtype="complex")
#matrixone=np.ones((4),dtype="complex")
plt.legend(fontsize=15)
plt.ylim([10**(-4),30])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"KINorm0505b",dpi=1000,bbox_inches="tight")
plt.show()