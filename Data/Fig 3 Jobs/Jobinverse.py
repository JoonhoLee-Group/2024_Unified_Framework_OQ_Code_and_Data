import math as math
import numpy as np
import sympy as sp
from mpmath import coth
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k
import os
import sys
import h5py
import multiprocessing
import matplotlib.pyplot as plt
from pyeom.inverse.inverse import *
kmaxcut=int(sys.argv[3])
N=500
wantdelt=0.1
kmax=13
Idval=np.array([[1,0],[0,1]],dtype = 'complex')
beta=5
kondo=float(sys.argv[1])
ga=kondo * np.pi / 2
No=500
hbar=1
om=np.linspace(-150,150,No)
yy=np.zeros(No,dtype = 'complex')
tarr = np.linspace(0,wantdelt*(N-1),num=N)
delt=(tarr[1]-tarr[0])

Hs=np.array([[0,1],[1,0]],dtype = 'complex')
wcut = 7.5
wc=wcut
ohmicity=float(sys.argv[2])
def J(xs):  # Spectral density (Ohmic bath with exponential cutoff)
    output = np.zeros_like(xs)
    output[xs > 0] = ga * (xs**ohmicity)[xs > 0] * np.exp(-xs[xs > 0] / wc)/(wc**(ohmicity-1))
    output[xs <= 0] = - ga * ((-xs)**ohmicity)[xs <= 0] * np.exp(xs[xs <= 0] / wc)/(wc**(ohmicity-1))
    #output=np.heaviside(xs,0.5)ga*(xs**ohmicity)/(wc**(1-ohmicity))*np.exp(-(xs)/wc)
    return output
etakkd=np.zeros(kmax+1,dtype='complex')

for iii in enumerate(om):
    #np.exp(beta*hbar*om[iii[0]]/2)/(math.sinh(beta*hbar*om[iii[0]]/2))
    yy[iii[0]]=1/(2*np.pi)*J(om[iii[0]])/(om[iii[0]]**2)*(1+coth(beta*hbar*om[iii[0]]/2))*(1-np.exp(-1j*om[iii[0]]*delt))
etanul=trapz(yy,om)

for difk in np.linspace(1,kmax,kmax,dtype='int'):
    for i in enumerate(om):
        #np.exp(beta*hbar*om[i[0]]/2)/(math.sinh(beta*hbar*om[i[0]]/2))
        yy[i[0]]=2/(1*np.pi)*J(om[i[0]])/(om[i[0]]**2)*(1+coth(beta*hbar*om[i[0]]/2))*((math.sin(om[i[0]]*delt/2))**2)*np.exp(-1j*om[i[0]]*delt*(difk))
    etakkd[difk]=trapz(yy,om)
onetwo=[1,-1]

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
    

def binseq(k):
    return [''.join(x) for x in itertools.product('0123', repeat=k)] #all possible paths 4^k         


I0_val=np.array([Influencenull(0,0),Influencenull(0,1),Influencenull(1,0),Influencenull(1,1)], dtype = "complex")
I_val=np.zeros((4,4,kmax+1),dtype = 'complex')
A1=np.zeros((4,len(tarr)),dtype = 'complex')
A2=np.zeros((4,len(tarr)),dtype = 'complex')
A3=np.zeros((4,len(tarr)),dtype = 'complex')
A4=np.zeros((4,len(tarr)),dtype = 'complex')

P_val=np.zeros((4,4),dtype = 'complex')


for i in np.arange(0,kmaxcut+1):
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

U=np.zeros((4,4,N),dtype = 'complex')
P_valn=scln.expm(np.kron(1j*Hs*delt/2,Idval)+np.kron(Idval,(-1j*Hs*delt/2)))
P_one=P_valn.copy()
K=np.matmul(P_valn,P_valn)
Memory=np.zeros((4,4,kmax+1),dtype = 'complex')
def binseq(k):
    return [''.join(x) for x in itertools.product('0123', repeat=k)] #all possible paths 4^k     

def updateU(args):
    path, k, I_val, K, I0_val, comb = args
    temp = 1.0 + 0.0j
    temp2 = I0_val[int(path[0])]
    
    for c in comb:
        temp *= I_val[int(path[c[0]]), int(path[c[1]]), c[1] - c[0]]

    for t in range(0, k - 1):
        temp2 *= K[int(path[t]), int(path[t + 1])] * I0_val[int(path[t + 1])]

    return path[0], path[-1], temp * temp2
    
for k in range(1, kmax + 1):
    paths = binseq(k)
    comb = list(itertools.combinations(range(0, k), 2))  # all pairs 0 to kmax
    results = [updateU((path, k, I_val, K, I0_val, comb)) for path in paths]  # Assuming these values are computed somewhere
    for result in results:
        U[int(result[0]), int(result[1]), int(k)] += result[2]
AT=np.zeros((4,len(tarr)),dtype = 'complex')

UU=np.zeros((4,4,N),dtype = 'complex')


for i in range(1,N):
    U[:,:,i]=np.matmul(P_one,np.matmul(U[:,:,i],P_one))
MemoryTTMold=np.zeros((4,4,kmax+1),dtype = 'complex')
UU[:,:,:]=U[:,:,:].copy()
MemoryTTMold[:,:,1]=U[:,:,1].copy()
for k in range(2,kmax+1):
    tempM=np.zeros((4,4),dtype = 'complex')
    for j in range(1,k):
        tempM += np.matmul(MemoryTTMold[:, :, j],UU[:, :, k - j])
    MemoryTTMold[:,:,(k)]=(UU[:,:,(k)]-tempM)

for k in np.arange(kmax,kmax+1):
    UU=np.zeros((4,4,N),dtype = 'complex')
    UU[:,:,0:k+1]=U[:,:,0:k+1]
    A1[:,0]=[1,0,0,0]
    for i in np.arange(k+1,N):
        for j in range(1,k+1):       
            UU[:,:,i]=np.add(UU[:,:,i],np.matmul(MemoryTTMold[:,:,j],UU[:,:,i-j].copy()),out=UU[:,:,i])
        #AT[:,i]=np.matmul(UU[:,:,i],A[:,0])
    for i in range(1,N):  
        #UU[:,:,i]=np.matmul(P_one,np.matmul(UU[:,:,i],P_one))
        A1[:,i]=np.matmul(UU[:,:,i],A1[:,0])

    
for k in np.arange(kmax,kmax+1):
    UU=np.zeros((4,4,N),dtype = 'complex')
    UU[:,:,0:k+1]=U[:,:,0:k+1]
    A2[:,0]=[0,0,0,1]
    for i in np.arange(k+1,N):
        for j in range(1,k+1):       
            UU[:,:,i]=np.add(UU[:,:,i],np.matmul(MemoryTTMold[:,:,j],UU[:,:,i-j].copy()),out=UU[:,:,i])
        #AT[:,i]=np.matmul(UU[:,:,i],A[:,0])
    for i in range(1,N):  
        #UU[:,:,i]=np.matmul(P_one,np.matmul(UU[:,:,i],P_one))
        A2[:,i]=np.matmul(UU[:,:,i],A2[:,0])

    #plt.plot(tarr,A2[0,:],color=(0.5, k/30, +k/10), ls='solid',linewidth=1,marker='.',markevery=10)

for k in np.arange(kmax,kmax+1):
    UU=np.zeros((4,4,N),dtype = 'complex')
    UU[:,:,0:k+1]=U[:,:,0:k+1]
    A3[:,0]=[1/2,1/2,1/2,1/2]
    for i in np.arange(k+1,N):
        for j in range(1,k+1):       
            UU[:,:,i]=np.add(UU[:,:,i],np.matmul(MemoryTTMold[:,:,j],UU[:,:,i-j].copy()),out=UU[:,:,i])
        #AT[:,i]=np.matmul(UU[:,:,i],A[:,0])
    for i in range(1,N):  
        #UU[:,:,i]=np.matmul(P_one,np.matmul(UU[:,:,i],P_one))
        A3[:,i]=np.matmul(UU[:,:,i],A3[:,0])

    #plt.plot(tarr,A3[0,:],color=(0.5, k/30, +k/10), ls='solid',linewidth=1,marker='.',markevery=10)
for k in np.arange(kmax,kmax+1):
    UU=np.zeros((4,4,N),dtype = 'complex')
    UU[:,:,0:k+1]=U[:,:,0:k+1]
    A4[:,0]=[1,1/2+1/2j,1/2-1/2j,0]
    for i in np.arange(k+1,N):
        for j in range(1,k+1):       
            UU[:,:,i]=np.add(UU[:,:,i],np.matmul(MemoryTTMold[:,:,j],UU[:,:,i-j].copy()),out=UU[:,:,i])
        #AT[:,i]=np.matmul(UU[:,:,i],A[:,0])
    for i in range(1,N):  
        #UU[:,:,i]=np.matmul(P_one,np.matmul(UU[:,:,i],P_one))
        A4[:,i]=np.matmul(UU[:,:,i],A4[:,0])


Uex=UfromA(A1,A2,A3,A4)
A5=A1.copy()
for i in range(1,N):  
    A5[:,i]=np.matmul(Uex[:,:,i],A1[:,0])
#plt.plot(A5[0,:])
#plt.plot(A5[3,:])
#plt.plot(A1[0,:])
#plt.plot(A1[3,:])
Mex=KfromU(Uex,kmax)
resI=IfromK(Mex[:,:,1:],kmax-1,P_one)
Iex=resI[1]
I0ex=resI[0]
ww=np.linspace(0,wcut+51,111)
eta=etafromI(Iex,resI[0],kmax-1)
#plt.plot(eta[:])
#plt.plot(etakkd[:],ls='dashed')
JJJ=Jfrometa(eta,kmax,beta,delt,ww)
mypath = f"inverseep0cut{kondo}{ohmicity}"
if not os.path.isdir(mypath):
    os.makedirs(mypath)
f = h5py.File(f"{mypath}/inverse{kondo}beta{beta}delt{delt}kmax{kmax}.h5", "w")  
f["J"]=JJJ.copy()
f["Mex"]=Mex.copy()
f["A1"]=A1.copy()
f["A2"]=A2.copy()
f["A3"]=A3.copy()
f["A4"]=A4.copy()
f["U"]=Uex.copy()
f["I1"]=Iex.copy()
f["I0"]=I0ex.copy()
f["w"]=ww.copy()
f["eta"]=eta.copy()
f.close()