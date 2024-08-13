import math as math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k
import os
import sys
import h5py
##Change parameters below to fit physical setup
kmax=5
N=200
wantdelt=0.1
Idval=np.array([[1,0],[0,1]],dtype = "complex")
beta=5
ga=0.1 * np.pi / 2
No=500
hbar=1
om=np.linspace(-150,150,No)
yy=np.zeros(No,dtype = "complex")
tarr = np.linspace(0,wantdelt*(N-1),num=N)
delt=(tarr[1]-tarr[0])
#Hs=np.array([[1,1],[1,-1]],dtype = "complex")
wcut = 7.5
def Hs(t): #Spectral density
    #offd=1/(1+t)
    #offd=np.sin(t)
    #offd=1/(1+3*(t-3)**2)
    offd=1
    diag=1*np.sin(t)
    return np.array([[1-diag,offd],[offd,-1+diag]],dtype = "complex")
def J(x): #Spectral density
    return ga*x*np.exp(-x/wcut) if x>0 else ga*x*np.exp(x/wcut) #Ohmic bath
etakkd=np.zeros(kmax+1,dtype="complex")
delt2=delt*1
for iii in enumerate(om):
    yy[iii[0]]=1/(2*np.pi)*J(om[iii[0]])/(om[iii[0]]**2)*np.exp(beta*hbar*om[iii[0]]/2)/(math.sinh(beta*hbar*om[iii[0]]/2))*(1-np.exp(-1j*om[iii[0]]*delt2))
etanul=trapz(yy,om)
for difk in np.linspace(1,kmax,kmax,dtype="int"):
    for i in enumerate(om):
        yy[i[0]]=2/(1*np.pi)*J(om[i[0]])/(om[i[0]]**2)*np.exp(beta*hbar*om[i[0]]/2)/(math.sinh(beta*hbar*om[i[0]]/2))*((math.sin(om[i[0]]*delt2/2))**2)*np.exp(-1j*om[i[0]]*delt2*(difk))
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
    return ["".join(x) for x in itertools.product("0123", repeat=k)] #all possible paths 4^k
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
U=np.zeros((4,4,N),dtype = "complex")
#P_valn=scln.expm(np.kron(1j*Hs*delt/2,Idval)+np.kron(Idval,(-1j*Hs*delt/2)))
#P_one=P_valn.copy()
#K=np.matmul(P_valn,P_valn)
def G(i,j,t):
    temp=scln.expm(np.kron(1j*Hs(t)*delt/2,Idval)+np.kron(Idval,(-1j*Hs(t)*delt/2)))
    return temp[i,j]
def Gm(t):
    temp=scln.expm(np.kron(1j*Hs(t)*delt/2,Idval)+np.kron(Idval,(-1j*Hs(t)*delt/2)))
    return temp
def K(i,j,t):
    temp=scln.expm(np.kron(1j*Hs(t+delt/2)*delt/2,Idval)+np.kron(Idval,(-1j*Hs(t+delt/2)*delt/2)))@scln.expm(np.kron(1j*Hs(t)*delt/2,Idval)+np.kron(Idval,(-1j*Hs(t)*delt/2)))
    return temp[i,j]
T=np.zeros((4,4,kmax+1),dtype = "complex")

def Ls(t):
    return np.kron(Hs(t),Idval)+np.kron(Idval,(-1*Hs(t)))
Id4=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
#Lsmin1=-1j*Ls*delt+Id4
def L(t):
    return -1j*Ls(t)*delt+Id4
I0=I0_val.copy()
T1=np.zeros((4),dtype = "complex")

for j in range(4):
    T1[j]+=(I0_val[j])
T2=np.zeros((4,4),dtype = "complex")

for j in range(4):
    for k in range(4):
        T2[j,k]+=(I0_val[j]*(I_val[j,k,1]-1)*I0_val[k])
T3=np.zeros((4,4,4),dtype = "complex")
for j in range(4):
    for k in range(4):
        for n in range(4):
            T3[j,k,n]+=(((I_val[j,n,2]-1)*I_val[j,k,1]*I_val[k,n,1]+(I_val[j,k,1]-1)*(I_val[k,n,1]-1))*I0_val[j]*I0_val[k]*I0_val[n])
T4=np.zeros((4,4,4,4),dtype = "complex")
for p in range(4):
    for j in range(4):
        for k in range(4):
            for n in range(4):
                T4[j,k,n,p]+=((I_val[j,p,3]*I_val[j,n,2]*I_val[k,p,2]*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1]+I_val[j,k,1]-1+I_val[k,n,1]-I_val[j,n,2]*I_val[j,k,1]*I_val[k,n,1]-I_val[j,k,1]*I_val[n,p,1]+I_val[n,p,1]-I_val[k,p,2]*I_val[k,n,1]*I_val[n,p,1])*I0_val[j]*I0_val[p]*I0_val[k]*I0_val[n])

T5=np.zeros((4,4,4,4,4),dtype = "complex")
for p in range(4):
    for j in range(4):
        for k in range(4):
            for n in range(4):
                for l in range(4):
                    T5[j,k,n,p,l]+=(I_val[j,l,4]-1)*(I_val[j,p,3])*(I_val[k,l,3])*(I_val[j,n,2])*(I_val[k,p,2])*(I_val[n,l,2])*(I_val[j,k,1])*(I_val[k,n,1])*(I_val[n,p,1])*(I_val[p,l,1])*I0_val[j]*I0_val[p]*I0_val[k]*I0_val[n]*I0_val[l]

#T6=np.zeros((4,4,4,4,4,4),dtype = "complex")
#for p in range(4):
#    for j in range(4):
#        for k in range(4):
#            for n in range(4):
#                for l in range(4):
#                    for m in range(4):
#                        T6[j,k,n,p,l,m]+=(I_val[j,m,5]-1)*(I_val[k,m,4])*(I_val[j,l,4])*(I_val[n,m,3])*(I_val[j,p,3])*(I_val[k,l,3])*(I_val[p,m,2])*(I_val[j,n,2])*(I_val[k,p,2])*(I_val[n,l,2])*(I_val[j,k,1])*(I_val[k,n,1])*(I_val[n,p,1])*(I_val[p,l,1])*(I_val[l,m,1])*I0_val[j]*I0_val[p]*I0_val[k]*I0_val[n]*I0_val[l]*I0_val[m]


def P1(t):
    P1t=np.zeros((4,4,4),dtype = "complex")
    for i in range(4):
        for j in range(4):
            for k in range(4):
                P1t[i,j,k]=G(i,j,t+0.5*delt)*G(j,k,t)
    return P1t
def P2(t):
    P2t=np.zeros((4,4,4,4),dtype = "complex")
    for i in range(4):
        for m in range(4):
            for j in range(4):
                for k in range(4):
                    P2t[i,j,k,m]=G(i,j,t+1.5*delt)*K(j,k,t+0.5*delt)*G(k,m,t)
    return P2t
def P3(t):
    P3t=np.zeros((4,4,4,4,4),dtype = "complex")
    for i in range(4):
        for n in range(4):
            for j in range(4):
                for k in range(4):
                    for p in range(4):
                        P3t[i,j,k,n,p]=G(i,j,t+2.5*delt)*K(j,k,t+1.5*delt)*K(k,n,t+0.5*delt)*G(n,p,t)
    return P3t
def P4(t):
    P4t=np.zeros((4,4,4,4,4,4),dtype = "complex")
    for i in range(4):
        for n in range(4):
            for j in range(4):
                for k in range(4):
                    for p in range(4):
                        for l in range(4):
                            P4t[i,j,k,n,p,l]=G(i,j,t+3.5*delt)*K(j,k,t+2.5*delt)*K(k,n,t+1.5*delt)*K(n,p,t+0.5*delt)*G(p,l,t)
    return P4t
def P5(t):
    P5t=np.zeros((4,4,4,4,4,4,4),dtype = "complex")
    for i in range(4):
        for n in range(4):
            for j in range(4):
                for k in range(4):
                    for p in range(4):
                        for l in range(4):
                            for m in range(4):
                                P5t[i,j,k,n,p,l,m]=G(i,j,t+4.5*delt)*K(j,k,t+3.5*delt)*K(k,n,t+2.5*delt)*K(n,p,t+1.5*delt)*K(p,l,t+0.5*delt)*G(l,m,t)
    return P5t
def P6(t):
    P6t=np.zeros((4,4,4,4,4,4,4,4),dtype = "complex")
    for i in range(4):
        for n in range(4):
            for j in range(4):
                for k in range(4):
                    for p in range(4):
                        for l in range(4):
                            for m in range(4):
                                for o in range(4):
                                    P6t[i,j,k,n,p,l,m,o]=G(i,j,t+5.5*delt)*K(j,k,t+4.5*delt)*K(k,n,t+3.5*delt)*K(n,p,t+2.5*delt)*K(p,l,t+1.5*delt)*K(l,m,t+0.5*delt)*G(m,o,t)
    return P6t

UUGQME=np.zeros((4,4,N),dtype = "complex")
ATGQME=np.zeros((4,len(tarr)),dtype = "complex")
ATGQME[0,0]=1
for i in range(4):
    for j in range(4):
        for k in range(4):
            UUGQME[i,k,1]+=G(i,j,0+0.5*delt)*(I0_val[j])*G(j,k,0*delt)
print(UUGQME[:,:,1])
for i in range(4):
        for m in range(4):
            for j in range(4):
                for k in range(4):
                    UUGQME[i,m,2]+=G(i,j,0+1.5*delt)*(I0_val[j]*(I_val[j,k,1])*I0_val[k])*K(j,k,0+0.5*delt)*G(k,m,0)
for i in range(4):
        t=0
        for n in range(4):
            for j in range(4):
                for k in range(4):
                    for p in range(4):
                        UUGQME[i,p,3]+=(I_val[j,n,2])*I_val[j,k,1]*I_val[k,n,1]*I0_val[j]*I0_val[k]*I0_val[n]*G(i,j,t+2.5*delt)*K(j,k,t+1.5*delt)*K(k,n,t+0.5*delt)*G(n,p,t)
for i in range(4):
        t=0
        for n in range(4):
            for j in range(4):
                for k in range(4):
                    for p in range(4):
                        for l in range(4):
                            UUGQME[i,l,4]+=((I_val[j,p,3]*I_val[j,n,2]*I_val[k,p,2]*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1])*I0_val[j]*I0_val[p]*I0_val[k]*I0_val[n])*G(i,j,t+3.5*delt)*K(j,k,t+2.5*delt)*K(k,n,t+1.5*delt)*K(n,p,t+0.5*delt)*G(p,l,t)

for i in range(4):
        t=0
        for n in range(4):
            for j in range(4):
                for k in range(4):
                    for p in range(4):
                        for l in range(4):
                            for m in range(4):
                                UUGQME[i,m,5]+=((I_val[j,l,4]*I_val[k,l,3]*I_val[n,l,2]*I_val[p,l,1]*I_val[j,p,3]*I_val[j,n,2]*I_val[k,p,2]*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1])*I0_val[j]*I0_val[p]*I0_val[k]*I0_val[n]*I0_val[l])*G(i,j,t+4.5*delt)*K(j,k,t+3.5*delt)*K(k,n,t+2.5*delt)*K(n,p,t+1.5*delt)*K(p,l,t+0.5*delt)*G(l,m,t)

#for i in range(4):
#        t=0
#        for n in range(4):
#            for j in range(4):
#                for k in range(4):
#                    for p in range(4):
#                        for l in range(4):
#                            for m in range(4):
#                                for x in range(4):
#                                    UUGQME[i,x,6]+=((I_val[j,m,5]*I_val[k,m,4]*I_val[n,m,3]*I_val[p,m,2]*I_val[l,m,1]*I_val[j,l,4]*I_val[k,l,3]*I_val[n,l,2]*I_val[p,l,1]*I_val[j,p,3]*I_val[j,n,2]*I_val[k,p,2]*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1])*I0_val[j]*I0_val[p]*I0_val[k]*I0_val[n]*I0_val[l]*I0_val[m])*G(i,j,t+5.5*delt)*K(j,k,t+4.5*delt)*K(k,n,t+3.5*delt)*K(n,p,t+2.5*delt)*K(p,l,t+1.5*delt)*K(l,m,t+0.5*delt)*G(m,x,t)



for i in np.arange(1,kmax+1):
    ATGQME[:,i]=np.matmul(UUGQME[:,:,i],A[:,0])
for i in np.arange(kmax+1,N):
    print(i)
    K1=np.einsum("ijk,j->ik", P1(delt*i), T1, optimize=True)+L(delt*i)
    K2=np.einsum("ijkl,jk->il", P2(delt*i), T2, optimize=True)
    K3=np.einsum("ijklm,jkl->im", P3(delt*i), T3, optimize=True)
    K4=np.einsum("ijklmn,jklm->in", P4(delt*i), T4, optimize=True)
    K5=np.einsum("ijklmnp,jklmn->ip", P5(delt*i), T5, optimize=True)
    #K6=np.einsum("ijklmnpo,jklmnp->io", P6(delt*i), T6, optimize=True)
    UUGQME[:,:,i]=-L(delt*i)@UUGQME[:,:,i-1]
    UUGQME[:,:,i]+=np.matmul(K1,UUGQME[:,:,i-1].copy())
    UUGQME[:,:,i]+=np.matmul(K2,UUGQME[:,:,i-2].copy())
    UUGQME[:,:,i]+=np.matmul(K3,UUGQME[:,:,i-3].copy())
    UUGQME[:,:,i]+=np.matmul(K4,UUGQME[:,:,i-4].copy())
    UUGQME[:,:,i]+=np.matmul(K5,UUGQME[:,:,i-5].copy())
    #UUGQME[:,:,i]+=np.matmul(K6,UUGQME[:,:,i-6].copy())
mypath = f"drivendata"
if not os.path.isdir(mypath):
    os.makedirs(mypath)
f = h5py.File(f"{mypath}/td25.h5", "w")
f["UB"]=UUGQME.copy()
for i in np.arange(kmax+1,N):
    UUGQME[:,:,i]=Gm(i*delt-0.5) @ UUGQME[:,:,i] @ Gm(0)
    ATGQME[:,i]=np.matmul(UUGQME[:,:,i],A[:,0])


f["A"]=ATGQME.copy()
f["t"]=tarr.copy()
f["U"]=UUGQME.copy()
f.close()