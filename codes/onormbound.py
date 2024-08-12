import itertools
import math as math
import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k
import tensorflow as tf

kmax=20
N=200
wantdelt=0.1
letters=["i","j","k","l","a","b","c","d","m","n","o","p","q","w","r","t"]
memorylength=kmax

Idval=np.array([[1,0],[0,1]],dtype = 'complex')
beta=5
ga=0.1* np.pi / 2
No=500
hbar=1
om=np.linspace(-150,150,No)
yy=np.zeros(No,dtype = 'complex')
tarr = np.linspace(0,wantdelt*(N-1),num=N)
delt=(tarr[1]-tarr[0])
Hs=np.array([[0,1],[1,0]],dtype = 'complex')
wcut = 7.5
def onorm(M):
    MM=np.transpose(M)@M
    eigenvalues,eigenvectors=np.linalg.eig(MM)
    
    return np.sqrt(np.max(eigenvalues))

def J(x): #Spectral density
    return ga*x*np.exp(-x/wcut) if x>0 else ga*x*np.exp(x/wcut) #Ohmic bath

etakkd=np.zeros(kmax+1,dtype='complex')


delt2=delt*1
for iii in enumerate(om):
    yy[iii[0]]=1/(2*np.pi)*J(om[iii[0]])/(om[iii[0]]**2)*np.exp(beta*hbar*om[iii[0]]/2)/(math.sinh(beta*hbar*om[iii[0]]/2))*(1-np.exp(-1j*om[iii[0]]*delt2))
etanul=trapz(yy,om)

for difk in np.linspace(1,kmax,kmax,dtype='int'):
    for i in enumerate(om):
        yy[i[0]]=2/(1*np.pi)*J(om[i[0]])/(om[i[0]]**2)*np.exp(beta*hbar*om[i[0]]/2)/(math.sinh(beta*hbar*om[i[0]]/2))*((math.sin(om[i[0]]*delt2/2))**2)*np.exp(-1j*om[i[0]]*delt2*(difk))
    etakkd[difk]=trapz(yy,om)
onetwo=[1,-1]
P_valn=scln.expm(np.kron(1j*Hs*delt/2,Idval)+np.kron(Idval,(-1j*Hs*delt/2)))
GG=P_valn.copy()
JJ=GG @ GG
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
Inorm=np.zeros((kmax+1),dtype = 'complex')
TI_val=np.zeros((4,4,kmax+1),dtype = 'complex')

A=np.zeros((4,len(tarr)),dtype = 'complex')
A[0,0]=1
P_val=np.zeros((4,4),dtype = 'complex')


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


for i in np.arange(0,kmax+1):
    TI_val[0,0,i]=Influencediff(0,0,0,0,i)-1
    TI_val[0,1,i]=Influencediff(1,0,0,0,i)-1
    TI_val[0,2,i]=Influencediff(0,1,0,0,i)-1
    TI_val[0,3,i]=Influencediff(1,1,0,0,i)-1

    TI_val[1,0,i]=Influencediff(0,0,1,0,i)-1
    TI_val[1,1,i]=Influencediff(1,0,1,0,i)-1
    TI_val[1,2,i]=Influencediff(0,1,1,0,i)-1
    TI_val[1,3,i]=Influencediff(1,1,1,0,i)-1

    TI_val[2,0,i]=Influencediff(0,0,0,1,i)-1
    TI_val[2,1,i]=Influencediff(1,0,0,1,i)-1
    TI_val[2,2,i]=Influencediff(0,1,0,1,i)-1
    TI_val[2,3,i]=Influencediff(1,1,0,1,i)-1

    TI_val[3,0,i]=Influencediff(0,0,1,1,i)-1
    TI_val[3,1,i]=Influencediff(1,0,1,1,i)-1
    TI_val[3,2,i]=Influencediff(0,1,1,1,i)-1
    TI_val[3,3,i]=Influencediff(1,1,1,1,i)-1
ITnorm=Inorm.copy()
for i in np.arange(kmax+1):
    Inorm[i]=onorm(I_val[:,:,i])/1
    ITnorm[i]=onorm(TI_val[:,:,i])
I0norm=np.linalg.norm(I0_val)/1
Inorm[0]=np.max(I0_val)-1
crestbound=np.zeros(kmax+1,dtype='complex')
crestbound[1]=ITnorm[1]*I0norm**2
crestbound[2]=ITnorm[2]*Inorm[1]**2*I0norm**3
crestbound[3]=ITnorm[3]*Inorm[2]**2*Inorm[1]**3*I0norm**4
crestbound[4]=ITnorm[4]*Inorm[3]**2*Inorm[2]**3*Inorm[1]**4*I0norm**5
crestbound[5]=ITnorm[5]*Inorm[4]**2*Inorm[3]**3*Inorm[2]**4*Inorm[1]**5*I0norm**6
crestbound[6]=ITnorm[6]*Inorm[5]**2*Inorm[4]**3*Inorm[3]**4*Inorm[2]**5*Inorm[1]**6*I0norm**7
#crestbound[7]=ITnorm[7]*Inorm[6]**2*Inorm[5]**3*Inorm[4]**4*Inorm[3]**5*Inorm[2]**6*Inorm[1]**7*I0norm**8
print(Inorm)
plt.plot(Inorm[:])
plt.show()