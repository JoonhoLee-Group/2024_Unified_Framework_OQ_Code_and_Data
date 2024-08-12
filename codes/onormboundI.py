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
kondo=0.1
ga=kondo* np.pi / 2
No=500
hbar=1
ohmicity=1
om=np.linspace(-150,150,No)
yy=np.zeros(No,dtype = 'complex')
tarr = np.linspace(0,wantdelt*(N-1),num=N)
delt=(tarr[1]-tarr[0])
Hs=np.array([[0,1],[1,0]],dtype = 'complex')
wcut = 7.5
wc=wcut
def onorm(M):
    MM=np.transpose(M)@M
    eigenvalues,eigenvectors=np.linalg.eig(MM)
    
    return np.sqrt(np.max(eigenvalues))

def J(xs):  # Spectral density (Ohmic bath with exponential cutoff)
    output = np.zeros_like(xs)
    output[xs > 0] = ga * (xs**ohmicity)[xs > 0] * np.exp(-xs[xs > 0] / wc)/(wc**(ohmicity-1))
    output[xs <= 0] = - ga * ((-xs)**ohmicity)[xs <= 0] * np.exp(xs[xs <= 0] / wc)/(wc**(ohmicity-1))
    #output=np.heaviside(xs,0.5)ga*(xs**ohmicity)/(wc**(1-ohmicity))*np.exp(-(xs)/wc)
    return output

def wordgenerate(memorylength):
    for max_up in [memorylength]:
        all_paths = []
        dyckwords=[]
        # max_up = 8
        get_path(["" for i in range(2*max_up)], max_up, 0, 0, 0, all_paths)
        #print(f"max_up {max_up}, num_path {len(all_paths)}")
        for path in all_paths:
            print_path = []
            for element in path:
                if element == "up":
                    print_path.append("1")
                else:
                    print_path.append("0")
            dyckwords.append(print_path)
        return dyckwords
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
    Inorm[i]=onorm(I_val[:,:,i])/4
    ITnorm[i]=onorm(TI_val[:,:,i])/4
I0norm=np.linalg.norm(I0_val)/1
Inorm[0]=np.max(I0_val)
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

letters=["i","j","k","l","a","b","c","d","m","n","o","p","q","w","r","t"]
memorylength=kmax

Idval=np.array([[1,0],[0,1]],dtype = 'complex')

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

def get_path(N, max_up, n_up, n_diff, i, all_paths):
    if n_up == max_up:
        for j in range(i, len(N)):
            N[j] = "down"
        all_paths.append(N.copy())
        return
    elif len(N) - n_up == max_up:
        for j in range(i, len(N)):
            N[j] = "up"
        all_paths.append(N.copy())
        return
    if n_diff == 0:
        N[i] = "up"
        get_path(N, max_up, n_up+1, n_diff+1, i+1, all_paths)
    else:
        N[i] = "up"
        get_path(N, max_up, n_up+1, n_diff+1, i+1, all_paths)
        N[i] = "down"
        get_path(N, max_up, n_up, n_diff-1, i+1, all_paths)
        

def dycktocor(path):
    cl=0
    height=0
    set=[onorm(GG)/4]
    setsym=[]
    x=0
    count=1
    indic="si"
    TI_val=I_val.copy()
    for i in range(4):
        for j in range(4):
            TI_val[i,j]+=-1
    JJ=GG @ GG
    for i in enumerate(path):   
        if i[1]=='1':
            cl+=int(i[1])
            height+=1
            count=1
        else:
            if count==1:
                #set.append(f"T^{height}_(x_{int(i[0]/2-height/2)}x_{int(i[0]/2+height/2)})")
                set.append(onorm(TI_val[:,:,height])/4)
                setsym.append(f"T^{height}_(x_{int(i[0]/2-height/2)}x_{int(i[0]/2+height/2)})")
                ss=list(itertools.combinations(range(int(i[0]/2-height/2),int(height/2+i[0]/2+1)),2))
                x+=height
                #print(range(int(i[0]/2-height/2),int(height/2+i[0]/2+1)))
                #print(ss)
                indic=indic+f", {letters[int(i[0]/2-height/2)]}{letters[int(i[0]/2+height/2)]}"
                for j in ss:
                    k=j[1]-j[0]
                    if j[1]-j[0]<height:
                        if f"I^{k}_(x_{int(j[0])}x_{int(j[1])})" not in setsym:
                            setsym.append(f"I^{k}_(x_{int(j[0])}x_{int(j[1])})")
                            set.append(onorm(I_val[:,:,k])/4)
                            indic=indic+f", {letters[int(j[0])]}{letters[int(j[1])]}"
                cl=0    
            height+=-1
            count=0
    for i in range(int(len(path)/2+1)):
        indic=indic+f", {letters[int(i)]}"
        set.append(np.linalg.norm(I0_val))
    for i in range(int(len(path)/2)):
        indic=indic+f", {letters[int(i)]}{letters[int(i+1)]}"
        set.append(onorm(JJ)/4)
    indic=indic+f", {letters[int(len(path)/2)]}e->se"
    set.append(onorm(GG)/4)
    #set.append(x)
    set.append(indic)
    K=1.0+0.j
    for i in set[:-1]:
        K*=i
    #print(setsym)
    return(K)
for i in np.arange(kmax+1):
    Inorm[i]=onorm(I_val[:,:,i])/4
    ITnorm[i]=onorm(TI_val[:,:,i])
I0norm=np.linalg.norm(I0_val)/1
Inorm[0]=np.max(I0_val)

plt.semilogy(range(kmax+1),Inorm[:])
plt.xlabel('Delta k')
plt.ylabel('Norm bound of I_k')
plt.title(f'epsilon is zero, kondo is {kondo}, delt is {delt}, ohmicity is {ohmicity}')
plt.show()