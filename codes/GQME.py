import math as math
import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k

kmax=5
N=200
wantdelt=0.1

Idval=np.array([[1,0],[0,1]],dtype = 'complex')
beta=5
ga=0.5* np.pi / 2
No=500
hbar=1
om=np.linspace(-150,150,No)
yy=np.zeros(No,dtype = 'complex')
tarr = np.linspace(0,wantdelt*(N-1),num=N)
delt=(tarr[1]-tarr[0])
Hs=np.array([[1,0],[0,-1]],dtype = 'complex')
wcut = 7.5


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

U=np.zeros((4,4,N),dtype = 'complex')
P_valn=scln.expm(np.kron(1j*Hs*delt/2,Idval)+np.kron(Idval,(-1j*Hs*delt/2)))
P_one=P_valn.copy()
K=np.matmul(P_valn,P_valn)
Memory=np.zeros((4,4,kmax+1),dtype = 'complex')
def binseq(k):
    return [''.join(x) for x in itertools.product('0123', repeat=k)] #all possible paths 4^k     


for k in range(1,kmax+1):
    paths=binseq(k)
    comb=list(itertools.combinations(range(0,k),2)) #all pairs 0 to kmax
    
    for path in paths:
        temp=1.0+0.0j
        temp2=I0_val[int(path[0])]
        for c in comb:
            temp*=I_val[int(path[int(c[0])]),int(path[int(c[1])]),int(c[1]-c[0])]
            
        for t in range(0,k-1):
            temp2*=K[int(path[int(t)]),int(path[int(t+1)])]*I0_val[int(path[t+1])]#I_val[int(path[int(0)]),int(path[int(t)]),int(t)]*I0_val[int(path[t])]#*I0[int(path[int(t+1)])]
        
        U[int(path[0]),int(path[-1]),int(k)]+=temp*temp2    
for i in range(4):
    for j in range(4):
        Memory[i,j,1]=K[i,j].copy()
        Memory[i,j,1]*=I_val[i,j,1]*I0_val[i]#*I0_val[j]#*I0z[j]

for k in range(2,kmax):
    tempM=np.zeros((4,4),dtype = 'complex')
    for j in range(1,k):
        tempM += np.matmul(Memory[:, :, j],U[:, :, k - j + 1])
    Memory[:,:,(k)]=(U[:,:,(k+1)]-tempM)

for i in    range(kmax+1,N):
    for j in range(1,kmax+1):       
        U[:,:,i]=np.add(U[:,:,i],np.matmul(Memory[:,:,j],U[:,:,i-j].copy()),out=U[:,:,i])
Utilde=U.copy()
for i in range(1,N):
    U[:,:,i]=np.matmul(P_one,np.matmul(U[:,:,i],P_one))
    A[:,i]=np.matmul(U[:,:,i],A[:,0])



from pyeom.quapi.quapi import quapi_propagate
kmax=kmax
nomega = No
omega = om
ohmicity=1
wc=wcut
A1 = quapi_propagate([1,0,0,0],N, delt, omega, beta,ga,ohmicity,wc,  Hs, kmax-1)

UU=np.zeros((4,4,N),dtype = 'complex')
AT=np.zeros((4,len(tarr)),dtype = 'complex')

MemoryTTM=np.zeros((4,4,kmax+1),dtype = 'complex')
UU[:,:,0:kmax+1]=Utilde[:,:,0:kmax+1]
MemoryTTM[:,:,1]=np.matmul(UU[:,:,2],np.linalg.inv(UU[:,:,1]))
#for i in range(4):
#    for j in range(4):
#        MemoryTTM[i,j,1]=K[i,j].copy()
#        MemoryTTM[i,j,1]*=I_val[i,j,1]*I0_val[i]#*I0_val[j]#*I0z[j]

for k in range(2,kmax):
    tempM=np.zeros((4,4),dtype = 'complex')
    for j in range(1,k):
        tempM += np.matmul(MemoryTTM[:, :, j],UU[:, :, k - j + 1])
    MemoryTTM[:,:,(k)]=(UU[:,:,(k) + 1]-tempM)
#from pyeom.ttm.ttm import generate_Mnew
#MemoryTTM=generate_Mnew(Utilde,kmax+1)
from pyeom.redfield.redfield import redfield_propagate
A2 = redfield_propagate([1,0,0,0],N, delt, beta,ga,ohmicity,wc,  Hs,np.array([[1,0],[0,-1]]))
from pyeom.asmatpi.asmatpi import asmatpi_propagate
A3 = asmatpi_propagate([1,0,0,0],N, delt, omega, beta, ga, ohmicity, wc, Hs, kmax)

for i in np.arange(kmax+1,N):
    for j in range(1,kmax+1):       
        UU[:,:,i]=np.add(UU[:,:,i],np.matmul(MemoryTTM[:,:,j],UU[:,:,i-j].copy()),out=UU[:,:,i])
    #AT[:,i]=np.matmul(UU[:,:,i],A[:,0])
for i in range(1,N-1):  
    UU[:,:,i]=np.matmul(P_one,np.matmul(UU[:,:,i],P_one))
    AT[:,i]=np.matmul(UU[:,:,i],A[:,0])

UU=np.zeros((4,4,N),dtype = 'complex')

MemoryTTMold=np.zeros((4,4,kmax+1),dtype = 'complex')
UU[:,:,0:kmax+1]=U[:,:,0:kmax+1]
MemoryTTMold[:,:,1]=UU[:,:,1]#np.matmul(UU[:,:,2],np.linalg.inv(UU[:,:,1]))
for k in range(2,kmax+1):
    tempM=np.zeros((4,4),dtype = 'complex')
    for j in range(1,k):
        tempM += np.matmul(MemoryTTMold[:, :, j],UU[:, :, k - j])
    MemoryTTMold[:,:,(k)]=(UU[:,:,(k)]-tempM)
print(MemoryTTMold[:,:,:])
exit()
ATold=AT.copy()
for i in range(1,kmax+1):
    ATold[:,i]=np.matmul(UU[:,:,i].copy(),A[:,0])
for i in np.arange(kmax+1,N):
    for j in range(1,kmax+1):       
        UU[:,:,i]=np.add(UU[:,:,i],np.matmul(MemoryTTMold[:,:,j],UU[:,:,i-j].copy()),out=UU[:,:,i])
    #AT[:,i]=np.matmul(UU[:,:,i],A[:,0])
for i in range(1,N-1):  
    ATold[:,i+1]=np.matmul(UU[:,:,i],A[:,0])

I121=MemoryTTM[1,2,1]/K[1,2]/Utilde[2,2,1]

I2=np.zeros((4,4),dtype='complex')
denum=np.zeros((4,4),dtype='complex')
for i in range(4):
    for k in range(4):
        for j in range(4):
            denum[i,k]+=MemoryTTM[i,j,1]*MemoryTTM[j,k,1]*Utilde[k,k,1]
for i in range(4):
    for k in range(4):
        I2[i,k]=1+(MemoryTTM[i,k,2])/(denum[i,k])
Memlength=np.zeros(kmax+1)
for i in range(kmax+1):
    Memlength[i]=np.linalg.norm(MemoryTTM[:,:,i])
MK=np.zeros((4,4,kmax+1),dtype = 'complex')
G=P_one.copy()
Ls=np.kron(Hs,Idval)+np.kron(Idval,(-1*Hs))
Id4=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
Lsmin1=-1j*Ls*delt+Id4
I0=I0_val.copy()
for i in range(4):
    for k in range(4):
        a=0.0+0.0j
        for j in range(4):
            a+=G[i,j]*I0[j]*G[j,k]
        MK[i,k,1]=1/(delt**2)*(a-(Lsmin1[i,k]))
for i in range(4):
    for m in range(4):
        for j in range(4):
            for k in range(4):
                MK[i,m,2]+=1/(delt**2)*(G[i,j]*I0_val[j]*K[j,k]*(I_val[j,k,1]-1)*I0_val[k]*G[k,m])
for i in range(4):
    for p in range(4):
        for j in range(4):
            for k in range(4):
                for n in range(4):
                    MK[i,p,3]+=1/(delt**2)*(G[i,j]*K[j,k]*K[k,n]*((I_val[j,n,2]-1)*I_val[j,k,1]*I_val[k,n,1]+(I_val[j,k,1]-1)*(I_val[k,n,1]-1))*I0_val[j]*I0_val[k]*I0_val[n]*G[n,p])
for i in range(4):
    for p in range(4):
        for j in range(4):
            for k in range(4):
                for n in range(4):
                    for l in range(4):
                        MK[i,l,4]+=1/(delt**2)*(G[i,j]*K[j,k]*K[k,n]*K[n,p]*(I_val[j,p,3]*I_val[j,n,2]*I_val[k,p,2]*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1]+I_val[j,k,1]-1+I_val[k,n,1]-I_val[j,n,2]*I_val[j,k,1]*I_val[k,n,1]-I_val[j,k,1]*I_val[n,p,1]+I_val[n,p,1]-I_val[k,p,2]*I_val[k,n,1]*I_val[n,p,1])*I0_val[j]*I0_val[p]*I0_val[k]*I0_val[n]*G[p,l])
MKK4=np.zeros((4,4),dtype='complex')
for i in range(4):
    for p in range(4):
        for j in range(4):
            for k in range(4):
                for n in range(4):
                    for l in range(4):
                        MKK4[i,l]+=1/(delt**2)*(G[i,j]*K[j,k]*K[k,n]*K[n,p]*I0_val[j]*I0_val[p]*I0_val[k]*I0_val[n]*G[p,l]*((I_val[j,p,3]-1)*I_val[j,n,2]*I_val[k,p,2]*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1]+I_val[k,n,1]*((I_val[j,n,2]-1)*(I_val[k,p,2]-1)*I_val[j,k,1]*I_val[n,p,1]+(I_val[j,n,2]-1)*(I_val[n,p,1]-1)*I_val[j,k,1]+(I_val[n,p,1])*(I_val[k,p,2]-1)*(I_val[j,k,1]-1))+(I_val[j,k,1]-1)*(I_val[k,n,1]-1)*(I_val[n,p,1]-1)))
MKK5=np.zeros((4,4),dtype='complex')
for i in range(4):
    for p in range(4):
        for j in range(4):
            for k in range(4):
                for n in range(4):
                    for l in range(4):
                        for o in range(4):
                            MKK5[i,o]+=1/(delt**2)*(G[i,j]*K[j,k]*K[k,n]*K[n,p]*K[p,l]*I0_val[j]*I0_val[p]*I0_val[k]*I0_val[n]*I0_val[p]*G[l,o]*((I_val[j,l,4]-1)*(I_val[k,l,3])*(I_val[j,p,3])*I_val[j,n,2]*I_val[k,p,2]*I_val[n,l,2]*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1]*I_val[p,l,1]\
                                +(I_val[k,l,3]-1)*(I_val[j,p,3]-1)*I_val[j,n,2]*I_val[k,p,2]*I_val[n,l,2]*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1]*I_val[p,l,1]\
                                    +(I_val[k,l,3]-1)*(I_val[j,n,2]-1)*I_val[k,p,2]*(I_val[n,l,2])*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1]*I_val[p,l,1]\
                                        +(I_val[j,p,3]-1)*I_val[j,n,2]*I_val[k,p,2]*(I_val[n,l,2]-1)*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1]*I_val[p,l,1]\
                                            +(I_val[j,p,3]-1)*I_val[j,n,2]*I_val[k,p,2]*I_val[j,k,1]*I_val[k,n,1]*I_val[n,p,1]*(I_val[p,l,1]-1)\
                                                +(I_val[k,l,3]-1)*I_val[k,p,2]*(I_val[n,l,2])*(I_val[j,k,1]-1)*I_val[k,n,1]*I_val[n,p,1]*I_val[p,l,1]\
                                                    +(I_val[j,n,2]-1)*(I_val[k,p,2]-1)*(I_val[n,l,2]-1)*(I_val[j,k,1])*I_val[k,n,1]*I_val[n,p,1]*I_val[p,l,1]\
                                                        +(I_val[j,n,2]-1)*(I_val[k,p,2]-1)*(I_val[n,l,2]-1)*(I_val[j,k,1])*I_val[k,n,1]*I_val[n,p,1]*I_val[p,l,1]\
                                                            +(I_val[n,l,2]-1)*(I_val[j,k,1]-1)*(I_val[k,n,1]-1)*I_val[n,p,1]*I_val[p,l,1]\
                                                                +(I_val[k,p,2]-1)*(I_val[j,k,1]-1)*I_val[k,n,1]*I_val[n,p,1]*(I_val[p,l,1]-1)\
                                                                    +(I_val[j,n,2]-1)*(I_val[j,k,1])*I_val[k,n,1]*(I_val[n,p,1]-1)*(I_val[p,l,1]-1)\
                                                                        +(I_val[j,k,1]-1)*(I_val[k,n,1]-1)*(I_val[n,p,1]-1)*(I_val[p,l,1]-1)\
                            ))
MK[:,:,5]=MKK5
MemoryTTMold4=np.zeros((4,4),dtype = 'complex')
MK12=1/(delt**2)*(MemoryTTMold[:,:,1]-(Lsmin1))
MK22=1/(delt**2)*(MemoryTTMold[:,:,2])
MK33=1/(delt**2)*(MemoryTTMold[:,:,3])
MK44=1/(delt**2)*(MemoryTTMold[:,:,4])
MK55=1/(delt**2)*(MemoryTTMold[:,:,5])
MemoryTTMold4=1/(delt**2)*(UU[:,:,4]-UU[:,:,3]@UU[:,:,1]-UU[:,:,1]@UU[:,:,3]+UU[:,:,1]@UU[:,:,2]@UU[:,:,1])
MKTTM=MK.copy()
MKTTM[:,:,1]=MK12
MKTTM[:,:,2]=MK22
MKTTM[:,:,3]=MK33
MKTTM[:,:,4]=MK44
MKTTM[:,:,5]=MK55
UUGQME=np.zeros((4,4,N),dtype = 'complex')
UUGQME[:,:,:kmax+1]=U[:,:,:kmax+1]
ATGQME=np.zeros((4,len(tarr)),dtype = 'complex')
ATGQME[0,0]=1
for i in range(1,kmax+1):
    ATGQME[:,i]=np.matmul(UUGQME[:,:,i],ATGQME[:,0])
    
for i in np.arange(kmax+1,N):
    UUGQME[:,:,i]=Lsmin1@UUGQME[:,:,i-1]
    for j in range(1,kmax+1):       
        UUGQME[:,:,i]=np.add(UUGQME[:,:,i],delt**2*np.matmul(MK[:,:,j],UUGQME[:,:,i-j].copy()),out=UUGQME[:,:,i])
    ATGQME[:,i]=np.matmul(UUGQME[:,:,i],A[:,0])
plt.rc('font', family='serif')
plt.rc('xtick')
plt.rc('ytick')
#plt.rc('text', usetex=True)
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)
xax=np.array([1,2,3,4,5])*delt
#ax.plot(xax,MK[1,2,1:],color='k', ls='solid',label=r'$\mathcal{K}_{12}$')
#ax.plot(xax,MKTTM[1,2,1:],ls='none',color='k',marker='.')

#ax.plot(xax,MK[0,2,1:],color='blue', ls='solid',label=r'$\mathcal{K}_{02}$')
#ax.plot(xax,MKTTM[0,2,1:],color='blue', ls='none',marker='.')

#ax.plot(xax,MK[0,3,1:],color='red', ls='solid',label=r'$\mathcal{K}_{03}$')
#ax.plot(xax,MKTTM[0,3,1:],color='red', ls='none',marker='.')
fnorm=np.zeros((kmax+1),dtype='complex')
matrixone=np.ones((4),dtype='complex')
#ax.plot(xax,MK[0,0,1:],color='green', ls='solid',label=r'$\mathcal{K}_{00}$')
#ax.plot(xax,MKTTM[0,0,1:],color='green', ls='none',marker='.')
for a in range(kmax+1):
    fnorm[a]=np.linalg.norm(I_val[:,:,a]-matrixone)

I0_valm=np.zeros((4,4),dtype='complex')
for v in range(4):
    I0_valm[v,v]=I0_val[v]  
fnorm[0]=np.linalg.norm(I0_valm)
#print(np.linalg.norm(I0_valm))
#print(sc.stats.linregress(np.log(np.array([1,2,3,4,5,6,7])), y=np.log(fnorm[1:]), alternative='two-sided'))
ax.semilogy(fnorm)
#ax.plot(tarr,A[0,:],color='0.25', ls='none',linewidth=1.5,marker='.',markevery=3,label='TTSMaTPI    ')
#ax.plot(tarr,ATold[0,:],color='0.1', ls='none',linewidth=1.5,marker='x',markevery=4,label='TTM')
ax.set_xlabel('Time')
#ax.set_ylabel(r'$\rho_{11}$')
#ax.set_ylim([0.3,1])
#ax.set_xlim([0,20])
plt.legend()
print(MemoryTTMold[:,:,3])
#print((I_val[:,:,1]-1))
#print(MemoryTTMold4[:,:])
plt.show()
