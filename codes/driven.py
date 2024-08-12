import math as math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k
kmax=4
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
for i in np.arange(1,5):
    ATGQME[:,i]=np.matmul(UUGQME[:,:,i],A[:,0])
for i in np.arange(5,N):
    print(i)
    K1=np.einsum("ijk,j->ik", P1(delt*i), T1, optimize=True)+L(delt*i)
    K2=np.einsum("ijkl,jk->il", P2(delt*i), T2, optimize=True)
    K3=np.einsum("ijklm,jkl->im", P3(delt*i), T3, optimize=True)
    K4=np.einsum("ijklmn,jklm->in", P4(delt*i), T4, optimize=True)
    UUGQME[:,:,i]=-L(delt*i)@UUGQME[:,:,i-1]
    UUGQME[:,:,i]+=np.matmul(K1,UUGQME[:,:,i-1].copy())
    UUGQME[:,:,i]+=np.matmul(K2,UUGQME[:,:,i-2].copy())
    UUGQME[:,:,i]+=np.matmul(K3,UUGQME[:,:,i-3].copy())
    UUGQME[:,:,i]+=K4
    ATGQME[:,i]=np.matmul(UUGQME[:,:,i],A[:,0])

plt.rc("font", family="serif")
plt.rc("xtick")
plt.rc("ytick")
#plt.rc("text", usetex=True)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(tarr,ATGQME[0,:],color="k", ls="solid",label=r"$\rho_{00}$")
ax.plot(tarr,np.real(ATGQME[1,:]),color="0.25", ls="dashed",label=r"$Re\rho_{01}=Re\rho_{10}$")
ax.plot(tarr,ATGQME[3,:],color="0.5", ls="solid",label=r"$\rho_{11}$")
#ax.plot(tarr,ATold[0,:],color=‘0.1’, ls=‘none’,linewidth=1.5,marker=‘x’,markevery=4,label=‘TTM’)
plt.xlabel("Time")
#ax.set_ylabel(r"$\sigma_z$")
#ax.set_ylim([-0.7,1])
#ax.set_xlim([0,20])

file_path = 'Driven_HEOM_results\driven_heom_d_eps_1.out'

data = []  # to store the parsed data

with open(file_path, 'r') as file:
    for line in file:
        # Split each line into a list of values using space as the delimiter
        values = [float(val) for val in line.split()]
        data.append(values)

# Now 'data' is a list of lists, where each inner list represents a line of values
# You can access specific values using indexing, e.g., data[0] for the first line
# or data[0][1] for the second value in the first line.

# Example: Print the entire data
time=[]
y=[]
for line in data:
    time.append((line[0]))
    y.append((line[1]))

plt.plot(time,y,label=r"$\rho_{00}$ HEOM")
plt.legend()
plt.savefig("d15", dpi=200, format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None,
       )
plt.show()