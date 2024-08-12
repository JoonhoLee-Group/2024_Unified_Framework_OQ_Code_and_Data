import math as math
import numpy as np
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools

from pyeom.quapi.influence import compute_quapi_influence
def asmatpi_propagate(A0,N, delt, omega, beta, ga, ohmicity, wc, Hs, kmax): 

    Idval=np.array([[1,0],[0,1]],dtype = 'complex')
    beta=beta
    No=500
    hbar=1
    om=np.linspace(-150,150,No)
    yy=np.zeros(No,dtype = 'complex')
    tarr = np.linspace(0,N*delt,num=N)
    delt=(tarr[1]-tarr[0])
    Hs=np.array([[1,-1],[-1,-1]],dtype = 'complex')
    I0, I, Ioz, Iz, IN = compute_quapi_influence(omega, beta, delt, ga, ohmicity, wc, kmax, hbar=1.0)
    onetwo=[1,-1]
    def binseq(k):
        return [''.join(x) for x in itertools.product('0123', repeat=k)] #all possible paths 4^k         

    A=np.zeros((4,len(tarr)),dtype = 'complex')
    A[:,0]=A0
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
            temp2=I0[int(path[0])]
            for c in comb:
                temp*=I[int(path[int(c[0])]),int(path[int(c[1])]),int(c[1]-c[0])]
                
            for t in range(0,k-1):
                temp2*=K[int(path[int(t)]),int(path[int(t+1)])]*I0[int(path[t+1])]#I_val[int(path[int(0)]),int(path[int(t)]),int(t)]*I0_val[int(path[t])]#*I0[int(path[int(t+1)])]
            
            U[int(path[0]),int(path[-1]),int(k)]+=temp*temp2
    for i in range(4):
        for j in range(4):
            Memory[i,j,1]=K[i,j].copy()
            Memory[i,j,1]*=I[i,j,1]*I0[i]#*I0_val[j]#*I0z[j]

    for k in range(2,kmax+1):
        tempM=np.zeros((4,4),dtype = 'complex')
        for j in range(1,k):
            tempM += np.matmul(Memory[:, :, j],U[:, :, k - j])
        Memory[:,:,(k)]=(U[:,:,(k)]-tempM)

    for i in range(1,kmax+1):
        A[:,i]=np.matmul(U[:,:,i].copy(),A[:,0])
    for i in np.arange(kmax+1,N):
        for j in range(1,kmax+1):       
            U[:,:,i]=np.add(U[:,:,i],np.matmul(Memory[:,:,j],U[:,:,i-j].copy()),out=U[:,:,i])
    for i in range(1,N):
        U[:,:,i]=np.matmul(P_one,np.matmul(U[:,:,i],P_one))
        A[:,i]=np.matmul(U[:,:,i],A[:,0])
    return A