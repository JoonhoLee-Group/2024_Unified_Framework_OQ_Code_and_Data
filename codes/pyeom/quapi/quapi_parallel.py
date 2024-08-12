import itertools

import numpy
from scipy.linalg import expm

from pyeom.quapi.influence import compute_quapi_influence
import multiprocessing

def binseq(k):
    return [
        "".join(x) for x in itertools.product("0123", repeat=k)
    ]  # all possible paths 4^k
def updateUIterative(args):
    path, kmax, I, P, rdt, time = args
    idxm = path[1:]
    temp = (
        P[idxm[-2:]] *rdt[time][path[:-1]]
    )  # Pij Rjkl
    for i in range(1, kmax + 1):
    
        temp *= I[(path[-1]),(path[-1-i]),  i]
    
    return idxm, temp,path[-1]

def quapi_propagate(rho0,N, delt, omega, beta, ga, ohmicity, wc, Hs, kmax,NP):
    Id = numpy.array([[1, 0], [0, 1]], dtype="complex")
    P = expm(numpy.kron(1j * Hs * delt, Id) + numpy.kron(Id, (-1j * Hs * delt)))
    A = numpy.zeros((4, N), dtype="complex")
    A[:, 0] = rho0
    pool = multiprocessing.Pool(processes=NP)    
    
    I0, I, Ioz, Iz, IN = compute_quapi_influence(omega, beta, delt, ga, ohmicity, wc, kmax, hbar=1.0)
    
    paths = binseq(kmax + 1)
    paths = [tuple(map(int, s)) for s in paths]

    pathsfin = binseq(kmax)
    pathsfin = [tuple(map(int, s)) for s in pathsfin]


    #for finalN in numpy.arange(kmax, N):
    
    rdt = numpy.zeros([N] + [4] * kmax, dtype="complex")
    #rdt2 = numpy.zeros([N] + [4] * kmax, dtype="complex")
    init = numpy.array([0] * (kmax))
    #rdt[tuple(init)] = 1
    for i in range(4):
        init[0]=i
        rdt[0][tuple(init)] = rho0[i]
    ##Initial time propagation
    if kmax>1:
        comb=list(itertools.combinations(range(1,kmax),2)) #all pairs 0 to kmax
        print(comb)
        for path in pathsfin:
            
            temp=1.0+0.0j
            temp2=P[int(path[int(0)]),int(path[int(1)])]*Ioz[int(path[0])]
            temp3=1.0+0.0j
            for c in comb:
                temp*=I[int(path[int(c[1])]),int(path[int(c[0])]),int(c[1]-c[0])]
                
            for t in range(1,kmax-1):
            
                temp2*=P[int(path[int(t)]),int(path[int(t+1)])]#*I0[int(path[int(t+1)])]
            
            for t in range(1,kmax):
                temp3*=Iz[int(path[int(0)]),int(path[int(t)]),int(t)]*I0[int(path[t])]
            
            rdt[kmax][path]+=temp*temp2*temp3*rho0[path[0]]

    #--------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------#
    
    # propagation from 1 to when the right side of the window hits N
    if kmax>1:
        
        for time in numpy.arange(kmax, N-1):
            results = pool.map(updateUIterative, [(path, kmax, I, P, rdt,time) for path in paths])
            for result in results:
                rdt[time+1][result[0]] += result[1]*I0[result[2]]
            
        pool.close()
        pool.join()
        
    for time in numpy.arange(1, N-kmax-1):
        for path in pathsfin:
            A[(path[-1])][time-1] += (
                rdt[time][path]#*I0[path[0]]
            )
    # exact short time propagation from 1 to kmax
    A[:,0]=rho0
    for k in numpy.arange(1,kmax+1):
        U=numpy.zeros((4,4),dtype='complex')
        paths=binseq(k+1)
        comb=list(itertools.combinations(range(1,k+1),2)) #all pairs 0 to kmax
        for path in paths:
            temp=1.0+0.0j
            temp2=P[int(path[int(0)]),int(path[int(1)])]*Ioz[int(path[0])]
            temp3=1.0+0.0j
            for c in comb:
                temp*=I[int(path[int(c[0])]),int(path[int(c[1])]),int(c[1]-c[0])]
                
            for t in range(1,k):
                temp2*=P[int(path[int(t)]),int(path[int(t+1)])]#*I0[int(path[int(t+1)])]
                
            for t in range(1,k+1):
                temp3*=Iz[int(path[int(0)]),int(path[int(t)]),int(t)]*I0[int(path[t])]
            U[int(path[0]),int(path[-1])]+=temp*temp2*temp3
        
        
        A[:,k]=numpy.matmul(U,A[:,0])
    A[:,0]=rho0
    return A



def TTM(rho,rhoinitwant,tex):
    import numpy as numpy
    
    #here rho is [i][j][k] where i goes from 0 to 3 , j are the linearly independent copies , and k goes from 0 to t
    #tex is the timestep N we would like to extrapolate to
    
    dim=len(rho[:,0,0]) #dim is the length of our density vector, e.g., for spin boson this is 4
    kmax=len(rho[0,0,:]) #training length
    U=numpy.zeros((dim,dim,tex),dtype='complex') #propagator
    A=numpy.zeros((dim,tex),dtype='complex') #outputted data
    A[:,0]=rhoinitwant
    Memory=numpy.zeros((dim,dim,kmax+1),dtype = 'complex') #memory matrices
    rhoinitinv = numpy.linalg.inv(rho[:,:,0]) 
    
    for i in range(1,kmax): #this fits the propagators from time series data
        rhor = rho[:,:,i]
        U[:,:,i] = (np.matmul(rhor,rhoinitinv))
    
    Memory[:,:,1]=U[:,:,1].copy() # if we choose M21=U10
    
    for k in range(2,kmax+1): #perform TTM decomposition up until kmax
        tempM=numpy.zeros((dim,dim),dtype = 'complex')
        for j in range(1,k):
            tempM += numpy.matmul(Memory[:, :, j],U[:, :, k - j])
        Memory[:,:,(k)]=(U[:,:,(k)]-tempM)

    for i in range(1,kmax+1):
        A[:,i]=numpy.matmul(U[:,:,i].copy(),A[:,0])
    
    for i in numpy.arange(kmax+1,tex):
        for j in range(1,kmax+1):       
            U[:,:,i]=numpy.add(U[:,:,i],numpy.matmul(Memory[:,:,j],U[:,:,i-j].copy()),out=U[:,:,i])

    for i in range(1,tex):
        A[:,i]=numpy.matmul(U[:,:,i],A[:,0])
    
    return A