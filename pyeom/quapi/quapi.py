import itertools

import numpy
from scipy.linalg import expm

from pyeom.quapi.influence import compute_quapi_influence


def binseq(k):
    return [
        "".join(x) for x in itertools.product("0123", repeat=k)
    ]  # all possible paths 4^k


def quapi_propagate(rho0,N, delt, omega, beta, ga, ohmicity, wc, Hs, kmax):
    Id = numpy.array([[1, 0], [0, 1]], dtype="complex")
    P = expm(numpy.kron(1j * Hs * delt, Id) + numpy.kron(Id, (-1j * Hs * delt)))
    A = numpy.zeros((4, N), dtype="complex")
    A[:, 0] = rho0

    I0, I, Ioz, Iz, IN = compute_quapi_influence(omega, beta, delt, ga, ohmicity, wc, kmax, hbar=1.0)
    
    paths = binseq(kmax + 1)
    paths = [tuple(map(int, s)) for s in paths]

    pathsfin = binseq(kmax)
    pathsfin = [tuple(map(int, s)) for s in pathsfin]


    #for finalN in numpy.arange(kmax, N):
    
    rdt = numpy.zeros([N] + [4] * kmax, dtype="complex")
    #rdt2 = numpy.zeros([N] + [4] * kmax, dtype="complex")
    init = numpy.array([0] * (kmax + 1))
    #rdt[tuple(init)] = 1
    for i in range(4):
        init[1]=i
        rdt[tuple(init)] = rho0[i]
    ##Initial time propagation
    if kmax>1:
        for path in paths:  # ijkl
            idxp = path[:-1]  # ijk
            idxm = path[1:]  # jkl
            temp = P[idxp[:2]] * rdt[0][idxm]  # Pij Rjkl
            
            for i in range(1, kmax + 1):
                temp *= Iz[(path[0]), (path[i]), i] #Iij (1 timestep dist)
            rdt[1][idxp] += temp  * I0[idxp[1]] * Ioz[idxp[0]]
    else:
        for path in paths:  # ijkl
            idxp = path[:-1]  # ijk
            idxm = path[1:]  # jkl
            temp = P[path] * rdt[0][idxm]  # Pij Rjkl
            
            for i in range(1, kmax + 1):
                temp *= Iz[(path[0]), (path[i]), i] #Iij (1 timestep dist)
            rdt[1][idxp] += temp  * I0[path[1]] * Ioz[idxp[0]]

    # propagation from 1 to when the right side of the window hits N
    if kmax>1:
        for time in numpy.arange(1, N-1):
            for path in paths:
                idxp = path[:-1]
                idxm = path[1:]
                temp = (
                    P[idxp[:2]] * rdt[time][idxm]
                )  # Pij Rjkl
                for i in range(1, kmax + 1):
                    temp *= I[(path[0]), (path[i]), i]
                rdt[time + 1][idxp] += temp * I0[idxp[1]]
    else:
        for time in numpy.arange(1, N-1):
            for path in paths:
                idxp = path[:-1]
                idxm = path[1:]
                temp = (
                    P[path] * rdt[time][idxm]
                )  # Pij Rjkl
                for i in range(1, kmax + 1):
                    temp *= I[(path[0]), (path[i]), i]
                rdt[time + 1][idxp] += temp * I0[path[1]]
        
    for time in numpy.arange(kmax+1, N):
        for path in pathsfin:
            A[(path[0])][time] += (
                rdt[time][path]#*I0[path[0]]
            )
    # exact short time propagation from 1 to kmax
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