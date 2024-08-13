import itertools

import numpy
from scipy.linalg import expm

from pyeom.quapi.influence import compute_quapi_influence
import multiprocessing




def binseq(k):
    return [
        "".join(x) for x in itertools.product("0123", repeat=k)
    ]  # all possible paths 4^k

def updateUExact(args):
    path, k, I, P, I0, Iz, Ioz,comb = args
    temp=1.0+0.0j
    temp2=P[int(path[int(0)]),int(path[int(1)])]*Ioz[int(path[0])]
    temp3=1.0+0.0j
    for c in comb:
        temp*=I[int(path[int(c[0])]),int(path[int(c[1])]),int(c[1]-c[0])]
                
    for t in range(1,k):
        temp2*=P[int(path[int(t)]),int(path[int(t+1)])]#*I0[int(path[int(t+1)])]
                
    for t in range(1,k+1):
        temp3*=Iz[int(path[int(0)]),int(path[int(t)]),int(t)]*I0[int(path[t])]
    return path[0], path[-1], temp * temp2 * temp3
def updaterdtinit(args):
    path, kmax, I, P, I0, Iz, Ioz,comb = args
    temp=1.0+0.0j
    temp2=P[int(path[int(0)]),int(path[int(1)])]*Ioz[int(path[0])]
    temp3=1.0+0.0j
    for c in comb:
        temp*=I[int(path[int(c[1])]),int(path[int(c[0])]),int(c[1]-c[0])]
        
    for t in range(1,kmax-1):
    
        temp2*=P[int(path[int(t)]),int(path[int(t+1)])]#*I0[int(path[int(t+1)])]
    
    for t in range(1,kmax):
        temp3*=Iz[int(path[int(0)]),int(path[int(t)]),int(t)]*I0[int(path[t])]
    return path, path[0], temp * temp2 * temp3



def updateUIterative(args):
    path, kmax, I, P, I0,rdtemp= args
    idxp = path[:-1]
    idxm = path[1:]
    temp = (
        P[idxm[-2:]] *rdtemp
    )  # Pij Rjkl
    for i in range(1, kmax + 1):
    
        temp *= I[(path[-1]),(path[-1-i]),  i]
    
    return idxm, temp,path[-1]
def quapi_propagate(rho0,N, delt, omega, beta, ga, ohmicity, wc, Hs, kmax,Nprocs):
    
    num_processes = Nprocs  # Number of parallel processes
    pool = multiprocessing.Pool(processes=num_processes)    
    
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
    init = numpy.array([0] * (kmax))
    #rdt[tuple(init)] = 1
    for i in range(4):
        init[0]=i
        rdt[0][tuple(init)] = rho0[i]
    ##Initial time propagation
    if kmax>1:
        comb=list(itertools.combinations(range(1,kmax),2)) #all pairs 0 to kmax
        results = pool.map(updaterdtinit, [(path, kmax, I, P, I0, Iz, Ioz,comb) for path in paths])
        for result in results:
            rdt[kmax,result[0]] += result[2]*rho0[result[1]]


    #--------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------#
    
    # propagation from 1 to when the right side of the window hits N rdt[time][idxp]
    if kmax>1:
        for time in numpy.arange(kmax, N-1):
            results = pool.map(updateUIterative, [(path, kmax, I, P, I0,rdt[time][path[:-1]]) for path in paths])
            for result in results:
                rdt[time+1,result[0]] += result[1]*I0[result[2]]
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
        results = pool.map(updateUExact, [(path, k, I, P, I0, Iz, Ioz,comb) for path in paths])
        for result in results:
            U[int(result[0]), int(result[1])] += result[2]
        A[:,k]=numpy.matmul(U,A[:,0])
    pool.close()
    pool.join()
    return A

