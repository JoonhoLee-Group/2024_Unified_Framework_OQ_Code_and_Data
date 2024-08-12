def smatpi_propagate(N, delt, omega, beta, ga, ohmicity, wc, Hs, kmax): #smatpi hard coded kmax=rmax
    import math as math
    import numpy as np
    import sympy as sp
    import scipy.linalg as scln
    from scipy.integrate import trapz
    import itertools
    from sympy.tensor.array.expressions import ArraySymbol
    from sympy.abc import i, j, k
    from sympy import MatrixSymbol, Matrix, eye
    
    from pyeom.quapi.influence import compute_quapi_influence
    
    Iz = ArraySymbol("Iz", (4,4, kmax+1))
    Mkk1_sp = sp.MutableDenseNDimArray(np.zeros((4, 4, kmax+1), dtype=complex))
    Inf= ArraySymbol("Inf", (4,4, kmax+1))
    I0z = MatrixSymbol('I0z', 4,1)
    I0 = MatrixSymbol('I0', 4,1)
    Id=eye(2)
    Idval = np.array([[1, 0], [0, 1]], dtype="complex")
    
    I0_val, I_val, Ioz_val, Iz_val, IN_val = compute_quapi_influence(omega, beta, delt, ga, ohmicity, wc, kmax, hbar=1.0)
    
    A = np.zeros((4, N), dtype="complex")
    A[0, 0] = 1.0
    
    P_val=scln.expm(np.kron(1j*Hs*delt,Idval)+np.kron(Idval,(-1j*Hs*delt)))
    U_sp=sp.MutableDenseNDimArray(np.zeros((4, 4, N), dtype=complex))
    U=np.zeros((4,4,N),dtype = 'complex')
    
    P=MatrixSymbol('P',4,4)

    for k in range(1,kmax+1):
        paths=binseq(k+1)
        comb=list(itertools.combinations(range(1,k+1),2)) #all pairs 0 to kmax
        for path in paths:
            temp=1.0+0.0j
            temp2=P[int(path[int(0)]),int(path[int(1)])]*I0z[int(path[0])]
            temp3=1.0+0.0j
            for c in comb:
                temp*=Inf[int(path[int(c[0])]),int(path[int(c[1])]),int(c[1]-c[0])]
                
            for t in range(1,k):
                temp2*=P[int(path[int(t)]),int(path[int(t+1)])]#*I0[int(path[int(t+1)])]
                
            for t in range(1,k+1):
                temp3*=Iz[int(path[int(0)]),int(path[int(t)]),int(t)]*I0[int(path[t])]
            U_sp[int(path[0]),int(path[-1]),int(k)]+=temp*temp2*temp3
    
    for i in range(4):
        for j in range(4):
            Mkk1_sp[i,j,1]=P[i,j].copy()
            Mkk1_sp[i,j,1]*=Inf[i,j,1]*I0[i]

    for k in range(2,kmax+1):
        tempM=sp.MutableDenseNDimArray(np.zeros((4, 4), dtype=complex))
        
        for j in range(1,k):
            tempM += np.matmul(Mkk1_sp[:, :, j],U_sp[:, :, k - j])
        Mkk1_sp[:,:,(k)]=U_sp[:,:,(k)]-tempM
    for k in range(2,kmax+1):
        for i in range(4):
            for j in range(4):
                        Mkk1_sp[i, j, k] = Mkk1_sp[i, j, k]/I0z[j,0]
    for k in range(1,kmax+1): #substitute Ik0 to Ik+1 1
        for i in range(4):
            for j in range(4):
                for l in range(4):
                    Mkk1_sp[i, j, k] = Mkk1_sp[i, j, k].subs(I0z[l,0],Ioz_val[l])
                    Mkk1_sp[i, j, k] = Mkk1_sp[i, j, k].subs(I0[l,0],I0_val[l])
                    for m in range(4):
                        Mkk1_sp[i, j, k] = Mkk1_sp[i, j, k].subs(P[l,m],P_val[l,m])
                        for n in range(kmax+1):
                            Mkk1_sp[i, j, k] = Mkk1_sp[i, j, k].subs(Inf[l,m,n],I_val[l,m,n])
                            Mkk1_sp[i, j, k] = Mkk1_sp[i, j, k].subs(Iz[l,m,n],I_val[l,m,n])

        paths=binseq(k+1)
        comb=list(itertools.combinations(range(1,k+1),2)) #all pairs 0 to kmax
        
        for path in paths:
            temp=1.0+0.0j
            temp2=P_val[int(path[int(0)]),int(path[int(1)])]*Ioz_val[int(path[0])]
            temp3=1.0+0.0j
            for c in comb:
                temp*=1*I_val[int(path[int(c[0])]),int(path[int(c[1])]),int(c[1]-c[0])]
                
            for t in range(1,k):
                temp2*=P_val[int(path[int(t)]),int(path[int(t+1)])]
            
            for t in range(1,k+1):
                temp3*=I0_val[int(path[int(t)])]*Iz_val[int(path[int(0)]),int(path[int(t)]),int(t)]
                
            U[int(path[0]),int(path[-1]),int(k)]+=temp*temp2*temp3

    Memory=np.array(Mkk1_sp, dtype='complex')
    
    for i in range(1,kmax+1):
        A[:,i]=np.matmul(U[:,:,i].copy(),A[:,0])
    for i in np.arange(kmax+1,N):
        for j in range(1,kmax+1):       
            U[:,:,i]=np.add(U[:,:,i],np.matmul(Memory[:,:,j],U[:,:,i-j].copy()),out=U[:,:,i])
        A[:,i]=np.matmul(U[:,:,i],A[:,0])
    
    return A