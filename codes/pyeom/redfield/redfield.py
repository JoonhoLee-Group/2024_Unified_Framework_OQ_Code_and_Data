import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from numpy.linalg import eig
from scipy.linalg import expm
##Setup system
def redfield_propagate(rho0,N, delt, beta, ga, ohmicity, wc, Hs, SL): 
    tarr = np.linspace(0,N*delt,num=N)
    T=1/beta
    def J(x): #Spectral density
        return ga*x**ohmicity*np.exp(-x/wc) #Ohmic bath
    def nb(o): #Bose occupation distribution
        return 1/(math.exp(o*T)-1)
    def k(o): #Calculating real part of bath correlation function laplace transform
        if abs(o)<10**-3:
            return ga*T
        elif o<0:
            return J(abs(o))*nb(abs(o))
        else:
            return J(abs(o))*(nb(abs(o))+1)
        
    n=len(Hs)
    
    rhos=np.reshape(rho0,(n,n))
    rhot=np.zeros((n,n,len(tarr)),dtype = 'complex_')
    rett=np.zeros((n**2,len(tarr)),dtype = 'complex_')
    rhoDt=np.zeros((n,n,len(tarr)),dtype = 'complex_')
    R=np.zeros((n,n,n,n),dtype = 'complex_')
    DHs,V=eig(Hs)
    
    S=np.matmul(np.matmul(np.linalg.inv(V),SL),V)
    rhosD=np.matmul(np.matmul(np.linalg.inv(V),rhos),V)
    ##Construct Liouvillian tensor
    nn=np.linspace(0,n-1,n,dtype=np.int8)
    for a in nn:
        for b in nn:
            R[a][b][a][b]=R[a][b][a][b]+1j*(DHs[a]-DHs[b])
            for c in nn:
                for d in nn:
                    R[a,b,d,b]=R[a,b,d,b]-S[a,c]*S[c,d]*k(DHs[d]-DHs[c])/2
                    R[a,b,a,c]=R[a,b,a,c]-S[b,d]*S[d,c]*k(DHs[c]-DHs[d])/2
                    R[a,b,c,d]=R[a,b,c,d]+(S[d,b]*S[a,c]*k(DHs[c]-DHs[a])/2+S[c,a]*S[b,d]*k(DHs[d]-DHs[b])/2)
        
    ##Dynamics
    L=np.reshape(R,(n**2,n**2))
    
    rhosv=np.reshape(rhosD,n**2)
    for tin in enumerate(tarr):
        Lexp=expm(L*tin[1])
        rhoDt[0:,0:,tin[0]]=np.reshape(np.matmul(Lexp,rhosv),(n,n))
        rhot[0:,0:,tin[0]]=np.matmul(np.matmul(V,rhoDt[0:,0:,tin[0]]),np.linalg.inv(V))
        rett[:,tin[0]]=np.reshape(rhot[:,:,tin[0]],n**2)
    return rett
