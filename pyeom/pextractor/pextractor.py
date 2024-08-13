import numpy as np
def pextractor_exact(rho):
    A0inv = np.linalg.inv(rho[:,:,0])
    U=np.zeros(len(rho[:,0,0]),len(rho[:,0,0]),len(rho[0,0,:]))
    for i in range(1,len(rho[0,0,:])): #this fits the propagators from time series data
        U[:,:,i] = (np.matmul(rho[:,:,i],A0inv))
    return U