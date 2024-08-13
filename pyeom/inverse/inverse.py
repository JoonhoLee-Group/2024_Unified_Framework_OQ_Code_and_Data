import math as math
import numpy as np
from pyeom.dyck.dyck import *
from scipy.interpolate import interp1d

def etafromI(I_val,I0_val,kmax):
    etakkk=np.zeros(kmax+1,dtype='complex')
    for i in range(kmax+1):
        etakkk[i]=1/4*np.log(I_val[1,2,i])+1/4*np.log(I_val[1,0,i])
    etakkk[0]=-1/2*np.log(I0_val[0])-1/2*np.log(I0_val[1])    
    return etakkk

def UfromA(A1,A2,A3,A4):
    U=np.zeros((4,4,len(A1[0,:])),dtype='complex') 
    rhor = np.zeros((4,4,len(A1[0,:])),dtype='complex')
    U0=np.transpose(np.array([A1[:,0],A2[:,0],A3[:,0],A4[:,0]],dtype='complex'))
    U0inv = np.linalg.inv(U0)

    for i in range(1,len(A1[0,:])): #this fits the propagators from time series data
        rhor[:,0,i]=A1[:,i]
        rhor[:,1,i]=A2[:,i]
        rhor[:,2,i]=A3[:,i]
        rhor[:,3,i]=A4[:,i]
        U[:,:,i] = (np.matmul(rhor[:,:,i],U0inv))
    return U

def KfromU(UU,kmax):
    Memory=np.zeros((4,4,kmax+1),dtype = 'complex')
    Memory[:,:,1]=UU[:,:,1]
    for k in range(2,kmax+1):
        tempM=np.zeros((4,4),dtype = 'complex')
        for j in range(1,k):
            tempM += np.matmul(Memory[:, :, j],UU[:, :, k - j])
        Memory[:,:,(k)]=(UU[:,:,(k)]-tempM)
    return Memory

def IfromK(K,kmax,G):
    I_val=np.zeros((4,4,kmax+1),dtype = 'complex')
    #I_val=Iactual.copy()
    I0_val=np.diag(np.linalg.inv(G) @ (np.array(K[:,:,0])) @ np.linalg.inv(G))
    for i in range(4):
        for j in range(4):
            I_val[i,j,0]+=1
    for k in range(1,kmax+1):
        dyckwords=wordgenerate(k)
        #print(dyckwords)
        
        KKK=np.zeros((4,4),dtype='complex')
        for i in dyckwords[1:]:
            KKK+= dycktocor(i,G,I_val,I0_val)
        #print(KKK)
        #print(K[:,:,k])
        numerator=(np.linalg.inv(G) @ K[:,:,k] @ np.linalg.inv(G) -np.linalg.inv(G) @ KKK @ np.linalg.inv(G))
        denum=np.linalg.inv(G)@dycktocord(dyckwords[0],G,I_val[:,:,:],I0_val)@ np.linalg.inv(G)
        
        #print(denum)
        for i in range(4):
            for j in range(4):
                I_val[i,j,k]=numerator[i,j]/denum[i,j]
        for i in range(4):
            for j in range(4):
                I_val[i,j,k]+=1
    return I0_val, I_val
def Jfrometa(eta,kmax,beta,delt,ww):
    #eta[0]=eta[1]
    etakkd=np.append(np.conjugate(np.flip(eta)),eta[1:])
    
    
    x=np.arange(-kmax+1,kmax,dtype='complex')

    wlim=len(ww)
    F=np.zeros(wlim,dtype='complex')
    JJ=np.zeros(wlim,dtype='complex')
    
    
    #yy[iii[0]]=1/(2*np.pi)*J(om[iii[0]])/(om[iii[0]]**2)*(1+coth(beta*hbar*om[iii[0]]/2))*(1-np.exp(-1j*om[iii[0]]*delt))

    interpolated_function = interp1d(x, etakkd, kind='linear')

    # Define the new x-coordinates where you want to interpolate values
    new_x = np.linspace(min(x),max(x),10000,dtype=complex)
    
    # Use the interpolation function to get interpolated values
    interpolated_y = interpolated_function(new_x)
    yy=new_x.copy()
    
    for w in  enumerate(ww):
        for iii in enumerate(new_x):
            yy[iii[0]]= interpolated_y[iii[0]]*np.exp(1j*delt*iii[1]*w[1])
        F[w[0]]=trapz(yy,new_x)/(2*np.pi)
        JJ[w[0]]=F[w[0]]*np.pi*w[1]**2 *np.sinh(beta*w[1]/2)/(2*np.sin(w[1]*delt/2)**2*np.exp(beta*w[1]/2))*(delt)
    return JJ
