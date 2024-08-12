def onorm(M):
    MM = np.transpose(M.conj()) @ M
    eigenvalues, eigenvectors = np.linalg.eigh(MM)
    return np.sqrt(np.max(eigenvalues))







import math as math
import numpy as np
import sympy as sp
from mpmath import coth
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k
import h5py
import matplotlib.pyplot as plt
from pyeom.inverse.inverse import *
plt.rc("font", family="serif")
plt.rc("xtick")


plt.rc("ytick")
fig = plt.figure()  # width=8 inches, height=6 inches

ax = fig.add_subplot(1, 1, 1)
N=500
wantdelt=0.1
t=np.linspace(0,wantdelt*(N-1),num=N)

Idval=np.array([[1,0],[0,1]],dtype = 'complex')
Hs=np.array([[0,1],[1,0]],dtype = 'complex')
delt=wantdelt
G=scln.expm(np.kron(1j*Hs*delt/2,Idval)+np.kron(Idval,(-1j*Hs*delt/2)))
#filename = f'jobsdone\jobgqme0.50.5\dyn0.5beta5delt0.1ohmicity0.5kmax14.h5'
kcut=12
filename = f"jobgqmecut20.11.00.1{kcut}\dyn0.1beta5delt0.1ohmicity1.0kmax13.h5"
h5 = h5py.File(filename,'r')
kmax=13
#MK=np.zeros((4,4,kmax+1),dtype='complex')
M = h5["M"]  # VSTOXX futures data
A = np.array(h5["10AT"])
U1 = h5["U"]
print(list(A))
UU=np.zeros((4,4,N),dtype = 'complex')
UU[:,:,1]=M[:,:,1]
Mnorm=np.zeros((kmax+1),dtype = 'complex')
U=np.zeros((4,4,N),dtype = 'complex')
for i in range(1,kmax+1):
    U[:,:,i]=np.matmul(np.linalg.inv(G),np.matmul(U1[:,:,i],np.linalg.inv(G)))
MK=KfromU(U,kmax)
#ax.plot(np.arange(0,14)*wantdelt,MK[1,2,:],color="k", ls="solid",label="Dyck/INFPI $\mathcal{K}_{12}^N$")
#ax.plot(np.arange(0,14)*wantdelt,MK[0,0,:],color=(0.6,0.6,0.6), ls="solid",label="Dyck/INFPI $\mathcal{K}_{00}^N$")
#ax.plot(np.arange(0,14)*wantdelt,MK[0,3,:],color=(0.2,0.2,0.2), ls="solid",label="Dyck/INFPI $\mathcal{K}_{03}^N$")
for i in range(kmax+1):
    Mnorm[i]=onorm(MK[:,:,i]/(delt**2))
#ax.plot(t,A[0,:],color="k")
ax.semilogy(np.arange(2,kmax+1)*wantdelt,Mnorm[2:],color="k", ls="solid",marker='none',markersize=12,label=r'$\|\mathcal{K}_{N}\|$')

h5.close()

ax.set_facecolor((0.9,1,1))
plt.xlabel("Time",fontsize=15)
kcut=8
filename = f"jobgqmecut20.11.00.1{kcut}\dyn0.1beta5delt0.1ohmicity1.0kmax13.h5"
h5 = h5py.File(filename,'r')

#MK=np.zeros((4,4,kmax+1),dtype='complex')
M = h5["M"]  # VSTOXX futures data
A = np.array(h5["10AT"])
print(list(A))
U1 = h5["U"]
UU=np.zeros((4,4,N),dtype = 'complex')
UU[:,:,1]=M[:,:,1]
Mnorm=np.zeros((kmax+1),dtype = 'complex')

for i in range(1,kmax+1):
    U[:,:,i]=np.matmul(np.linalg.inv(G),np.matmul(U1[:,:,i],np.linalg.inv(G)))
MK=KfromU(U,kmax)
#ax.plot(np.arange(0,14)*wantdelt,MK[1,2,:],color="k", ls="solid",label="Dyck/INFPI $\mathcal{K}_{12}^N$")
#ax.plot(np.arange(0,14)*wantdelt,MK[0,0,:],color=(0.6,0.6,0.6), ls="solid",label="Dyck/INFPI $\mathcal{K}_{00}^N$")
#ax.plot(np.arange(0,14)*wantdelt,MK[0,3,:],color=(0.2,0.2,0.2), ls="solid",label="Dyck/INFPI $\mathcal{K}_{03}^N$")
for i in range(kmax+1):
    Mnorm[i]=onorm(MK[:,:,i]/(delt**2))
#ax.plot(t,A[0,:],color="k")
ax.semilogy(np.arange(2,kmax+1)*wantdelt,Mnorm[2:],color="#808080", ls="dashed",marker='none',markersize=12,label=r'$\|\mathcal{K}_{N}\|$')

h5.close()


kcut=4
filename = f"jobgqmecut20.11.00.1{kcut}\dyn0.1beta5delt0.1ohmicity1.0kmax13.h5"
h5 = h5py.File(filename,'r')
#MK=np.zeros((4,4,kmax+1),dtype='complex')
M = h5["M"]  # VSTOXX futures data
A = np.array(h5["10AT"])
U1 = h5["U"]
print(list(A))
UU=np.zeros((4,4,N),dtype = 'complex')
UU[:,:,1]=M[:,:,1]
Mnorm=np.zeros((kmax+1),dtype = 'complex')

for i in range(1,kmax+1):
    U[:,:,i]=np.matmul(np.linalg.inv(G),np.matmul(U1[:,:,i],np.linalg.inv(G)))
MK=KfromU(U,kmax)
#ax.plot(np.arange(0,14)*wantdelt,MK[1,2,:],color="k", ls="solid",label="Dyck/INFPI $\mathcal{K}_{12}^N$")
#ax.plot(np.arange(0,14)*wantdelt,MK[0,0,:],color=(0.6,0.6,0.6), ls="solid",label="Dyck/INFPI $\mathcal{K}_{00}^N$")
#ax.plot(np.arange(0,14)*wantdelt,MK[0,3,:],color=(0.2,0.2,0.2), ls="solid",label="Dyck/INFPI $\mathcal{K}_{03}^N$")
for i in range(kmax+1):
    Mnorm[i]=onorm(MK[:,:,i]/(delt**2))
#ax.plot(t,A[0,:],color="k")
ax.semilogy(np.arange(2,kmax+1)*wantdelt,Mnorm[2:],color="#404040", ls="dotted",marker='x',markersize=3,label=r'order 13, $k_{{max}}=5$')

h5.close()

filename = f'jobsdone/jobgqme0.11.0/dyn0.1beta5delt0.1ohmicity1.0.h5'

h5 = h5py.File(filename,'r')
kmax=14
#MK=np.zeros((4,4,kmax+1),dtype='complex')
M = h5["M"]  # VSTOXX futures data
A = np.array(h5["10AT"])
print(list(A))
UU=np.zeros((4,4,N),dtype = 'complex')
UU[:,:,1]=M[:,:,1]
Mnorm=np.zeros((kmax+1),dtype = 'complex')
for k in range(2,kmax+1):
    tempM=np.zeros((4,4),dtype = 'complex')
    for j in range(1,k):
        tempM += np.matmul(M[:, :, j],UU[:, :, k - j])
    (UU[:,:,(k)])=M[:,:,(k)]+tempM
for i in range(1,kmax+1):
    UU[:,:,i]=np.matmul(np.linalg.inv(G),np.matmul(UU[:,:,i],np.linalg.inv(G)))
MK=KfromU(UU,kmax)
#ax.plot(np.arange(0,14)*wantdelt,MK[1,2,:],color="k", ls="solid",label="Dyck/INFPI $\mathcal{K}_{12}^N$")
#ax.plot(np.arange(0,14)*wantdelt,MK[0,0,:],color=(0.6,0.6,0.6), ls="solid",label="Dyck/INFPI $\mathcal{K}_{00}^N$")
#ax.plot(np.arange(0,14)*wantdelt,MK[0,3,:],color=(0.2,0.2,0.2), ls="solid",label="Dyck/INFPI $\mathcal{K}_{03}^N$")
for i in range(kmax+1):
    Mnorm[i]=onorm(M[:,:,i]/(delt**2))
#ax.plot(t,A[0,:],color="k")
#ax.semilogy(np.arange(2,kmax+1)*wantdelt,Mnorm[2:],color="k", ls="none",markersize=12,marker='.')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.ylabel(r'$\|\mathcal{K}_{N}\|$',fontsize=15)

plt.savefig(f"DiffnormK011",dpi=1000,bbox_inches="tight")
h5.close()
plt.show()