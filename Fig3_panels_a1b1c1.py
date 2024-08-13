

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
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
N=200
wantdelt=0.1
t=np.linspace(0,wantdelt*(N-1),num=N)

Idval=np.array([[1,0],[0,1]],dtype = 'complex')
Hs=np.array([[0,1],[1,0]],dtype = 'complex')
delt=wantdelt
G=scln.expm(np.kron(1j*Hs*delt/2,Idval)+np.kron(Idval,(-1j*Hs*delt/2)))
##Change panels here
panel='b'
if panel=='c':
    filename = f'Data\Fig 3 Data\jobgqme0.50.5\dyn0.5beta5delt0.1ohmicity0.5kmax14.h5'
if panel=='b':
    filename = f'Data\Fig 3 Data\jobgqme0.51.0\dyn0.5beta5delt0.1ohmicity1.0.h5'
if panel=='a':
    filename = f'Data\Fig 3 Data\jobgqme0.11.0\dyn0.1beta5delt0.1ohmicity1.0.h5'

h5 = h5py.File(filename,'r')
kmax=14
#MK=np.zeros((4,4,kmax+1),dtype='complex')
M = h5["M"]  # VSTOXX futures data
A = np.array(h5["13AT"])
U = h5["U"]
MemoryTTMold=KfromU(U,kmax)
MemoryTTMold=M
#plt.plot(MemoryTTMold[1,1,:])
#plt.show()

ax.set_facecolor((0.9,1,1))
UU=np.zeros((4,4,N),dtype = 'complex')
#UU[:,:,1]=MemoryTTMold[:,:,1]
UU=np.array(U)
for i in range(1,N):
    UU[:,:,i]=np.matmul(np.linalg.inv(G),np.matmul(U[:,:,i],np.linalg.inv(G)))

for k in np.arange(1,kmax+1):
    UUU=np.zeros((4,4,N),dtype = 'complex')
    UUU[:,:,0:k+1]=UU[:,:,0:k+1]
    for i in np.arange(k+1,N):
        for j in range(1,k+1):       
            UUU[:,:,i]=np.add(UUU[:,:,i].copy(),np.matmul(MemoryTTMold[:,:,j],UUU[:,:,i-j].copy()),out=UUU[:,:,i])
        A[:,i]=np.matmul(UUU[:,:,i],A[:,0])
    for i in range(1,N):  
        #UU[:,:,i]=np.matmul(P_one,np.matmul(UU[:,:,i],P_one))
        A[:,0]=[1,0,0,0]
        A[:,i]=np.matmul(UUU[:,:,i],A[:,0])
    if k==1:
        ax.plot(t,A[0,:]-A[3,:],color=(1-k/kmax,1-k/kmax,1-k/kmax),label=f"Dyck, order={1}")
    if k==kmax:
        ax.plot(t,A[0,:]-A[3,:],color=(1-k/kmax,1-k/kmax,1-k/kmax),label=f"Dyck, order={kmax}")
    else:
        ax.plot(t,A[0,:]-A[3,:],color=(1-k/kmax,1-k/kmax,1-k/kmax))
#for i in range(N):
#    UU[:,:,i]=np.matmul(np.linalg.inv(G),np.matmul(U[:,:,i],np.linalg.inv(G)))
#    A[:,i]=(np.matmul(UU[:,:,i],[1,0,0,0]))
#for k in range(2,kmax+1):
#    tempM=np.zeros((4,4),dtype = 'complex')
#    for j in range(1,k):
#        tempM += np.matmul(M[:, :, j],UU[:, :, k - j])
#    (UU[:,:,(k)])=M[:,:,(k)]+tempM
#for i in range(1,kmax+1):
#    UU[:,:,i]=np.matmul(np.linalg.inv(G),np.matmul(UU[:,:,i],np.linalg.inv(G)))
#MK=KfromU(UU,kmax)
#Mex=h5["Mex"]
#ax.plot(np.arange(2,kmax+1)*wantdelt,Mex[1,2,2:]/(delt),color="k", ls="solid",label="Dyck/INFPI $\mathcal{K}_{12}^N$")
#ax.plot(np.arange(2,kmax+1)*wantdelt,Mex[0,0,2:]/(delt),color=(0.6,0.6,0.6), ls="solid",label="Dyck/INFPI $\mathcal{K}_{00}^N$")
#ax.plot(np.arange(2,kmax+1)*wantdelt,Mex[0,3,2:]/(delt),color=(0.2,0.2,0.2), ls="solid",label="Dyck/INFPI $\mathcal{K}_{03}^N$")



#filename = f'GQMETEST/wrongdyn0.7853981633974483beta5delt0.1.h5'

h5.close()

plt.xlabel("Time",fontsize=15)
ax.set_ylabel(r"$\sigma_z$",fontsize=15)
#ax.set_ylim([-0.2,1])
ax.set_xlim([0,15-0.1])

#plt.legend()
#plt.savefig("plots/K051.png",dpi=1000)
#plt.show()



























































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
filename = f'Data\heomforK0.015.0/tmax5.0beta5alpha0.5s0.5initial1.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A1 = np.transpose(h5["rho"])

filename = f'Data\Fig 3 Data\heomforK0.015.0/tmax5.0beta5alpha0.5s0.5initial2.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A2 = np.transpose(h5["rho"])

filename = f'Data\Fig 3 Data\heomforK0.015.0/tmax5.0beta5alpha0.5s0.5initial3.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A3 = np.transpose(h5["rho"])

filename = 'Data\Fig 3 Data\heomforK0.015.0/tmax5.0beta5alpha0.5s0.5initial4.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A4 = np.transpose(h5["rho"])

Uex=UfromA(A1,A2,A3,A4)
A5=A1.copy()
for i in range(1,len(A1)):  
    A5[:,i]=np.matmul(Uex[:,:,i],[1,0,0,0])
mmax=50
Mex=KfromU(Uex,mmax)
plt.rc("font", family="serif")
plt.rc("xtick")
plt.rc("ytick")
t=np.arange(0,0.01*(mmax+1),0.01)
ddelt=0.01
#ax.plot(t[2:],Mex[1,2,2:]/(ddelt),color=(0,0,0),alpha=1, ls="dashed",label="HEOM $\mathcal{K}_{12}^N$")
#ax.plot(t[2:],Mex[0,0,2:]/(ddelt),color=(0.6,0.6,0.6),alpha=1, ls="dashed",label="HEOM $\mathcal{K}_{00}^N$")
#ax.plot(t[2:],Mex[0,3,2:]/(ddelt),color=(0.2,0.2,0.2),alpha=1, ls="dashed",label="HEOM $\mathcal{K}_{03}^N$")

#plt.savefig("plots/K051HEOM.png",dpi=1000)

#filename = f'tmax15.0beta5alpha0.5s0.5initial1.h5'
if panel=='c':
    filename="Data\Fig 3 Data\heomforK0.145.00.5/tmax45.0beta5alpha0.5s0.5initial1.h5"

if panel=='b':
    filename="Data\Fig 3 Data\heompaper10.115.0/tmax15.0beta5alpha0.5s1initial1.h5"
if panel=='a':
    filename="Data\Fig 3 Data\heompaper10.115.0/tmax15.0beta5alpha0.1s1initial1.h5"

h5 = h5py.File(filename,'r')
#kmax=14
mmax=150
A5 = np.transpose(h5["rho"])
plt.plot(np.arange(0,0.1*(mmax+2),0.1),A5[0,:]-A5[3,:],color='red',ls='dashed',alpha=0.7,label='exact')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.savefig(f"Fig2{panel}1.eps",format='eps')
plt.show()