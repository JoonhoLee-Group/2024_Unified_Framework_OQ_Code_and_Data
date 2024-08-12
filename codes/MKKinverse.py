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
filename = f'ttt/ttt/heomdynamicssubohmicpapertrueyes0.110.0/tmax10.0beta5alpha0.5s0.5initial1.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A1 = np.transpose(h5["rho"])

filename = f'ttt/ttt/heomdynamicssubohmicpapertrueyes0.110.0/tmax10.0beta5alpha0.5s0.5initial2.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A2 = np.transpose(h5["rho"])

filename = f'ttt/ttt/heomdynamicssubohmicpapertrueyes0.110.0/tmax10.0beta5alpha0.5s0.5initial3.h5'

h5 = h5py.File(filename,'r')
#kmax=14
A3 = np.transpose(h5["rho"])

filename = f'ttt/ttt/heomdynamicssubohmicpapertrueyes0.110.0/tmax10.0beta5alpha0.5s0.5initial4.h5'

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
t=np.arange(0,0.1*(mmax+1),0.1)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(t[1:],Mex[1,2,1:],color="k", ls="solid",label="$\mathcal{K}_{12}^N$")
ax.plot(t[1:],Mex[0,0,1:],color=(0.6,0.6,0.6), ls="solid",label="$\mathcal{K}_{00}^N$")
ax.plot(t[1:],Mex[0,3,1:],color=(0.2,0.2,0.2), ls="solid",label="$\mathcal{K}_{03}^N$")

plt.show()