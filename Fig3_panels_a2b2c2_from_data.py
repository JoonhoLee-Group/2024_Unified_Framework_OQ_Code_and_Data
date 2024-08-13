import math as math
import numpy as np
import sympy as sp
from mpmath import coth
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k
import os
import sys
import h5py
import multiprocessing
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
##This code is skipping the rho->U->K because they are exact in principle
###Set kondo and ohmicity below to plot the respective plots

kkmax=15
N=500
wantdelt=0.05

Idval=np.array([[1,0],[0,1]],dtype = 'complex')
beta=5
kondo=0.5
ga=kondo * np.pi / 2
No=500
hbar=1
om=np.linspace(-150,150,No)
yy=np.zeros(No,dtype = 'complex')
tarr = np.linspace(0,wantdelt*(N-1),num=N)
delt=(tarr[1]-tarr[0])

Hs=np.array([[0,1],[1,0]],dtype = 'complex')
wcut = 7.5
wc=wcut
ohmicity=0.5
num_processes = 1
plt.rc("font", family="serif")
plt.rc("xtick")
plt.rc("ytick")
#plt.rc("text", usetex=True)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0.9,1,1))

plt.xlabel("$\omega$",fontsize=15)
plt.ylabel("$J(\omega$)",fontsize=15)
def J(xs):  # Spectral density (Ohmic bath with exponential cutoff)
    output = np.zeros_like(xs)
    output[xs > 0] = ga * (xs**ohmicity)[xs > 0] * np.exp(-xs[xs > 0] / wc)/(wc**(ohmicity-1))
    output[xs <= 0] = - ga * ((-xs)**ohmicity)[xs <= 0] * np.exp(xs[xs <= 0] / wc)/(wc**(ohmicity-1))
    #output=np.heaviside(xs,0.5)ga*(xs**ohmicity)/(wc**(1-ohmicity))*np.exp(-(xs)/wc)
    return output
for kmax in [2,4,6,8,10,12,14,16]:
    #put in the file path here, from jobinverse.py
    mypath = ''
    filename = f"{mypath}/inverse{kondo}beta{beta}delt{delt}kmax{kmax}.h5"

    h5 = h5py.File(filename,'r')
    ww = h5["w"]
    JJ = h5["J"]
    i=kmax
    if kmax==2:
        plt.plot(ww,np.real(JJ),label='extracted (order 2)',color=(1-i/18,1-i/18,1-i/18))#
    elif kmax==16:
        plt.plot(ww,np.real(JJ),label='extracted (order 16)',color=(1-i/18,1-i/18,1-i/18))#
    #elif kmax>20:
        #plt.plot(ww,np.real(JJ),label=f'$\eta$ inverse Fourier (order {kmax})',alpha=0.7,color='blue')#
    else:
        plt.plot(ww,np.real(JJ),color=(1-i/18,1-i/18,1-i/18))#
    #print(beta)
    #print(delt)
plt.plot(ww,np.real(J(ww)),label='analytical',color='red',alpha=0.7,ls='dashed')
plt.xlabel('$\omega$',fontsize=15)
#plt.legend(fontsize=15)
plt.xlim([0,22.5])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"Fig32{kondo}{ohmicity}.eps",format='eps')
plt.show()
