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
####Modify this code (and the from data code) accordingly to produce Figs. S3 and S6


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
    etakkdp=np.zeros(kmax+1,dtype='complex')
    etakkdm=np.zeros(kmax+1,dtype='complex')
    for iii in enumerate(om):
        #np.exp(beta*hbar*om[iii[0]]/2)/(math.sinh(beta*hbar*om[iii[0]]/2))
        yy[iii[0]]=1/(2*np.pi)*J(om[iii[0]])/(om[iii[0]]**2)*(1+coth(beta*hbar*om[iii[0]]/2))*(1-np.exp(-1j*om[iii[0]]*delt))
    etanul=trapz(yy,om)
    for difk in np.linspace(1,kmax,kmax,dtype='int'):

        for i in enumerate(om):
            #np.exp(beta*hbar*om[i[0]]/2)/(math.sinh(beta*hbar*om[i[0]]/2))
            yy[i[0]]=2/(1*np.pi)*J(om[i[0]])/(om[i[0]]**2)*(1+coth(beta*hbar*om[i[0]]/2))*((math.sin(om[i[0]]*delt/2))**2)*np.exp(-1j*om[i[0]]*delt*(difk))
        etakkdp[difk]=trapz(yy,om)
    for difk in np.linspace(-kmax,-1,kmax+1,dtype='int'):

        for i in enumerate(om):
            #np.exp(beta*hbar*om[i[0]]/2)/(math.sinh(beta*hbar*om[i[0]]/2))
            yy[i[0]]=2/(1*np.pi)*J(om[i[0]])/(om[i[0]]**2)*(1+coth(beta*hbar*om[i[0]]/2))*((math.sin(om[i[0]]*delt/2))**2)*np.exp(- 1j*om[i[0]]*delt*(difk))
        etakkdm[difk]=trapz(yy,om)
        
    onetwo=[1,-1]
    for i in enumerate(om):
            #np.exp(beta*hbar*om[i[0]]/2)/(math.sinh(beta*hbar*om[i[0]]/2))
        yy[i[0]]=2/(1*np.pi)*J(om[i[0]])/(om[i[0]]**2)*(1+coth(beta*hbar*om[i[0]]/2))*((math.sin(om[i[0]]*delt/2))**2)*np.exp(- 1j*om[i[0]]*delt*(0))
        etakkdnul=trapz(yy,om)
    #etakkdp[0]=etakkdnul
    etakkdp[0]=etakkdp[1]
    def Influencediff(dx1,dx2,dx3,dx4,diff):
        x1=onetwo[dx1]
        x2=onetwo[dx2]
        x3=onetwo[dx3]
        x4=onetwo[dx4]
        Sum=-1/hbar*(x3-x4)*(etakkdp[diff]*x1-np.conjugate(etakkdp[diff])*x2)     # eq 12 line 1 Nancy quapi I           
        return np.exp(Sum)

    def Influencenull(dx1,dx2): # eq 12 line 2 Nancy quapi I    
        x1=onetwo[dx1]
        
        x2=onetwo[dx2]
        Sum=-1/hbar*(x1-x2)*((etanul)*x1-np.conjugate(etanul)*x2)                
        return np.exp(Sum)
        

    def binseq(k):
        return [''.join(x) for x in itertools.product('0123', repeat=k)] #all possible paths 4^k         


    I0_val=np.array([Influencenull(0,0),Influencenull(0,1),Influencenull(1,0),Influencenull(1,1)], dtype = "complex")
    I_val=np.zeros((4,4,kmax+1),dtype = 'complex')
    A=np.zeros((4,len(tarr)),dtype = 'complex')
    A[0,0]=1
    A[3,0]=0
    P_val=np.zeros((4,4),dtype = 'complex')


    for i in np.arange(0,kmax+1):
        I_val[0,0,i]=Influencediff(0,0,0,0,i)
        I_val[0,1,i]=Influencediff(1,0,0,0,i)
        I_val[0,2,i]=Influencediff(0,1,0,0,i)
        I_val[0,3,i]=Influencediff(1,1,0,0,i)

        I_val[1,0,i]=Influencediff(0,0,1,0,i)
        I_val[1,1,i]=Influencediff(1,0,1,0,i)
        I_val[1,2,i]=Influencediff(0,1,1,0,i)
        I_val[1,3,i]=Influencediff(1,1,1,0,i)

        I_val[2,0,i]=Influencediff(0,0,0,1,i)
        I_val[2,1,i]=Influencediff(1,0,0,1,i)
        I_val[2,2,i]=Influencediff(0,1,0,1,i)
        I_val[2,3,i]=Influencediff(1,1,0,1,i)

        I_val[3,0,i]=Influencediff(0,0,1,1,i)
        I_val[3,1,i]=Influencediff(1,0,1,1,i)
        I_val[3,2,i]=Influencediff(0,1,1,1,i)
        I_val[3,3,i]=Influencediff(1,1,1,1,i)
    etakkd=[]
    etakkd=np.append(etakkdm[1:],2*etanul)
    #etakkd=np.append(etakkdm[1:],etakkdp)
    etakkd=np.append(etakkd,etakkdp[1:])
    x=np.arange(-kmax,kmax+1,dtype='complex')
    wlim=111
    F=np.zeros(wlim,dtype='complex')
    JJ=np.zeros(wlim,dtype='complex')
    yyy=x.copy()
    ww=np.linspace(0,wcut+15,wlim)
    #yy[iii[0]]=1/(2*np.pi)*J(om[iii[0]])/(om[iii[0]]**2)*(1+coth(beta*hbar*om[iii[0]]/2))*(1-np.exp(-1j*om[iii[0]]*delt))

    interpolated_function = interp1d(x, etakkd, kind='linear')

    # Define the new x-coordinates where you want to interpolate values
    new_x = np.linspace(min(x),max(x),10000,dtype='complex')

    # Use the interpolation function to get interpolated values
    interpolated_y = interpolated_function(new_x)
    yyy=new_x.copy()
    for w in  enumerate(ww):
        for iii in enumerate(new_x):
            yyy[iii[0]]= interpolated_y[iii[0]]*np.exp(1j*delt*iii[1]*w[1])
        F[w[0]]=trapz(yyy,new_x)/(2*np.pi)
        JJ[w[0]]=F[w[0]]*np.pi*w[1]**2 *np.sinh(beta*w[1]/2)/(2*np.sin(w[1]*delt/2)**2*np.exp(beta*w[1]/2))*(delt)
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
#plt.savefig(f"Fig32{kondo}{ohmicity}.eps",format='eps')
plt.show()
