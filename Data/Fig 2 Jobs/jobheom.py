import numpy as np
import engine.mps as tn
from engine.heom import *

import numpy as np
import scipy as sp
import h5py 
import sys
import os
#current function used to define the support points for the aaa algorithm. The best choice of support points will depend on the nature of the spectral function you are considering (e.g. where are the interesting features of the spectral function, discontinuities, derivative discontinuities, sharp peaks)
#This choice works rather well for exponential cutoffs with challenging points at zero but won't work well for generic spectral densities.
def generate_grid_points(N, wc, wmin=1e-8):
    Z1 = softmspace(wmin, 20*wc, N)
    nZ1 = -np.flip(Z1)
    Z = np.concatenate((nZ1, Z1))
    return Z

def setup_heom_correlation_functions(Sw, Z1, nmax = 500, aaa_tol = 1e-4):
    #first compute the aaa decomposition of the spectral function
    func1, p, r, z = aaa.AAA_algorithm(Sw, Z1, nmax=nmax, tol=aaa_tol)
    
    #and convert that to the heom correlation function coefficients
    dk, zk = AAA_to_HEOM(p, r)

    #return the function for optional plotting as well as the coefficients
    return func1, dk, zk

def heom_dynamics(rho0, Hsys, S, Sw, dt, tmax, nbose, Lmin, chimax, wmax, tol = 1e-6, chi_mpo_max = None, mpo_tol = 1e-8, aaa_tol = 1e-4, Naaa = 1000, plot_intermediate = False, plot_aaa = False, yrange=None):
    nhilb = Hsys.shape[0]
    nliouv = nhilb*nhilb


    Lsys = commutator(Hsys)
    Us = sp.linalg.expm(-1.0j*dt*Lsys)

    #Z1 are the set of support points used in the aaa algorithm.  To change the set of support points simply change the definition of Z1
    Z1 = generate_grid_points(Naaa, wmax)
    print("Constructing bath correlation functions", file=sys.stderr)
    Sw_aaa, dk, zk = setup_heom_correlation_functions(Sw, Z1, nmax=500, aaa_tol=aaa_tol)


    nt = int(tmax/dt)+1

    #now build the propagator matrices for HEOM
    print("Build HEOM propagator matrices", file=sys.stderr)
    prop = build_propagator_matrices(S, dk, zk, dt, nbose, Lmin=Lmin)
     
    Nmodes = len(prop)+1
    d = np.ones(Nmodes, dtype = int)*nbose
    for i, Uk in enumerate(prop):
        d[i+1] = Uk.shape[0]//Lsys.shape[0]
    d[0] = Lsys.shape[0]


    #setup the ado mps
    A = tn.mps(chi = np.ones(len(prop), dtype=int), d=d, init='zeros', dtype=np.complex128)

    A[0][0, :, 0] = rho0.flatten()
    for i in range(1, len(A)):
        A[i][0, 0, 0] = 1.0
    A.orthogonalise()

    print("Build HEOM propagator MPO", file=sys.stderr)
    #and build it into an MPO
    Uf, Ub = build_propagator(prop, A, method='naive', tol = mpo_tol, nbond = chi_mpo_max)

    rhos = np.zeros((nt+1, nliouv), dtype = np.complex128)

    rhos[0, :] = rho0.flatten()

    fig = None
    ax = None
    line = None
    num = None

    print(0, end=' ' )
    for j in range(rhos.shape[1]):
        print(np.real(rhos[0, j]), np.imag(rhos[0, j]), end=' ')
    print(A.maximum_bond_dimension())

    n = 1
    for i in range(nt):
        apply_mpo_propagator(Us, Uf, Ub, A, method='naive', tol=tol, nbond=chimax)
        n *= np.sqrt(A.normalise())
        rho = extract_rho(A)*n
        rhos[i+1, :] = rho
        print((i+1)*dt, end=' ' )
        for j in range(len(rho)):
            print(np.real(rho[j]), np.imag(rho[j]), end=' ')
        print(A.maximum_bond_dimension())
        sys.stdout.flush()

    return rhos


def main():
    sx = np.array([[0, 1], [1, 0]], dtype = np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype = np.complex128)

    #setup the system Hamiltonian
    eps = 0.0
    delta = 1.0
    Hsys = eps*sz + delta*sx

    #and the system part of the system bath coupling operator
    S = sz

    #function defining the bath spectral function
    beta = 10
    s = 0.5
    wc = 15
    alpha = float(sys.argv[1])

    def Sw(w):
        if beta == None:
            return np.sign(w)*np.pi/2.0*alpha*wc*np.power(np.abs(w)/wc, s)*np.exp(-np.abs(w)/wc)*np.where(w > 0, 1.0, 0.0)
        else:
            return np.sign(w)*np.pi/2.0*alpha*wc*np.power(np.abs(w)/wc, s)*np.exp(-np.abs(w)/wc)*0.5*(1+1.0/np.tanh(beta*w/2.0))


    #the initial value of the system density operator
    rho0 = np.zeros((2,2), dtype=np.complex128)
    rho0[0, 0] = 1.0

    #set up the evolution parameters
    dt = 0.005
    tmax = 5.0

    #setup the ados parameters - accuracy of representation of auxiliary density operator
    nbose = 24
    Lmin = 6
    tol = 1e-8
    chimax = 90

    #setup the heom propagator MPO parameters - accuracy of the representation of the short time propagator
    chi_mpo_max = 4
    mpo_tol = 1e-10

    #setup the aaa parameters - controls accuracy of representation of bath correlation function 
    #convergence with respect to the aaa_tol parameter depends significantly on the choice of support points used, and also on the form of the bath correlation function.  
    #Here I am using a softmspace grid with a rather tight zero.  This approach seems to work quite well for baths with an exponential cutoff at zero temperature but doesn't necessarily work the best for e.g. Debye baths.
    #The choice of support points is somewhat an art and the accuracy can really only be determined by monitoring both the accuracy of the aaa fitting of the spectral function and also the correlation function.  
    #To change the 
    aaa_tol = 1e-7
    wmax = wc
    
    rhos = heom_dynamics(rho0, Hsys, S, Sw, dt, tmax, nbose, Lmin, chimax, wmax, tol=tol, chi_mpo_max=chi_mpo_max, mpo_tol=mpo_tol, aaa_tol = aaa_tol, plot_intermediate = False, yrange=[0.990, 1], plot_aaa=False)
    mypath = f"heomdynamicssubohmic{tmax}"
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    f = h5py.File(f"{mypath}/tmax{tmax}beta{beta}alpha{alpha}s{s}.h5", "w")  
    f["rho"]=rhos.copy()
    f.close()
if __name__ == "__main__":
    main()