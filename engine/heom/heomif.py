from .aaa import AAA_algorithm
from .mps import mps

import numpy as np
import scipy as sp


def commutator(L):
    return np.kron(L, np.identity(L.shape[0])) - np.kron(np.identity(L.shape[0]), L.T)

def anti_commutator(L):
    return np.kron(L, np.identity(L.shape[0])) + np.kron(np.identity(L.shape[0]), L.T)


def Lkp(nbose, dk, S, s=0.5):
    b1 = np.zeros((nbose, nbose), dtype=np.complex128)
    
    for i in range(nbose-1):
        b1[i, i+1] = np.sqrt((i+1.0))*np.sqrt(np.abs(dk))

    Scomm = commutator(S)
    return np.kron(Scomm, b1)


def Lkm(nbose, dk, S, mind = True, s=0.5):
    b1 = np.zeros((nbose, nbose), dtype=np.complex128)
    
    coeff = 1
    if not mind:
        coeff = -1.0
    for i in range(nbose-1):
        b1[i+1, i] = coeff*np.sqrt((i+1.0))*dk/np.sqrt(np.abs(dk))

    Sop = None
    if mind:
        Sop = np.kron(S, np.identity(S.shape[0]))
    else:
        Sop = np.kron(np.identity(S.shape[0]), S.T)
    return np.kron(Sop, b1)

def nop(D2, nbose, zk):
    return -1.0j*np.kron(np.identity(D2), np.diag((np.arange(nbose))*zk))


def mode_operator(nbose, dk, zk,  S, mind):
    op = None
    s = 0.5
    if mind:
        op = Lkp(nbose, dk, S, s=s) + Lkm(nbose, dk, S, mind=mind, s= s) + nop(S.shape[0]**2, nbose, zk)
    else:
        op = Lkp(nbose, np.conj(dk), S, s=s) + Lkm(nbose, np.conj(dk), S, mind=mind, s=s) + nop(S.shape[0]**2, nbose, np.conj(zk))
    return op


def Mk(nbose, dk, zk, S, mind, dt, nf=None):
    op = None
    if(nf > nbose):
        op = mode_operator(nf, dk, zk, S, mind)
        s = S.shape[0]*S.shape[1]
        expm = sp.linalg.expm(-1.0j*dt*op).reshape(s, nf, s, nf)
        return expm[:, :nbose, :, :nbose].reshape(s*nbose, s*nbose)
    else:
        op = mode_operator(nbose, dk, zk, S, mind)
        return sp.linalg.expm(-1.0j*dt*op)

def compute_dimensions(S, dk, zk, L, Lmin = None):
    ds = np.ones(2*len(dk)+1, dtype = int)
    ds[0] = S.shape[0]*S.shape[1]

    minzk = np.amin(np.real(zk))
    if(Lmin == None):
        for i in range(len(dk)):
            nb = L
            ds[2*i+1] = nb
            ds[2*i+2] = nb
    else:
        for i in range(len(dk)):
            nb = max(int(L*minzk/np.real(zk[i])), Lmin)
            ds[2*i+1] = nb
            ds[2*i+2] = nb

    return ds

#build the short time HEOM propagator 
def build_propagator_matrices(S, dk, zk, dt, L, Lmin=None, sf = 1):
    ds = compute_dimensions(S, dk, zk, L, Lmin=Lmin)

    Uks = []
    for i in range(len(dk)):
        nb = ds[2*i+1]
        Uks.append(Mk(nb, dk[i], zk[i], S, True, dt/2.0, nf = sf*nb))
        Uks.append(Mk(nb, dk[i], zk[i], S, False, dt/2.0, nf = sf*nb))
    return Uks


def HEOM_bath_propagator(bath, dt):
    Lmin = None
    sf = 1
    if 'Lmin' in bath.keys():
        Lmin = bath['Lmin']
    if 'sf' in bath.keys():
        sf = bath['sf']
    return build_propagator_matrices(bath['S'], bath['d'], bath['z'], dt, bath['L'], Lmin=Lmin, sf=sf)

def HEOM_propagator(baths, dt):
    #if we have a list of baths
    if isinstance(baths, list):
        Uks = []
        for bath in baths:
            Uks = Uks + HEOM_bath_propagator(bath, dt)
        return Uks

    elif isinstance(baths, dict):
        return HEOM_bath_propagator(baths, dt)


def apply_propagator(Us, Uks, A, method='naive', tol=None, nbond = None):
    #apply non local two site gates.  Here we perform swap operations as we go, unless we are applying the last gate
    for i, Uk in enumerate(Uks):
        if(i+1 == len(Uks)):
            A.apply_two_site(Uk, i, i+1, method=method, tol=tol, nbond=nbond)
        else:
            A.apply_bond_tensor_and_swap(Uk, i, dir = 'right', tol=tol, nbond=nbond)

    A.apply_one_site(Us, -2)

    for i, Uk in reversed(list(enumerate(Uks))):
        if(i+1 == len(Uks)):
            A.apply_two_site(Uk, i, i+1, method=method, tol=tol, nbond=nbond)
        else:
            A.apply_bond_tensor_and_swap(Uk, i, dir = 'left', tol=tol, nbond=nbond)

    return A



def setup_HEOM_ados(rho0, prop):
    nliouv = rho0.flatten().shape[0]
    Nmodes = len(prop)+1
    d = np.zeros(Nmodes, dtype = int)
    for i, Uk in enumerate(prop):
        d[i+1] = Uk.shape[0]//nliouv
    d[0] = nliouv

    #setup the ado mps
    A = mps(chi = np.ones(len(prop), dtype=int), d=d, init='zeros', dtype=np.complex128)

    A[0][0, :, 0] = rho0.flatten()
    for i in range(1, len(A)):
        A[i][0, 0, 0] = 1.0
    A.orthogonalise()
    return A


def extract_rho(A):
    T = None
    for i in reversed(range(1, len(A))):
        if not isinstance(T, np.ndarray):
            T = A[i][:, 0, 0]
        else:
            M = A[i][:, 0, :]
            T = M@T

    Mi = A[0][:, :, :]
    T = np.tensordot(Mi, T, axes=([2], [0]))
    return T[0, :]


def Cr(func, t, tol=1e-8, limit=1000, workers=1, dx = 1e-12):
    return sp.integrate.quad_vec(lambda w : np.real(func(w)*np.exp(-1.0j*t*w)), dx, np.inf, limit=1000)[0]/np.pi + sp.integrate.quad_vec(lambda w : np.real(func(w)*np.exp(-1.0j*t*w)), -np.inf, -dx, limit=1000)[0]/np.pi + sp.integrate.quad_vec(lambda w : np.real(func(w)*np.exp(-1.0j*t*w)), -dx, dx, limit=1000, points=[0])[0]/np.pi 

def Ci(func, t, tol=1e-8, limit=1000, workers=1, dx = 1e-12):
    return sp.integrate.quad_vec(lambda w : np.imag(func(w)*np.exp(-1.0j*t*w)), dx, np.inf, limit=1000)[0]/np.pi + sp.integrate.quad_vec(lambda w : np.imag(func(w)*np.exp(-1.0j*t*w)), -np.inf, -dx, limit=1000)[0]/np.pi + sp.integrate.quad_vec(lambda w : np.imag(func(w)*np.exp(-1.0j*t*w)), -dx, dx, limit=1000, points=[0])[0]/np.pi 


