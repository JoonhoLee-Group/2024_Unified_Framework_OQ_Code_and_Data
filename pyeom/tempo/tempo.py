from scipy.integrate import quad
from scipy.linalg import expm
import numpy as np
from mpmath import coth, cosh, mp, sinh, sin, cos
import time

import matplotlib.pyplot as plt


import matplotlib.animation  as animation

def commutator(L):
    return np.kron(L, np.identity(L.shape[0])) - np.kron(np.identity(L.shape[0]), L.T)

def anti_commutator(L):
    return np.kron(L, np.identity(L.shape[0])) + np.kron(np.identity(L.shape[0]), L.T)
#mp.dps = 30

class tempo:
    def __init__(self, Sz, Jw, dt = None, T=0):
        self.S = Sz
        self.J = Jw
        self.T = T
        self.etas = []
        i, j = np.nonzero(Sz)
        if not np.all(i == j):
            raise ValueError("Input S matrix is not diagonal")      
        self.comm = np.zeros(Sz.shape[0]*Sz.shape[1], dtype=np.complex128) 
        self.acomm = np.zeros(Sz.shape[0]*Sz.shape[1], dtype=np.complex128) 
        for i in range(Sz.shape[0]):
            for j in range(Sz.shape[1]):
                self.comm[i*Sz.shape[0]+j] = Sz[i, i] - Sz[j, j]
                self.acomm[i*Sz.shape[0]+j] = Sz[i, i] + Sz[j, j]

    def I0(self):
        eta = self.etas[1]
        Ik = np.exp(-self.comm*(self.comm*np.real(eta) + self.acomm*1.0j*np.imag(eta)))
        return Ik

    def Ik(self, ind):
        eta = self.etas[ind+1] - 2.0*self.etas[ind] + self.etas[ind-1]
        C1, AC = np.meshgrid(self.comm, self.acomm)
        Ik = np.exp(-C1*(C1.T*np.real(eta) + AC*1.0j*np.imag(eta)))
        return Ik

    def discretise(self, K, dt):
        self.etas = [self.eta(x*dt) for x in range(0, K+3)]
        print(self.etas)
                

    def zeta(self, w, t):
        R = w**(-2) * self.J(w)*((1-np.cos(w*t)) + 1.0j*(np.sin(w*t)-w*t))
        return R

    def zetaT(self, w, t):
        return w**(-2) * self.J(w)*((1-cos(w*t))*coth(w/(2*self.T)) + 1.0j*(sin(w*t)-w*t))

    def etaRe(self, t):
        return quad(lambda w : np.real(self.zeta(w, t)), 0, np.inf, limit=1000)[0]/np.pi

    def etaReT(self, t):
        return quad(lambda w : np.real(self.zetaT(w, t)), 0, np.inf, limit=10000)[0]/np.pi

    def etaIm(self, t):
        return quad(lambda w : np.imag(self.zeta(w, t)), 0, np.inf, limit=1000)[0]/np.pi

    def etaImT(self, t):
        return quad(lambda w : np.imag(self.zetaT(w, t)), 0, np.inf, limit=10000)[0]/np.pi

    def eta(self, t):
        if(self.T==0):
            return self.etaRe(t) + 1.0j*self.etaIm(t)
        else:
            return self.etaReT(t) + 1.0j*self.etaImT(t)


def close_to_any(a, vals, **kwargs):
    return np.any(np.isclose(a, vals, **kwargs))

def sigmas(rho):
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1.0j], [1.0j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype = np.complex128)
    return [np.real(np.sum(sx.flatten()*rho.flatten())), np.real(np.sum(sy.flatten()*rho.flatten())), np.real(np.sum(sz.flatten()*rho.flatten()))]

def main(lmax):
    K = 200
    D = 2
    D2 = D**2

    T = 0.0
    delta = 1.0
    eps = 0.0

    alpha = 2
    wc = 25
    s = 1

    dt = 0.01
    nt =2500

    tol = 1e-9
    nbond = lmax
    etas = np.zeros(K)

    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype = np.complex128)

    H0 = (delta*sx + eps*sz)

    L0 = np.kron(H0, np.identity(D)) - np.kron(np.identity(D), H0.T)
    Us = expm(-1.0j*dt/2.0*L0)
    Usf = expm(-1.0j*dt*L0)

    def Jw(w):
        return np.pi/2.0*wc*alpha*np.power(w/wc, s)*np.exp(-w/wc)

    tempo_obj = tempo(sz, Jw, T=T)
    tempo_obj.discretise(K, dt)

    rho0 = np.zeros((D, D))
    rho0[0, 0] = 1.0

    I0 = tempo_obj.I0()

    A = tempo_adt.adt(rho0.flatten())

    A = tempo_adt.adt((Us@ (I0*rho0.flatten())))
    Iks = []

    ts = []
    szs = []
    tr=[]
    ts.append(0)
    szs.append(igmas(rho0))
    tr.append(np.real(np.sum(rho0.flatten())))
    print(ts[-1], szs[-1][0], szs[-1][1], szs[-1][2], 1, 1, 1)


    fig, ax = plt.subplots()
    line = ax.plot([], [])[0]
    ax.set(xlim=[0, nt*dt], ylim=[0, 1])


    I1 = None
    #for i in range(nt):
    def update(i):
        t1 = time.time()
        if(i+1 == len(ts)):
            I1 = tempo_obj.Ik(1)*I0
            
            if(i == 0):
                A.apply_IF0(I0, Us)
            else:
            #grow the influence functional object if we haven't reached the memory truncation criterion
                if(i == 1):
                    I1 = tempo_obj.Ik(i)*I0
                    Iks.append((I1*Usf))
                else:
                    if(i <= K ):
                        Iks.append(tempo_obj.Ik(i))
                Iks[0] = (Usf*I1)

                #now we apply the influence functional to this object
                A.apply_IF(Iks, method="naive", tol=tol, nbond=nbond)

            #truncate if we reach the memory truncation criterion
            if(len(A) > K):
                A.terminate()

            rho = A.rho()
            rhot = Us@rho

            ts.append(dt*(i+1))
            szs.append(sigmas(rhot))

        t2 = time.time()
        sza = np.array(szs[:(i+1)])
        line.set_xdata(ts[:(i+1)])
        line.set_ydata(sza[:,0]*sza[:,0] + sza[:,1]*sza[:,1] + sza[:, 2]*sza[:, 2])
        print(ts[-1], szs[-1][0], szs[-1][1], szs[-1][2], A.maximum_bond_dimension(), len(A), t2-t1)
        return line,

    ani = animation.FuncAnimation(fig=fig,func=update, frames=nt, interval=10)
    plt.show()
    #plt.plot(ts,np.array(szs)[:,0])
    #plt.plot(ts,np.array(szs)[:,1])
    #plt.plot(ts,np.array(szs)[:,2])
    #plt.plot(ts, tr)

if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main(200)
    #main(16)
    #main(32)
    #main(64)
    #main(128)
    #pr.disable()
    plt.show()
    #pr.print_stats()
