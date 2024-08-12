import scipy.sparse as spp
import numpy as np
from scipy.linalg import svd
from scipy.integrate import quad, quad_vec
from scipy import linalg as splinalg
import matplotlib.pyplot as plt

def argmax(values):
    print(max(values))
    return (max(values)), np.where(values == max(values))

def aaa(F,Z,tol=10**(-13),mmax=100):
# aaa  rational approximation of data F on set Z
#      [r,pol,res,zer,z,f,w,errvec] = aaa(F,Z,tol,mmax)
#  Input: F = vector of data values, or a function handle%         
#Z = vector of sample points%         tol = relative tolerance tol, set to 1e-13 if omitted
#         mmax: max type is (mmax-1,mmax-1), set to 100 if omitted%
# Output: r = AAA approximant to F (function handle)%         pol,res,zer = vectors of poles, residues, zeros%         
#z,f,w = vectors of support pts, function values, weights%         errvec = vector of errors at each step
    M = Z.shape[0]                         # number of sample points
    #if not isinstance(F, float):
    #    F = F(Z)             # convert function handle to vector
    #SF=spp.spdiags(F,0,M,M)
    J=range(1,M)
    z=[]
    f=[]
    C=np.zeros((M,mmax),dtype='complex')
    errvec = []
    w=None
    R = np.mean(F)
    # Main loop
    for m in range(mmax):
        j = np.argmax(np.abs(F-R))  # select next support point
        z.append(Z[j])
        f.append(F[j])
        
        # Update index vector
        #J = np.delete(J, np.where(J == j))
        Z = np.delete(Z, j, 0)
        F = np.delete(F, j, 0)
        C = np.delete(C, j, 0)
        SF = spp.diags(F)
        # Update Cauchy matrix
        #C = np.column_stack([C, 1.0 / (Z - Z[j])])
        C[:, m] = (1.0/(Z-z[m]))
        # Next column of Cauchy matrix
        Sf = np.diag(f)  # right scaling matrix
        
        A = SF @ C[:, :m+1] - C[:, :m+1] @ Sf  # Loewner matrix

        U, S, V = np.linalg.svd(A, full_matrices=False)
        V = np.conjugate(np.transpose(V))
        w = V[:, m]

        # Numerator and denominator
        N = C[:, :m+1] @ (w*f)
        D = C[:, :m+1] @ w

        # Rational approximation
        #R = F.copy()
        R = N / D

        # Max error at sample points
        err = np.linalg.norm(F - R, np.inf)
        errvec.append(err)

        # Stop if converged
        if err <= tol * np.linalg.norm(F, np.inf):
            break
    r = lambda zz: rhandle(zz, z, f, w)  # AAA approximant as function handle
    pol, res, zer = prz(r, z, f, w,tol)  # poles, residues, and zeros
    return r, pol, res, zer, z, f, w, errvec

def residue_integrand(theta, r, rs, poles):
    zvs = rs*np.exp(1.0j*theta)
    vals = r(poles+zvs)*zvs
    return vals

def compute_residues_integ(r,  poles, tol):
    #find the distance between pole i and the nearest pole to it - this ensures that we can evaluate the pole using
    vals = np.abs([xs - poles[np.argpartition(np.abs(poles - xs), 1)[1]] for xs in poles])/10.0
    rs = vals

    res = quad_vec(lambda x : residue_integrand(x, r, rs, poles), 0, 2.0*np.pi, epsrel = tol)[0]
    return res
def prz(r, z, f, w, tol):

    m = w.shape[0]
    
    #setup the generalised eigenproblem suitable for obtaining the poles of this function
    B = np.identity(m+1, dtype = np.complex128)
    B[0, 0] = 0
    M = np.zeros((m+1, m+1), dtype = np.complex128)
    M[1:, 1:] = np.diag(z)
    M[0, 1:] = w
    M[1:, 0] = np.ones(m, dtype=np.complex128)

    poles = splinalg.eigvals(M, B)
    poles = poles[~np.isinf(poles)]

    #setup the generalised eigenproblem suitable for obtaining the zeros of this function
    M[0, 1:] = w*f

    zeros = splinalg.eigvals(M, B)
    zeros = zeros[~np.isinf(zeros)]

    res = compute_residues_integ(r, poles, tol)
    
    return poles, res, zeros
def prz2(r, z, f, w):
    m = len(w)
    B = np.eye(m + 1)
    B[0, 0] = 0
    E = np.block([[0, np.transpose(w)], [np.ones((m, 1)), np.diag(z)]])
    
    # Compute poles
    pol = np.linalg.eigvals(E, B)
    pol = pol[~np.isinf(pol)]

    # Compute residues
    dz = 1e-5 * np.exp(2j * np.pi * np.arange(1, 5) / 4)
    res = r(np.add.outer(pol, dz)) @ dz / 4

    # Compute zeros
    E = np.block([[0, np.multiply(w, f)], [np.ones((m, 1)), np.diag(z)]])
    zer = np.linalg.eigvals(E, B)
    zer = zer[~np.isinf(zer)]

    return pol, res, zer

def rhandle(zz, z, f, w):
    zv = zz.ravel()  # vectorize zz if necessary
    CC = 1.0 / (np.subtract.outer(zv, z))  # Cauchy matrix
    r = (CC @ (w * f)) / (CC @ w)  # AAA approx as vector

    # Find values NaN = Inf/Inf if any
    ii = np.where(np.isnan(r))[0]
    
    # Force interpolation at NaN points
    for j in ii:
        r[j] = f[np.where(zv[j] == z)[0][0]]

    r = r.reshape(zz.shape)  # AAA approx
    return r

def y(w,wc=17.5):
    return 2*w*wc/(w*w+wc*wc)

x=np.linspace(0,100)
[r,pol,res,zer,z,f,w,errvec]=aaa(y(x),x)
plt.plot(x,r(x))
plt.plot(x,y(x),'r--')
print(pol)
print(res)
plt.show()