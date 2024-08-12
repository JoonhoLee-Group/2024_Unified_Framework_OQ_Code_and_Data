import numpy as np
from scipy.sparse import diags
from scipy.linalg import svd

def aaa(F, Z, tol=1e-13, mmax=100):
    M = len(Z)  # number of sample points

    # Convert function handle to vector if F is a function handle
    if not isinstance(F, np.ndarray):
        F = F(Z)

    Z = Z.reshape(-1, 1)
    F = F.reshape(-1, 1)

    # Left scaling matrix
    SF = diags(F[:, 0], format='csc')

    J = np.arange(1, M + 1)
    z = []
    f = []
    C = []
    errvec = []
    R = np.mean(F)

    for m in range(1, mmax + 1):
        _, j = np.argmax(np.abs(F - R))  # select next support point
        z.append(Z[j])
        f.append(F[j])

        # Update index vector
        J = J[J != j]

        # Update Cauchy matrix
        C = np.column_stack([C, 1.0 / (Z - Z[j])])

        # Next column of Cauchy matrix
        Sf = np.diag(f)  # right scaling matrix
        A = SF @ C - C @ Sf  # Loewner matrix

        _, _, V = svd(A[J, :], full_matrices=False)  # SVD
        w = V[:, -1]  # weight vector = min singular vector

        # Numerator and denominator
        N = C @ (w * f)
        D = C @ w

        # Rational approximation
        R = F.copy()
        R[J] = N[J] / D[J]

        # Max error at sample points
        err = np.linalg.norm(F - R, np.inf)
        errvec.append(err)

        # Stop if converged
        if err <= tol * np.linalg.norm(F, np.inf):
            break

    r = lambda zz: rhandle(zz, z, f, w)  # AAA approximant as function handle
    pol, res, zer = prz(r, z, f, w)  # poles, residues, and zeros
 
    return r, pol, res, zer, z, f, w, errvec

def prz(r, z, f, w):
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


# Example usage
# F, Z, tol, mmax are assumed to be defined
def y(x):
    return np.exp(x)

x=np.linspace(0,100)
r, pol, res, zer, z, f, w, errvec = aaa(y(x), x)
