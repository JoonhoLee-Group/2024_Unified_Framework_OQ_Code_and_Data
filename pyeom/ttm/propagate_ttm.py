import math as math

import numpy
import numpy as np

kmax = 5
N = 100
wantdelt = 0.2
Idval = np.array([[1, 0], [0, 1]], dtype="complex")
beta = 5
ga = 0.1 * np.pi / 2
No = 500
hbar = 1
om = np.linspace(-150, 150, No)
yy = np.zeros(No, dtype="complex")
tarr = np.linspace(0, wantdelt * (N - 1), num=N)
delt = tarr[1] - tarr[0]
print(delt)
Hs = np.array([[1, -1], [-1, -1]], dtype="complex")
wcut = 7.5

from pyeom.quapi.quapi_change import quapi_propagate

nomega = No
omega = numpy.linspace(-50, 50, nomega)

Hs = numpy.array([[1, 1], [1, -1]], dtype="complex")

ohmicity = 1
wc = 7.5
A1 = quapi_propagate([1, 0, 0, 0], N, delt, omega, beta, ga, ohmicity, wc, Hs, kmax)
A2 = quapi_propagate(
    [1 / 2, 1 / 2, 1 / 2, 1 / 2], N, delt, omega, beta, ga, ohmicity, wc, Hs, kmax
)
A3 = quapi_propagate(
    [1, 1 / 2 + 1 / 2j, 1 / 2 - 1 / 2j, 0],
    N,
    delt,
    omega,
    beta,
    ga,
    ohmicity,
    wc,
    Hs,
    kmax,
)
A4 = quapi_propagate([0, 0, 0, 1], N, delt, omega, beta, ga, ohmicity, wc, Hs, kmax)

U0 = np.array(
    [
        [1, 0, 0, 0],
        [1 / 2, 1 / 2, 1 / 2, 1 / 2],
        [1, 1 / 2 + 1 / 2j, 1 / 2 - 1 / 2j, 0],
        [0, 0, 0, 1],
    ]
)
U0 = U0.T.copy()
U0inv = numpy.linalg.inv(U0)

rhor = np.zeros((4, 4, len(tarr)), dtype="complex")
U = numpy.zeros((4, 4, N), dtype="complex")  # propagator
for i in range(0, N):  # this fits the propagators from time series data
    rhor[:, 0, i] = A1[:, i]
    rhor[:, 1, i] = A2[:, i]
    rhor[:, 2, i] = A3[:, i]
    rhor[:, 3, i] = A4[:, i]
    U[:, :, i] = rhor[:, :, i] @ U0inv

import numpy

from pyeom.ttm.ttm import comptue_Uk0, generate_M, generate_Mnew

# dynamical memory decomposition
rmax = 4
# Ms = generate_M(U, U[:, :, 0], rmax)
Ms = generate_Mnew(U, rmax)
Uapprox = numpy.zeros_like(U)
Uapprox[:, :, : rmax + 1] = U[:, :, : rmax + 1]
for it in range(rmax + 1, N):
    Uapprox[:, :, it] = comptue_Uk0(Uapprox, Ms, it, rmax)
A1t = np.einsum("ijt,j->it", Uapprox, U0[:, 0])

rmax = 4
Ms = generate_M(U, U[:, :, 0], rmax)
Uapprox = numpy.zeros_like(U)
Uapprox[:, :, : rmax + 1] = U[:, :, : rmax + 1]
for it in range(rmax + 1, N):
    Uapprox[:, :, it] = comptue_Uk0(Uapprox, Ms, it, rmax)

import matplotlib.pyplot as plt

A1t_original = np.einsum("ijt,j->it", Uapprox, U0[:, 0])
plt.plot(A1t[0, :-1], ls="--", label=f"rmax={rmax} (alternative)")
plt.plot(A1t_original[0, :-1], ls="--", label=f"rmax={rmax} (original)")
plt.plot(A1[0, :-1], label=f"quapi, kmax={kmax}")
plt.legend()
plt.show()
