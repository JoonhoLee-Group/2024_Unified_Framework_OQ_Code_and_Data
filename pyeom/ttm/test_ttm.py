import math as math

import numpy
import numpy as np

kmax = 4
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

# N = kmax*2  # timestep

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

# print(numpy.linalg.inv(U0inv))

rhor = np.zeros((4, 4, len(tarr)), dtype="complex")
U = numpy.zeros((4, 4, 100), dtype="complex")  # propagator
for i in range(0, N):  # this fits the propagators from time series data
    rhor[:, 0, i] = A1[:, i]
    rhor[:, 1, i] = A2[:, i]
    rhor[:, 2, i] = A3[:, i]
    rhor[:, 3, i] = A4[:, i]
    U[:, :, i] = rhor[:, :, i] @ U0inv

import numpy

# dynamical memory decomposition
ndim = 4
rmax = 4

from pyeom.ttm.ttm import comptue_Uk0, generate_M

Ms = generate_M(U, U[:, :, 0], rmax)
M21 = Ms[0, :, :]
M20 = Ms[1, :, :]
M30 = Ms[2, :, :]
M40 = Ms[3, :, :]

U10 = U[:, :, 1]
U20 = U[:, :, 2]
U30 = U[:, :, 3]
U40 = U[:, :, 4]
U50 = U[:, :, 5]

U20_new = M21 @ U10 + M20
U30_new = M21 @ U20 + M20 @ U10 + M30
U40_new = M21 @ U30 + M20 @ U20 + M30 @ U10 + M40

numpy.testing.assert_allclose(U20_new, U20)
numpy.testing.assert_allclose(U30_new, U30)
numpy.testing.assert_allclose(U40_new, U40)

U50 = comptue_Uk0(U, Ms, 5, rmax)
U50_new = M21 @ U40 + M20 @ U30 + M30 @ U20 + M40 @ U10
diff = U50 - U50_new
numpy.testing.assert_allclose(U50_new, U50)

Uref = U.copy()
U[:, :, 5] = U50.copy()  # overwrite the approximate dynamics

U60 = comptue_Uk0(U, Ms, 6, rmax)
U60_new = M21 @ U50 + M20 @ U40 + M30 @ U30 + M40 @ U20
diff = U60 - U60_new
numpy.testing.assert_allclose(U60_new, U60)


# Alternative SMATPI TTM
rmax = 3
U = Uref.copy()
Ms = []
U10 = U[:, :, 1]
U20 = U[:, :, 2]
U30 = U[:, :, 3]
M10 = U20 @ numpy.linalg.inv(U10)
Ms += [M10]

for ir in range(1, rmax):
    A = numpy.zeros_like(Ms[0])
    for x in range(ir):
        A += Ms[ir - x - 1] @ U[:, :, 2 + x]
    M = U[:, :, 2 + ir] - A
    Ms += [M]
Ms = numpy.array(Ms)

M20 = U30 - M10 @ U20
numpy.testing.assert_allclose(M20, Ms[1])
# print(Ms.shape)
