import numpy


# need to provide M21
# U is expected to start from t = 0
# compute M10 to Mrmax,0 based on U from t = 0 to t = rmax
def generate_M(U, M21, rmax):
    Ms = []
    Ms += [M21]  # M21 = U10
    for ir in range(1, rmax):
        A = numpy.zeros_like(Ms[0])
        for x in range(ir):
            A += Ms[ir - x - 1] @ U[:, :, 1 + x]
        M = U[:, :, 1 + ir] - A
        Ms += [M]
    Ms = numpy.array(Ms)

    return Ms


# This uses a different formulation of TMM motivated by alternative SMATPI
# U10 = U10
# U20 = M10 U10
# U30 = M10 U20 + M20
# U40 = M10 U30 + M20 U20 + M30
def generate_Mnew(U, rmax):
    assert rmax >= 1
    assert U.shape[-1] > rmax + 1

    Ms = []

    U10 = U[:, :, 1]
    U20 = U[:, :, 2]
    M10 = U20 @ numpy.linalg.inv(U10)
    Ms += [M10]

    for ir in range(1, rmax):
        A = numpy.zeros_like(Ms[0])
        for x in range(ir):
            A += Ms[ir - x - 1] @ U[:, :, 2 + x]
        M = U[:, :, 2 + ir] - A
        Ms += [M]
    Ms = numpy.array(Ms)

    return Ms


# compute Uk0 for k > rmax
# U is expected to start from t = 0
# k is target index
# U60_new = M21 @ U50 + M20 @ U40 + M30 @ U30 + M40 @ U20
def comptue_Uk0(U, Ms, k, rmax):
    Uk0 = numpy.zeros_like(U[:, :, 1])
    for ir in range(rmax):
        Uk0 += Ms[ir] @ U[:, :, k - ir - 1]
    return Uk0
