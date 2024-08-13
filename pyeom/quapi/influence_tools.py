import numpy

onetwo = [1, -1]


def Influencediff(dx1, dx2, dx3, dx4, diff, etakkd, hbar=1.0):
    x1 = onetwo[dx1]
    x2 = onetwo[dx2]
    x3 = onetwo[dx3]
    x4 = onetwo[dx4]
    Sum = (
        -1 / hbar * (x3 - x4) * (etakkd[diff] * x1 - numpy.conjugate(etakkd[diff]) * x2)
    )  # eq 12 line 1 Makri quapi I
    return numpy.exp(Sum)


def Influencediffz(
    dx1, dx2, dx3, dx4, diff, etakkdz, hbar=1.0
):  # eq 12 line 5 Makri quapi I
    x1 = onetwo[dx1]
    x2 = onetwo[dx2]
    x3 = onetwo[dx3]
    x4 = onetwo[dx4]
    Sum = (
        -1
        / hbar
        * (x3 - x4)
        * (etakkdz[diff] * x1 - numpy.conjugate(etakkdz[diff]) * x2)
    )
    return numpy.exp(Sum)


def Influencediffn(
    dx1, dx2, dx3, dx4, diff, etakkdn, hbar=1.0
):  # eq 12 line 5 Makri quapi I
    x1 = onetwo[dx1]
    x2 = onetwo[dx2]
    x3 = onetwo[dx3]
    x4 = onetwo[dx4]
    Sum = (
        -1
        / hbar
        * (x3 - x4)
        * (etakkdn[diff] * x1 - numpy.conjugate(etakkdn[diff]) * x2)
    )
    return numpy.exp(Sum)


def Influencenull(dx1, dx2, etanul, hbar=1.0):  # eq 12 line 2 Makri quapi I
    x1 = onetwo[dx1]

    x2 = onetwo[dx2]
    Sum = -1 / hbar * (x1 - x2) * ((etanul) * x1 - numpy.conjugate(etanul) * x2)
    return numpy.exp(Sum)


def Influencenullz(dx1, dx2, etanulz, hbar=1.0):  # eq 12 line 4 Makri quapi I
    x1 = onetwo[dx1]
    x2 = onetwo[dx2]
    Sum = -1 / hbar * (x1 - x2) * ((etanulz) * x1 - numpy.conjugate(etanulz) * x2)
    return numpy.exp(Sum)
