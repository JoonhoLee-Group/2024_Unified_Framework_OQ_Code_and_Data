import numpy

from pyeom.quapi.etas import form_etas
from pyeom.quapi.influence_tools import (
    Influencediff,
    Influencediffn,
    Influencediffz,
    Influencenull,
    Influencenullz,
)


def compute_quapi_influence(omega, beta, delt, ga, ohmicity, wc, kmax, hbar=1.0):
    etanul, etanulz, etakkd, etakkdn, etakkdz = form_etas(
        omega, beta, delt, ga, ohmicity, wc, kmax, hbar=hbar
    )

    I0 = numpy.array(
        [
            Influencenull(0, 0, etanul),
            Influencenull(1, 0, etanul),
            Influencenull(0, 1, etanul),
            Influencenull(1, 1, etanul),
        ],
        dtype="complex",
    )
    Ioz = numpy.array(
        [
            Influencenullz(0, 0, etanulz),
            Influencenullz(1, 0, etanulz),
            Influencenullz(0, 1, etanulz),
            Influencenullz(1, 1, etanulz),
        ],
        dtype="complex",
    )
    I = numpy.zeros((4, 4, kmax + 1), dtype="complex")
    Iz = numpy.zeros((4, 4, kmax + 1), dtype="complex")
    IN = numpy.zeros((4, 4, kmax + 1), dtype="complex")
    for i in numpy.arange(0, kmax + 1):
        I[0, 0, i] = Influencediff(0, 0, 0, 0, i, etakkd)
        I[0, 1, i] = Influencediff(1, 0, 0, 0, i, etakkd)
        I[0, 2, i] = Influencediff(0, 1, 0, 0, i, etakkd)
        I[0, 3, i] = Influencediff(1, 1, 0, 0, i, etakkd)

        I[1, 0, i] = Influencediff(0, 0, 1, 0, i, etakkd)
        I[1, 1, i] = Influencediff(1, 0, 1, 0, i, etakkd)
        I[1, 2, i] = Influencediff(0, 1, 1, 0, i, etakkd)
        I[1, 3, i] = Influencediff(1, 1, 1, 0, i, etakkd)

        I[2, 0, i] = Influencediff(0, 0, 0, 1, i, etakkd)
        I[2, 1, i] = Influencediff(1, 0, 0, 1, i, etakkd)
        I[2, 2, i] = Influencediff(0, 1, 0, 1, i, etakkd)
        I[2, 3, i] = Influencediff(1, 1, 0, 1, i, etakkd)

        I[3, 0, i] = Influencediff(0, 0, 1, 1, i, etakkd)
        I[3, 1, i] = Influencediff(1, 0, 1, 1, i, etakkd)
        I[3, 2, i] = Influencediff(0, 1, 1, 1, i, etakkd)
        I[3, 3, i] = Influencediff(1, 1, 1, 1, i, etakkd)
    for i in numpy.arange(0, kmax + 1):
        Iz[0, 0, i] = Influencediffz(0, 0, 0, 0, i, etakkdz)
        Iz[0, 1, i] = Influencediffz(1, 0, 0, 0, i, etakkdz)
        Iz[0, 2, i] = Influencediffz(0, 1, 0, 0, i, etakkdz)
        Iz[0, 3, i] = Influencediffz(1, 1, 0, 0, i, etakkdz)

        Iz[1, 0, i] = Influencediffz(0, 0, 1, 0, i, etakkdz)
        Iz[1, 1, i] = Influencediffz(1, 0, 1, 0, i, etakkdz)
        Iz[1, 2, i] = Influencediffz(0, 1, 1, 0, i, etakkdz)
        Iz[1, 3, i] = Influencediffz(1, 1, 1, 0, i, etakkdz)

        Iz[2, 0, i] = Influencediffz(0, 0, 0, 1, i, etakkdz)
        Iz[2, 1, i] = Influencediffz(1, 0, 0, 1, i, etakkdz)
        Iz[2, 2, i] = Influencediffz(0, 1, 0, 1, i, etakkdz)
        Iz[2, 3, i] = Influencediffz(1, 1, 0, 1, i, etakkdz)

        Iz[3, 0, i] = Influencediffz(0, 0, 1, 1, i, etakkdz)
        Iz[3, 1, i] = Influencediffz(1, 0, 1, 1, i, etakkdz)
        Iz[3, 2, i] = Influencediffz(0, 1, 1, 1, i, etakkdz)
        Iz[3, 3, i] = Influencediffz(1, 1, 1, 1, i, etakkdz)
    for i in numpy.arange(0, kmax + 1):
        IN[0, 0, i] = Influencediffn(0, 0, 0, 0, i, etakkdn)
        IN[0, 1, i] = Influencediffn(1, 0, 0, 0, i, etakkdn)
        IN[0, 2, i] = Influencediffn(0, 1, 0, 0, i, etakkdn)
        IN[0, 3, i] = Influencediffn(1, 1, 0, 0, i, etakkdn)

        IN[1, 0, i] = Influencediffn(0, 0, 1, 0, i, etakkdn)
        IN[1, 1, i] = Influencediffn(1, 0, 1, 0, i, etakkdn)
        IN[1, 2, i] = Influencediffn(0, 1, 1, 0, i, etakkdn)
        IN[1, 3, i] = Influencediffn(1, 1, 1, 0, i, etakkdn)

        IN[2, 0, i] = Influencediffn(0, 0, 0, 1, i, etakkdn)
        IN[2, 1, i] = Influencediffn(1, 0, 0, 1, i, etakkdn)
        IN[2, 2, i] = Influencediffn(0, 1, 0, 1, i, etakkdn)
        IN[2, 3, i] = Influencediffn(1, 1, 0, 1, i, etakkdn)

        IN[3, 0, i] = Influencediffn(0, 0, 1, 1, i, etakkdn)
        IN[3, 1, i] = Influencediffn(1, 0, 1, 1, i, etakkdn)
        IN[3, 2, i] = Influencediffn(0, 1, 1, 1, i, etakkdn)
        IN[3, 3, i] = Influencediffn(1, 1, 1, 1, i, etakkdn)

    return I0, I, Ioz, Iz, IN
