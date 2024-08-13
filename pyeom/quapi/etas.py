import numpy
from scipy.integrate import trapz


def Js(xs, ga,ohmicity,wc):  # Spectral density (Ohmic bath with exponential cutoff)
    output = numpy.zeros_like(xs)
    output[xs > 0] = ga * (xs**ohmicity)[xs > 0] * numpy.exp(-xs[xs > 0] / wc)/(wc**(ohmicity-1))
    output[xs <= 0] = - ga * ((-xs)**ohmicity)[xs <= 0] * numpy.exp(xs[xs <= 0] / wc)/(wc**(ohmicity-1))
    return output


def form_etas(omega, beta, delt, ga, ohmicity, wc,kmax, hbar=1):
    nomega = omega.size

    etakkd = numpy.zeros(kmax + 1, dtype="complex")
    etakkdz = numpy.zeros(kmax + 1, dtype="complex")
    etakkdn = numpy.zeros(kmax + 1, dtype="complex")

    # compute etanul
    yy = (
        1
        / (2 * numpy.pi)
        * Js(omega, ga,ohmicity,wc)
        / (omega**2)
        * numpy.exp(beta * hbar * omega / 2)
        / (numpy.sinh(beta * hbar * omega / 2))
        * (1 - numpy.exp(-1j * omega * delt))
    )
    etanul = trapz(yy, omega)

    # compute etanulz
    yy = (
        1
        / (2 * numpy.pi)
        * Js(omega, ga,ohmicity,wc)
        / (omega**2)
        * numpy.exp(beta * hbar * omega / 2)
        / (numpy.sinh(beta * hbar * omega / 2))
        * (1 - numpy.exp(-1j * omega * delt / 2))
    )
    etanulz = trapz(yy, omega)

    # compute etakkd
    for difk in range(1, kmax + 1, 1):
        yy = (
            2
            / (1 * numpy.pi)
            * Js(omega, ga,ohmicity,wc)
            / (omega**2)
            * numpy.exp(beta * hbar * omega / 2)
            / (numpy.sinh(beta * hbar * omega / 2))
            * ((numpy.sin(omega * delt / 2)) ** 2)
            * numpy.exp(-1j * omega * delt * difk)
        )
        etakkd[difk] = trapz(yy, omega)

    # compute etakkdz
    for difk in range(1, kmax + 1, 1):
        yy = (
            2
            / (1 * numpy.pi)
            * Js(omega, ga,ohmicity,wc)
            / (omega**2)
            * numpy.exp(beta * hbar * omega / 2)
            / (numpy.sinh(beta * hbar * omega / 2))
            * ((numpy.sin(omega * delt / 4)))
            * (numpy.sin(omega * delt / 2))
            * numpy.exp(-1j * omega * (difk * delt - delt / 4))
        )
        etakkdz[difk] = trapz(yy, omega)

    # compute etakkdn
    for difk in range(1, kmax + 1, 1):
        yy = (
            2
            / (1 * numpy.pi)
            * Js(omega, ga,ohmicity,wc)
            / (omega**2)
            * numpy.exp(beta * hbar * omega / 2)
            / (numpy.sinh(beta * hbar * omega / 2))
            * ((numpy.sin(omega * delt / 4)))
            * (numpy.sin(omega * delt / 2))
            * numpy.exp(-1j * omega * (difk * delt - delt / 4))
        )
        etakkdn[difk] = trapz(yy, omega)

    return etanul, etanulz, etakkd, etakkdn, etakkdz
