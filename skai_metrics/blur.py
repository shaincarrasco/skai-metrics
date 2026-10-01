"""The blur metric — and who is to blame for it.

Delivered image blur (FWHM) has two very different components:

    FWHM_delivered ~ FWHM_zenith * X^0.6 * (lambda/lambda_ref)^-0.2

  * ``FWHM_zenith`` — what the atmosphere is doing tonight. The scheduler
    cannot control it.
  * ``X^0.6``       — the airmass penalty (Kolmogorov turbulence theory).
    The scheduler *chooses* the airmass by choosing where to point.
  * ``lambda^-0.2`` — bluer light is blurred more. The scheduler chooses
    the band, but band choice is usually dictated by survey needs.

A good scheduler reward must not punish the agent for bad atmosphere, only
for pointing decisions. This module separates the two so schedules can be
judged on the part they actually control.
"""

import numpy as np

#: Kolmogorov-turbulence exponent: FWHM grows as airmass**0.6.
AIRMASS_EXPONENT = 0.6

#: Effective central wavelengths of the DECam bands (nm).
WAVELENGTH_NM = {"g": 475.0, "r": 635.0, "i": 775.0, "z": 925.0, "Y": 1000.0}

#: Reference band for cross-band seeing comparisons.
REFERENCE_BAND = "i"


def _wavelengths(band):
    if np.isscalar(band) or isinstance(band, str):
        return WAVELENGTH_NM[band]
    return np.asarray([WAVELENGTH_NM[b] for b in np.asarray(band)])


def zenith_seeing(fwhm, airmass):
    """Remove the airmass penalty: the seeing this exposure *would have had*
    at zenith. This is the atmosphere's contribution alone."""
    return np.asarray(fwhm, dtype=float) / np.asarray(airmass, dtype=float) ** AIRMASS_EXPONENT


def airmass_blur_factor(airmass):
    """X^0.6 — the multiplicative blur cost of pointing at airmass X.

    X=1.0 -> 1.00 (zenith), X=1.5 -> 1.28, X=2.0 -> 1.52.
    Squared, this is the t_eff the scheduler throws away: pointing at X=1.5
    costs (1.28)^-2 = 39 % of the blur term relative to zenith.
    """
    return np.asarray(airmass, dtype=float) ** AIRMASS_EXPONENT


def to_reference_band(fwhm, band, reference=REFERENCE_BAND):
    """Rescale FWHM to what it would be in the reference band (lambda^-0.2 law),
    so seeing in different filters can be compared on one axis."""
    lam = _wavelengths(band)
    lam_ref = WAVELENGTH_NM[reference]
    return np.asarray(fwhm, dtype=float) * (lam_ref / lam) ** -0.2


def blur_decomposition(fwhm, airmass, band):
    """Split delivered blur into (atmosphere, scheduler, band) factors.

    Returns a dict of arrays:
      ``zenith_i``   — seeing reduced to zenith *and* the i band: pure atmosphere.
      ``airmass_factor`` — X^0.6: the scheduler's pointing cost.
      ``band_factor``    — (lambda/lambda_i)^-0.2: cost of the chosen filter.
    The product of the three reproduces the delivered FWHM.
    """
    fwhm = np.asarray(fwhm, dtype=float)
    x_fac = airmass_blur_factor(airmass)
    lam = _wavelengths(band)
    band_fac = (lam / WAVELENGTH_NM[REFERENCE_BAND]) ** -0.2
    zen_i = fwhm / (x_fac * band_fac)
    return {"zenith_i": zen_i, "airmass_factor": x_fac, "band_factor": band_fac}
