"""The effective exposure time t_eff — image quality as a single number.

The physical model (Neilsen et al. 2019, used as the RL reward in
Terranova et al. 2023, arXiv:2311.18094):

    t_eff = eta^2 * (FWHM_fid / FWHM)^2 * (b_dark / b)

    eta   : atmospheric transmission (clouds); eta^2 = 10^(-0.8 * c) for
            cloud extinction c in magnitudes
    FWHM  : delivered seeing ("blur"); quadratic because a point source's
            light is spread over an area ~ FWHM^2
    b     : sky brightness; scales linearly because background noise
            variance is proportional to sky counts

Interpretation: an exposure of length tau in the given conditions detects
point sources as well as an exposure of length t_eff * tau taken in fiducial
conditions (FWHM_fid seeing, photometric sky, dark sky). t_eff is therefore a
*multiplier on time itself* — which is exactly why it works as a scheduler
reward: maximizing sum(t_eff * tau) maximizes effective open-shutter time.

Calibration on des-exposures.csv.gz (constants fit on a 54,686-exposure subset,
validated on the full 74,323 clean 90-s survey exposures):

  * blur and cloud terms reproduce the DES pipeline t_eff exactly in form
    (log-correlation 0.93 by themselves);
  * the fiducial FWHM is band-dependent and follows the atmospheric
    lambda^-0.2 law (fit: g 1.12", r 0.99", i 0.93", z 0.87", Y 0.90");
  * the released ``qc_sky`` column is a *fractional* sky-brightness excess;
    an empirical per-band exponent alpha_b calibrates it:
    sky term = (1 + qc_sky)^(-alpha_b).

  Full model vs. pipeline qc_teff: log-correlation 0.952, median |error| 15 %
  (naive paper constants: 24 %). See tests/test_metrics.py and
  scripts/make_figures.py (which writes the authoritative numbers to
  site/data/summary.json).
"""

import numpy as np

#: Per-band fiducial FWHM (arcsec), fit on des-exposures.csv. Follows lambda^-0.2.
FIDUCIAL_FWHM = {"g": 1.119, "r": 0.991, "i": 0.932, "z": 0.871, "Y": 0.904}

#: Per-band exponent of the empirical sky term (1 + qc_sky)^(-alpha).
SKY_ALPHA = {"g": 0.589, "r": 0.487, "i": 0.254, "z": 0.213, "Y": 0.881}

#: Survey-quality thresholds used by DES to decide whether an exposure counts
#: or must be retaken (Neilsen et al. 2019).
TEFF_MIN = {"g": 0.2, "r": 0.3, "i": 0.3, "z": 0.3, "Y": 0.2}


def _per_band(mapping, band):
    """Vectorized lookup of a per-band constant. `band` may be scalar or array."""
    if np.isscalar(band) or isinstance(band, str):
        return mapping[band]
    return np.asarray([mapping[b] for b in np.asarray(band)])


def blur_term(fwhm, band="i", fiducial=None):
    """(FWHM_fid / FWHM)^2 — the blur factor of t_eff, in (0, ~1.5].

    1.0 means fiducial seeing; 0.5 means the exposure is worth half its
    clock time because the image is blurred.
    """
    fid = fiducial if fiducial is not None else _per_band(FIDUCIAL_FWHM, band)
    return (fid / np.asarray(fwhm, dtype=float)) ** 2


def cloud_term(cloud_mag):
    """eta^2 = 10^(-0.8 c) for cloud extinction c in magnitudes (clipped at 0).

    c = 0: photometric sky, term = 1. c = 0.5 mag of cloud: term = 0.40.
    """
    c = np.clip(np.asarray(cloud_mag, dtype=float), 0, None)
    return 10 ** (-0.8 * c)


def sky_term(sky_excess, band="i"):
    """(1 + qc_sky)^(-alpha_b): dark-sky ratio from the fractional sky excess.

    qc_sky = 0 is nominal dark sky (term = 1); bright moon can push
    qc_sky to tens (term -> 0 fastest in the bluest bands).
    """
    alpha = _per_band(SKY_ALPHA, band)
    return (1.0 + np.clip(np.asarray(sky_excess, dtype=float), 0, None)) ** (-alpha)


def compute_teff(fwhm, cloud_mag=0.0, sky_excess=0.0, band="i"):
    """Calibrated t_eff from observing conditions.

    Parameters mirror the des-exposures.csv.gz columns
    (``qc_fwhm``, ``qc_cloud``, ``qc_sky``, ``filter``).
    """
    return (
        blur_term(fwhm, band)
        * cloud_term(cloud_mag)
        * sky_term(sky_excess, band)
    )


def effective_seconds(teff, exptime):
    """t_eff * tau — what an exposure is *worth* in fiducial-condition seconds.

    This is the quantity a scheduler should maximize the sum of.
    """
    return np.asarray(teff, dtype=float) * np.asarray(exptime, dtype=float)


def passes_survey_threshold(teff, band):
    """True where the exposure meets the DES keep-or-retake threshold."""
    return np.asarray(teff, dtype=float) >= _per_band(TEFF_MIN, band)
