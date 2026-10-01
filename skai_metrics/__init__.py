"""
skai_metrics — evaluation metrics and rewards for AI-based telescope schedulers.

Built and validated on 105,890 real DES exposures (des-exposures.csv).

Modules
-------
data      : loading + cleaning of the DES exposure table
teff      : the effective exposure time t_eff — the image-quality metric that
            serves as the RL reward (Neilsen et al. 2019; Terranova et al. 2023)
blur      : the blur (seeing/FWHM) metric, including the split between
            atmospheric blur (uncontrollable) and airmass blur (the scheduler's choice)
schedule  : schedule-level "goodness" metrics — per-night summaries and a
            composite score for comparing schedulers
uniformity: sky-level Uniform-Depth Quality (UDQ) — how deep and how *evenly*
            deep the survey's footprint is, plus the makeup-observation to-do list
"""

from .data import load_exposures
from .teff import (
    FIDUCIAL_FWHM,
    SKY_ALPHA,
    blur_term,
    cloud_term,
    sky_term,
    compute_teff,
    effective_seconds,
)
from .blur import (
    AIRMASS_EXPONENT,
    WAVELENGTH_NM,
    zenith_seeing,
    airmass_blur_factor,
    to_reference_band,
    blur_decomposition,
)
from .schedule import night_metrics, composite_score, DEFAULT_TEFF_THRESHOLDS
from .uniformity import (
    gini,
    hex_depth,
    uniformity_report,
    depth_deficit,
    uniformity_summary,
)

__version__ = "0.1.0"
