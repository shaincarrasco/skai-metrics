"""Schedule-level "goodness" metrics.

t_eff scores one exposure. A *schedule* is a night's worth of sequential
decisions, so its score must aggregate exposures AND account for time the
telescope wasted between them. The core identity:

    effective science seconds = sum(t_eff * exptime)

and the headline metric is how much of the wall-clock night was converted
into fiducial-condition science time:

    effective_efficiency = sum(t_eff * exptime) / night_span

Sub-scores separate *why* a night scored what it did:
  * yield     — effective_efficiency (everything combined)
  * quality   — fraction of exposures passing the DES keep-or-retake threshold
  * pointing  — mean fraction of the blur term retained given the airmasses the
                scheduler chose: X^(-1.2)  (1.0 = everything at zenith)

``pointing`` isolates the scheduler's own contribution: clouds and bad seeing
are not its fault, but airmass is a choice.
"""

import numpy as np
import pandas as pd

from .teff import TEFF_MIN, passes_survey_threshold
from .blur import AIRMASS_EXPONENT

DEFAULT_TEFF_THRESHOLDS = TEFF_MIN

#: Weights of the composite score (yield, quality, pointing).
DEFAULT_WEIGHTS = {"yield": 0.5, "quality": 0.25, "pointing": 0.25}


def night_metrics(df, group="night"):
    """Per-night schedule metrics from an exposure table.

    Parameters
    ----------
    df : pandas.DataFrame
        Needs columns ``dt, exptime, teff, filter, airmass`` and the grouping
        column (default ``night``, added by :func:`skai_metrics.load_exposures`).

    Returns
    -------
    pandas.DataFrame indexed by night with one row per night.
    """
    df = df.sort_values("dt")
    rows = []
    for night, g in df.groupby(group):
        span = (g["dt"].max() - g["dt"].min()).total_seconds() + g["exptime"].iloc[-1]
        open_shutter = g["exptime"].sum()
        effective = (g["teff"] * g["exptime"]).sum()
        passed = passes_survey_threshold(g["teff"], g["filter"])
        pointing = np.mean(g["airmass"] ** (-2 * AIRMASS_EXPONENT))
        rows.append(
            {
                group: night,
                "n_exposures": len(g),
                "night_span_s": span,
                "open_shutter_s": open_shutter,
                "effective_s": effective,
                "dead_time_s": span - open_shutter,
                "open_shutter_efficiency": open_shutter / span if span > 0 else np.nan,
                "effective_efficiency": effective / span if span > 0 else np.nan,
                "mean_teff": (g["teff"] * g["exptime"]).sum() / open_shutter,
                "pass_fraction": float(np.mean(passed)),
                "median_airmass": g["airmass"].median(),
                "pointing_score": pointing,
                "median_fwhm": g["qc_fwhm"].median() if "qc_fwhm" in g else np.nan,
                "median_cloud": g["qc_cloud"].median() if "qc_cloud" in g else np.nan,
            }
        )
    out = pd.DataFrame(rows).set_index(group)
    return out


def composite_score(metrics, weights=None):
    """Single 0-100 score per night from :func:`night_metrics` output.

    A transparent weighted average of three sub-scores, each already on a
    0-1 scale:

        yield    = effective_efficiency  (capped at 1)
        quality  = pass_fraction
        pointing = pointing_score

    This is deliberately simple (v0.1): the point is a *defensible, decomposable*
    number to iterate on with mentors, not a final answer. Change ``weights``
    to re-balance.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    total = sum(w.values())
    score = (
        w["yield"] * metrics["effective_efficiency"].clip(upper=1.0)
        + w["quality"] * metrics["pass_fraction"]
        + w["pointing"] * metrics["pointing_score"]
    ) / total
    return 100.0 * score
