"""Validation of skai_metrics against the real DES pipeline quality values.

Run from the skai-metrics folder:  python3 -m pytest tests/ -q
(or plain:  python3 tests/test_metrics.py)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skai_metrics import (
    load_exposures,
    compute_teff,
    blur_term,
    cloud_term,
    zenith_seeing,
    airmass_blur_factor,
    night_metrics,
    composite_score,
    gini,
    hex_depth,
    uniformity_report,
    depth_deficit,
)


def _survey():
    return load_exposures(survey_only=True)


def test_calibrated_teff_tracks_pipeline():
    """The calibrated model must track DES's own qc_teff closely."""
    f = _survey()
    pred = compute_teff(f["qc_fwhm"], f["qc_cloud"], f["qc_sky"], f["filter"])
    logcorr = np.corrcoef(np.log(pred), np.log(f["teff"]))[0, 1]
    med_err = np.median(np.abs(pred / f["teff"] - 1))
    assert logcorr > 0.95, f"log-corr {logcorr:.3f}"
    assert med_err < 0.16, f"median |err| {med_err:.3f}"


def test_blur_and_cloud_terms_alone_explain_most_of_teff():
    """Sanity: the paper's blur*cloud core should correlate strongly by itself."""
    f = _survey()
    core = blur_term(f["qc_fwhm"], f["filter"]) * cloud_term(f["qc_cloud"])
    logcorr = np.corrcoef(np.log(core), np.log(f["teff"]))[0, 1]
    assert logcorr > 0.90, f"log-corr {logcorr:.3f}"


def test_terms_behave_physically():
    assert cloud_term(0.0) == 1.0
    assert cloud_term(0.5) < cloud_term(0.1) < 1.0
    assert blur_term(2.0, "i") < blur_term(1.0, "i")
    # worse airmass -> more blur
    assert airmass_blur_factor(2.0) > airmass_blur_factor(1.0) == 1.0
    # zenith seeing removes the airmass penalty
    assert np.isclose(zenith_seeing(1.0 * 1.5**0.6, 1.5), 1.0)


def test_night_metrics_shape_and_ranges():
    f = _survey()
    nights = night_metrics(f)
    assert len(nights) > 400  # DES observed many hundreds of nights
    ok = nights[nights["n_exposures"] >= 20]
    assert (ok["open_shutter_efficiency"] <= 1.01).all()
    # identity: effective_eff = mean_teff * open_shutter_eff (t_eff can top 1.0
    # on exceptional nights, so no fixed cap is asserted)
    assert np.allclose(
        ok["effective_efficiency"],
        ok["mean_teff"] * ok["open_shutter_efficiency"],
    )
    assert ok["pass_fraction"].between(0, 1).all()
    scores = composite_score(ok)
    assert scores.between(0, 100).all()


def test_gini_bounds():
    # a perfectly uniform map has Gini 0; a maximally unequal one approaches 1
    assert gini([5, 5, 5, 5]) == 0.0
    assert gini([0, 0, 0, 100]) > 0.6
    assert 0.0 <= gini([1, 2, 3, 4, 5]) <= 1.0


def test_uniformity_report_identities():
    f = _survey()
    rep = uniformity_report(f, "i", target="p75")
    # UDQ is an area-weighted mean of completeness fractions -> in [0, 1]
    assert 0.0 <= rep["udq"] <= 1.0
    # by construction the target is the 75th percentile, so ~25% of hexes exceed it
    # -> there is always both a real deficit and a real surplus on DES data
    assert rep["deficit_exposures"] > 0
    assert rep["surplus_exposures"] > 0
    assert 0.0 <= rep["reallocatable_frac"] <= 1.0
    # the DES depth non-uniformity is a coverage problem, not a conditions one:
    # depth correlates far more with tiling count than with per-exposure teff
    assert rep["coverage_corr"] > rep["conditions_corr"]
    assert rep["driver"] == "coverage"


def test_depth_deficit_reaches_target():
    f = _survey()
    rep = uniformity_report(f, "i", target="p75")
    todo = depth_deficit(f, "i", target="p75")
    # every listed hex is genuinely below target and asks for >=1 makeup frame
    assert (todo["eff_s"] < todo["target_eff_s"]).all()
    assert (todo["need_exposures"] >= 1).all()
    # the per-hex list is the disaggregation of the report's headline deficit.
    # each hex rounds *up* to a whole makeup frame (no fractional exposures), so the
    # summed list must exceed the report's continuous total, but only modestly (the
    # rounding adds <1 frame per hex).
    listed, headline = todo["need_exposures"].sum(), rep["deficit_exposures"]
    assert headline <= listed <= headline + len(todo)
    assert listed < 1.3 * headline


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
