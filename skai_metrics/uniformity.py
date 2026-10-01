"""Uniform-Depth Quality (UDQ) — grading the *night sky*, not one exposure.

`teff` grades a single frame and `night_metrics` grades a single night. But a
survey like DES is judged by the map it leaves behind: how deep, and how
*evenly* deep, is every patch of sky it set out to cover. Weak-lensing and
large-scale-structure cosmology are limited not by the average depth but by the
**shallow, non-uniform tail** — a survey with the same total exposure time but
more uniform depth is scientifically worth more. DES's whole observing strategy
(Neilsen et al. 2019) is "tile the footprint uniformly" for exactly this reason.

This module builds the sky-level quality metric that follows from that fact.

The banked depth of a patch of sky is its total effective time,

    D(hex, band) = sum over exposures covering it of  teff * exptime           (1)

— the inverse-variance the co-add will stack there (see notebooks/night_sky_teff),
expressed at schedule time. Given a **target depth** T_b (the depth the survey
set out to reach in band b), each hex has a completeness

    comp = min(D / T_b, 1)                                                      (2)

and the survey's headline grade is the **area-weighted completeness** — the
fraction of the planned uniform survey that was actually banked:

    UDQ_b = sum(comp * area) / sum(area),   area = cos(dec)                     (3)

The same target splits the map into two actionable piles:

    deficit  = sum over hexes of  max(T_b - D, 0)     effective seconds short
    surplus  = sum over hexes of  max(D - T_b, 0)     effective seconds over

`deficit` is a literal makeup-observation to-do list: divide by (teff * 90 s) and
it is *the number of exposures* needed to bring the footprint up to target;
`min(surplus, deficit) / deficit` is how much of that shortfall is *already paid
for* and could be closed by re-pointing over-covered time instead of buying new
nights. `depth_deficit` returns the per-hex version of that list.

Two supporting numbers describe the shape of the map on its own terms, with no
target assumed:

  * the **Gini coefficient** of D (0 = perfectly uniform, higher = more unequal);
  * whether the non-uniformity is a *coverage* problem (too few tilings) or a
    *conditions* problem (bad seeing/clouds/sky) — the correlation of D with the
    number of images vs. with median teff. On DES survey data the answer is
    emphatically coverage (corr ~0.89 vs ~0.32): the shallow hexes simply have
    fewer tilings, and they cluster at the footprint edge. That is a fixable,
    schedulable problem, which is the point of surfacing it.
"""

import re

import numpy as np
import pandas as pd

#: Regex pulling the DES tile ("hex") and pass ("tiling") out of the object name,
#: e.g. "DES survey hex 1234 tiling 3". Same pattern used in the night_sky notebook.
HEX_PATTERN = re.compile(r"hex\s+(\S+)\s+tiling\s+(\d+)")


def gini(values):
    """Gini coefficient of a set of non-negative depths (0 = perfectly uniform).

    A single, scale-free number for how *unequal* the depth map is: 0 means every
    hex banked the same effective time, 0.5 means very lopsided. It needs no
    target and is the honest "how uniform did we end up" summary.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    # standard rank-weighted form: 2*sum(i*x_i)/(n*sum x) - (n+1)/n
    return float(2.0 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def _add_hex_columns(df):
    """Parse hex/tiling out of ``object`` and add the ``eff_s`` depth column.

    Returns a copy with ``hexid``, ``tiling`` and ``eff_s`` (= teff * exptime),
    dropping rows whose object string is not a survey hex.
    """
    out = df.copy()
    if "hexid" not in out.columns:
        ext = out["object"].astype(str).str.extract(HEX_PATTERN)
        out["hexid"] = ext[0]
        out["tiling"] = pd.to_numeric(ext[1], errors="coerce")
    out = out[out["hexid"].notna()].copy()
    if "eff_s" not in out.columns:
        out["eff_s"] = out["teff"] * out["exptime"]
    return out


def hex_depth(df, band=None):
    """Collapse the exposure list to one row per hex: its banked depth and context.

    Parameters
    ----------
    df : DataFrame
        Cleaned survey exposures (as from ``load_exposures(survey_only=True)``).
    band : str, optional
        Restrict to one filter. If ``None``, sums across all bands.

    Returns
    -------
    DataFrame indexed by ``hexid`` with columns:
        ``eff_s``   banked depth D — total effective seconds (eq. 1)
        ``n_images``number of exposures stacked there (the coverage / tiling count)
        ``ra``,``dec`` hex-center pointing (median of its exposures)
        ``area``    cos(dec): the true sky-area weight of the hex
        ``teff``    median per-exposure teff (the *conditions* it was taken in)
        ``airmass``,``fwhm`` median conditions, for the coverage-vs-conditions split
    """
    d = _add_hex_columns(df)
    if band is not None:
        d = d[d["filter"] == band]
    g = d.groupby("hexid").agg(
        eff_s=("eff_s", "sum"),
        n_images=("eff_s", "size"),
        ra=("ra_wrapped", "median"),
        dec=("dec", "median"),
        teff=("teff", "median"),
        airmass=("airmass", "median"),
        fwhm=("qc_fwhm", "median"),
    )
    g["area"] = np.cos(np.radians(g["dec"]))
    return g


def _resolve_target(depth, target):
    """Turn a target spec into an effective-seconds number.

    ``target`` may be:
      * a float — an explicit design depth in effective seconds
        (e.g. 10 planned tilings * 90 s = 900);
      * a string ``"pNN"`` — the NNth percentile of the achieved depth,
        i.e. "the depth the well-covered part of the footprint reached".
        Default ``"p75"`` reads as: the shortfall is the non-uniform tail
        below what three-quarters of the sky already got.
    """
    if isinstance(target, str):
        if not target.startswith("p"):
            raise ValueError("string target must look like 'p75'")
        return float(np.percentile(depth, float(target[1:])))
    return float(target)


def uniformity_report(df, band, target="p75"):
    """Grade one band's sky map. Returns a dict of decomposable numbers.

    Keys
    ----
    band, n_hexes, target_eff_s
    udq                 area-weighted completeness in [0,1] (eq. 3) — the headline grade
    gini                inequality of the depth map (0 = uniform)
    cv                  coefficient of variation of depth (std/mean)
    median_depth, mean_depth_areaw
    n_shallow           hexes below half the target (the worst offenders)
    deficit_exposures   makeup 90 s exposures needed to reach target everywhere
    surplus_exposures   over-target exposures that could be re-pointed
    reallocatable_frac  fraction of the deficit already paid for by the surplus
    coverage_corr       corr(depth, n_images) — how much is a *coverage* problem
    conditions_corr     corr(depth, teff)     — how much is a *conditions* problem
    driver              "coverage" or "conditions", whichever dominates
    """
    h = hex_depth(df, band)
    depth = h["eff_s"].to_numpy(dtype=float)
    area = h["area"].to_numpy(dtype=float)
    T = _resolve_target(depth, target)

    comp = np.minimum(depth / T, 1.0)
    udq = float(np.sum(comp * area) / np.sum(area))

    deficit = np.maximum(T - depth, 0.0)
    surplus = np.maximum(depth - T, 0.0)
    # convert eff-seconds to a count of *future* 90 s makeup frames. A makeup frame
    # would be taken in the band's typical conditions, not the shallow hex's own
    # (possibly unlucky, cloudy) history — so we divide by the band-median
    # per-exposure yield teff*90 s, not each hex's teff. Using the hex's own low
    # teff would wildly over-count frames for a hex that simply caught one bad night.
    typ_per_exp = float(np.median(h["teff"])) * 90.0
    deficit_exp = float(deficit.sum() / typ_per_exp)
    surplus_exp = float(surplus.sum() / typ_per_exp)

    cov_corr = float(np.corrcoef(depth, h["n_images"])[0, 1])
    cond_corr = float(np.corrcoef(depth, h["teff"])[0, 1])

    return {
        "band": band,
        "n_hexes": int(len(h)),
        "target_eff_s": T,
        "udq": udq,
        "gini": gini(depth),
        "cv": float(np.std(depth) / np.mean(depth)),
        "median_depth": float(np.median(depth)),
        "mean_depth_areaw": float(np.sum(depth * area) / np.sum(area)),
        "n_shallow": int(np.sum(depth < 0.5 * T)),
        "deficit_exposures": deficit_exp,
        "surplus_exposures": surplus_exp,
        "reallocatable_frac": float(min(surplus.sum(), deficit.sum()) / deficit.sum())
        if deficit.sum() > 0
        else 0.0,
        "coverage_corr": cov_corr,
        "conditions_corr": cond_corr,
        "driver": "coverage" if cov_corr >= cond_corr else "conditions",
    }


def depth_deficit(df, band, target="p75", min_shortfall_exp=1.0):
    """The makeup-observation to-do list: every hex short of target, worst first.

    Returns a DataFrame (one row per under-target hex) with the hex pointing, its
    current depth and coverage, and ``need_exposures`` — how many more 90 s frames
    (at that hex's own conditions) would bring it to target. This is the concrete
    deliverable a mentor can hand to a scheduler: *where* to point and *how much*.

    ``min_shortfall_exp`` drops hexes short by less than one exposure (noise).
    """
    h = hex_depth(df, band)
    T = _resolve_target(h["eff_s"].to_numpy(dtype=float), target)
    h = h.copy()
    h["target_eff_s"] = T
    h["shortfall_eff_s"] = (T - h["eff_s"]).clip(lower=0.0)
    # a makeup frame yields the band's typical teff*90 s (see uniformity_report):
    # project the count with band-median conditions, not this hex's own history
    typ_per_exp = float(np.median(h["teff"])) * 90.0
    h["need_exposures"] = np.ceil(h["shortfall_eff_s"] / typ_per_exp).astype(int)
    todo = h[h["need_exposures"] >= min_shortfall_exp]
    cols = ["ra", "dec", "n_images", "eff_s", "target_eff_s", "need_exposures", "teff", "airmass"]
    return todo[cols].sort_values("need_exposures", ascending=False)


def uniformity_summary(df, bands=("g", "r", "i", "z", "Y"), target="p75"):
    """Run :func:`uniformity_report` for several bands into one tidy DataFrame."""
    return pd.DataFrame([uniformity_report(df, b, target=target) for b in bands]).set_index("band")
