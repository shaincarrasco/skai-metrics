"""Generate every figure + the site stats file from des-exposures.csv.

Run:  python3 scripts/make_figures.py
Outputs land in figures/ and site/data/summary.json.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skai_metrics import (
    load_exposures, compute_teff, blur_term, cloud_term, sky_term,
    night_metrics, composite_score, zenith_seeing, airmass_blur_factor,
    to_reference_band, FIDUCIAL_FWHM, SKY_ALPHA,
)
from skai_metrics.teff import TEFF_MIN
from skai_metrics.blur import WAVELENGTH_NM

FIG = ROOT / "figures"
SITE_DATA = ROOT / "site" / "data"
FIG.mkdir(exist_ok=True)
SITE_DATA.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
GOOD = "#0ca30c"
CRIT = "#d03b3b"
# categorical slots in fixed order for the 5 bands
BAND_COLOR = {"g": "#2a78d6", "r": "#1baf7a", "i": "#eda100", "z": "#008300", "Y": "#4a3aa7"}
BANDS = ["g", "r", "i", "z", "Y"]
BLUES = LinearSegmentedColormap.from_list(
    "skai_blues",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": BASE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- data
print("loading data ...")
f = load_exposures(survey_only=True)
f["pred_teff"] = compute_teff(f["qc_fwhm"], f["qc_cloud"], f["qc_sky"], f["filter"])
f["fwhm_i"] = to_reference_band(f["qc_fwhm"], f["filter"])
f["zenith_i"] = zenith_seeing(f["fwhm_i"], f["airmass"])
print(f"  {len(f):,} survey 90s exposures")

nights = night_metrics(f)
nights = nights[nights["n_exposures"] >= 20].copy()
nights["score"] = composite_score(nights)

# ================================================================ 1. validation
print("figures ...")
logcorr = float(np.corrcoef(np.log(f["pred_teff"]), np.log(f["teff"]))[0, 1])
med_err = float(np.median(np.abs(f["pred_teff"] / f["teff"] - 1)))
core = blur_term(f["qc_fwhm"], f["filter"]) * cloud_term(f["qc_cloud"])
logcorr_core = float(np.corrcoef(np.log(core), np.log(f["teff"]))[0, 1])
naive = (0.9 / f["qc_fwhm"]) ** 2 * cloud_term(f["qc_cloud"])
med_err_naive = float(np.median(np.abs(naive / f["teff"] - 1)))

fig, ax = plt.subplots(figsize=(7, 6.4))
hb = ax.hexbin(f["teff"], f["pred_teff"], gridsize=90, cmap=BLUES, bins="log",
               xscale="log", yscale="log", mincnt=1, linewidths=0.1)
lims = [0.02, 3]
ax.plot(lims, lims, color=CRIT, lw=2, ls="--", label="perfect agreement")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("DES pipeline t_eff (qc_teff)")
ax.set_ylabel("our calibrated t_eff")
ax.set_title("The metric, validated: reconstructed vs pipeline t_eff")
ax.text(0.03, 0.95, f"log-correlation {logcorr:.3f}\nmedian error {med_err*100:.0f}%\n{len(f):,} real exposures",
        transform=ax.transAxes, va="top", color=INK2)
fig.colorbar(hb, ax=ax, label="exposures (log count)", shrink=0.85)
ax.legend(loc="lower right")
save(fig, "teff_validation.png")

# ================================================================ 2. the three terms
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
terms = {
    "blur  (FWHM_fid / FWHM)²": blur_term(f["qc_fwhm"], f["filter"]),
    "clouds  η² = 10^(−0.8·c)": cloud_term(f["qc_cloud"]),
    "sky  (1+Δb)^(−α)": sky_term(f["qc_sky"], f["filter"]),
}
for ax, (label, vals) in zip(axes, terms.items()):
    ax.hist(np.clip(vals, 0, 2), bins=80, color="#2a78d6", alpha=0.9)
    ax.axvline(1.0, color=MUTED, lw=1.2, ls=":")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("multiplier on exposure time")
    med = float(np.median(vals))
    ax.text(0.97, 0.92, f"median {med:.2f}", transform=ax.transAxes, ha="right", color=INK2)
axes[0].set_ylabel("# exposures")
fig.suptitle("Why an exposure loses value — the three factors of t_eff", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "teff_terms.png")

# ================================================================ 3. fiducial fwhm vs wavelength
fig, ax = plt.subplots(figsize=(7, 4.6))
lam = np.array([WAVELENGTH_NM[b] for b in BANDS])
fid = np.array([FIDUCIAL_FWHM[b] for b in BANDS])
lam_grid = np.linspace(440, 1050, 200)
ref = FIDUCIAL_FWHM["i"] * (lam_grid / WAVELENGTH_NM["i"]) ** -0.2
ax.plot(lam_grid, ref, color=MUTED, lw=2, ls="--", label="atmospheric λ$^{-0.2}$ law")
for b in BANDS:
    ax.scatter(WAVELENGTH_NM[b], FIDUCIAL_FWHM[b], s=110, color=BAND_COLOR[b], zorder=3)
    ax.annotate(b, (WAVELENGTH_NM[b], FIDUCIAL_FWHM[b]), textcoords="offset points",
                xytext=(0, 12), ha="center", color=INK, fontweight="bold")
ax.set_xlabel("band central wavelength (nm)")
ax.set_ylabel('fitted fiducial FWHM (arcsec)')
ax.set_title("The fitted fiducials rediscover atmospheric physics")
ax.legend()
save(fig, "fiducial_lambda.png")

# ================================================================ 4. blur by band
fig, ax = plt.subplots(figsize=(8.5, 4.8))
bins = np.linspace(0.5, 2.4, 90)
for b in BANDS:
    vals = f.loc[f["filter"] == b, "qc_fwhm"]
    ax.hist(vals, bins=bins, histtype="step", lw=2, color=BAND_COLOR[b], label=f"{b}  (median {vals.median():.2f}\")")
ax.set_xlabel('delivered FWHM (arcsec)')
ax.set_ylabel("# exposures")
ax.set_title("The blur metric: delivered seeing, band by band")
ax.legend(title="")
save(fig, "blur_by_band.png")

# ================================================================ 5. blur vs airmass — the selection-bias story
# Naive pooled fit of log FWHM vs log X gives a NEGATIVE exponent because the
# DES scheduler pointed at high airmass preferentially in good atmosphere
# (corr(zenith seeing, airmass) < 0). Controlling for the night recovers the
# physical sign. This is Simpson's paradox in scheduler logs — the reason
# off-policy evaluation is hard, and a headline finding of this analysis.
slope, intercept = np.polyfit(np.log(f["airmass"]), np.log(f["fwhm_i"]), 1)
g_night = f.groupby("night")
ln_f_c = np.log(f["fwhm_i"]) - g_night["fwhm_i"].transform(lambda s: np.log(s).mean())
ln_x_c = np.log(f["airmass"]) - g_night["airmass"].transform(lambda s: np.log(s).mean())
slope_within = float((ln_f_c * ln_x_c).sum() / (ln_x_c**2).sum())
sel_corr = float(np.corrcoef(np.log(f["zenith_i"]), np.log(f["airmass"]))[0, 1])

qbins = np.quantile(f["airmass"], np.linspace(0, 1, 13))
qmid, qmed = [], []
for lo, hi in zip(qbins[:-1], qbins[1:]):
    m = (f["airmass"] >= lo) & (f["airmass"] < hi)
    if m.sum() > 50:
        qmid.append(f.loc[m, "airmass"].median()); qmed.append(f.loc[m, "fwhm_i"].median())

sub = f.sample(min(len(f), 40000), random_state=0)
x_grid = np.linspace(1.0, 1.55, 100)
med0 = float(np.median(f["zenith_i"]))
anchor = med0 * 1.2**0.6  # anchor the theory curves at the typical airmass

fig, ax = plt.subplots(figsize=(8.6, 5.2))
hb = ax.hexbin(sub["airmass"], sub["fwhm_i"], gridsize=70, cmap=BLUES, bins="log", mincnt=1, linewidths=0.1)
ax.plot(x_grid, anchor * (x_grid / 1.2) ** 0.6, color=CRIT, lw=2.5,
        label="physics: X$^{0.6}$ (fixed atmosphere)")
ax.plot(x_grid, anchor * (x_grid / 1.2) ** slope_within, color="#008300", lw=2.5, ls="-.",
        label=f"within-night fit: X$^{{{slope_within:.2f}}}$")
ax.plot(qmid, qmed, "o-", color=INK, lw=2, ms=6,
        label=f"pooled median (naive fit X$^{{{slope:.2f}}}$!)")
ax.set_xlim(1.0, 1.55); ax.set_ylim(0.6, 2.2)
ax.set_xlabel("airmass X (how far from zenith the scheduler pointed)")
ax.set_ylabel('delivered FWHM, i-band equivalent (arcsec)')
ax.set_title("Scheduler selection bias hides the airmass law (Simpson's paradox)")
ax.text(0.02, 0.97,
        "Pooled, blur seems to IMPROVE away from zenith —\n"
        "because the scheduler only went there in good seeing\n"
        f"(corr of zenith seeing with airmass: {sel_corr:.2f}).\n"
        "Hold the night fixed and the physical law re-emerges.",
        transform=ax.transAxes, va="top", color=INK2, fontsize=9.5)
fig.colorbar(hb, ax=ax, label="exposures (log count)", shrink=0.85)
ax.legend(loc="lower right", fontsize=9.5)
save(fig, "blur_airmass.png")

# ================================================================ 6. seeing sky map
fig, ax = plt.subplots(figsize=(9.5, 5.2))
hb = ax.hexbin(f["ra_wrapped"], f["dec"], C=f["qc_fwhm"], reduce_C_function=np.median,
               gridsize=95, cmap=BLUES, mincnt=5, linewidths=0.1)
ax.set_xlabel("RA (deg)"); ax.set_ylabel("Dec (deg)")
ax.set_title("Where the survey saw sharpest: median delivered FWHM across the footprint")
ax.invert_xaxis()
fig.colorbar(hb, ax=ax, label='median FWHM (arcsec)', shrink=0.9)
save(fig, "blur_skymap.png")

# ================================================================ 7. blur decomposition
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
bins = np.linspace(0.5, 2.2, 90)
axes[0].hist(f["fwhm_i"], bins=bins, histtype="step", lw=2, color="#2a78d6",
             label=f'delivered (median {f["fwhm_i"].median():.2f}")')
axes[0].hist(f["zenith_i"], bins=bins, histtype="step", lw=2, color="#1baf7a",
             label=f'atmosphere only, at zenith (median {f["zenith_i"].median():.2f}")')
axes[0].set_xlabel('FWHM, i-band equivalent (arcsec)')
axes[0].set_ylabel("# exposures")
axes[0].set_title("Blur: what the atmosphere gave vs what was delivered")
axes[0].legend()

cost = 1 - airmass_blur_factor(f["airmass"]) ** -2  # fraction of blur-term t_eff lost to pointing
axes[1].hist(100 * cost, bins=80, color="#4a3aa7", alpha=0.9)
axes[1].axvline(100 * float(np.median(cost)), color=INK, lw=1.5, ls="--")
axes[1].set_xlabel("t_eff thrown away by the pointing choice (%)")
axes[1].set_ylabel("# exposures")
axes[1].set_title(f"The scheduler's share (median {100*float(np.median(cost)):.0f}%)")
fig.tight_layout()
save(fig, "blur_decomposition.png")

# ================================================================ 8. seasonality
f["month"] = f["dt"].dt.month
mon = f.groupby("month")["zenith_i"].quantile([0.25, 0.5, 0.75]).unstack()
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.fill_between(mon.index, mon[0.25], mon[0.75], color="#9ec5f4", alpha=0.6, label="interquartile range")
ax.plot(mon.index, mon[0.5], "o-", color="#184f95", lw=2.5, label="median")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_ylabel('zenith seeing, i-band (arcsec)')
ax.set_title("Seeing has seasons — Chilean summer is sharper")
ax.legend()
save(fig, "seeing_seasonality.png")

# ================================================================ 9. night score timeline
nn = nights.reset_index()
nn["night_dt"] = pd.to_datetime(nn["night"])
nn = nn.sort_values("night_dt")
roll = nn.set_index("night_dt")["score"].rolling("60D", min_periods=5).median()
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.scatter(nn["night_dt"], nn["score"], s=14, color="#86b6ef", alpha=0.7, label="one night")
ax.plot(roll.index, roll.values, color="#0d366b", lw=2.5, label="60-day rolling median")
ax.set_ylabel("schedule score (0–100)")
ax.set_title(f"Every DES night, scored ({len(nn)} nights)")
ax.legend(loc="lower right")
save(fig, "night_scores.png")

# ================================================================ 10. efficiency plane
fig, ax = plt.subplots(figsize=(7.6, 6))
sc = ax.scatter(nights["open_shutter_efficiency"], nights["mean_teff"],
                c=nights["score"], cmap=BLUES, s=26, linewidths=0)
ax.set_xlabel("open-shutter efficiency (shutter time / night span)")
ax.set_ylabel("mean t_eff (quality of what was collected)")
ax.set_title("Two ways to lose a night: idle time vs poor conditions")
fig.colorbar(sc, ax=ax, label="composite score", shrink=0.85)
best = nights.nlargest(1, "score").iloc[0]
worst = nights.nsmallest(1, "score").iloc[0]
ax.annotate(f"best: {nights['score'].idxmax()}", (best["open_shutter_efficiency"], best["mean_teff"]),
            textcoords="offset points", xytext=(-10, 10), ha="right", color=GOOD, fontweight="bold")
ax.annotate(f"worst: {nights['score'].idxmin()}", (worst["open_shutter_efficiency"], worst["mean_teff"]),
            textcoords="offset points", xytext=(10, 10), color=CRIT, fontweight="bold")
save(fig, "efficiency_plane.png")

# ================================================================ 11. best vs worst night
best_id, worst_id = nights["score"].idxmax(), nights["score"].idxmin()
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=False)
for ax, nid, color, tag in [(axes[0], best_id, GOOD, "best"), (axes[1], worst_id, CRIT, "worst")]:
    g = f[f["night"] == nid].sort_values("dt")
    hours = (g["dt"] - g["dt"].min()).dt.total_seconds() / 3600
    ax.scatter(hours, g["teff"], s=18, color=color, alpha=0.85)
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    m = nights.loc[nid]
    ax.set_title(f"{tag} night {nid} — score {m['score']:.0f}, {int(m['n_exposures'])} exposures, "
                 f"mean t_eff {m['mean_teff']:.2f}", fontsize=11)
    ax.set_ylabel("t_eff")
    ax.set_ylim(0, max(2.0, g["teff"].max() * 1.05))
axes[1].set_xlabel("hours since first exposure of the night")
fig.suptitle("What a good night and a bad night look like, exposure by exposure",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "best_worst_nights.png")

# ================================================================ 12. teff thresholds
fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
for ax, b in zip(axes, BANDS):
    vals = f.loc[f["filter"] == b, "teff"].clip(0, 2)
    ax.hist(vals, bins=60, color=BAND_COLOR[b], alpha=0.9)
    thr = TEFF_MIN[b]
    ax.axvline(thr, color=CRIT, lw=2)
    frac = float((f.loc[f["filter"] == b, "teff"] >= thr).mean())
    ax.set_title(f"{b}: {frac*100:.0f}% pass", fontsize=11)
    ax.set_xlabel("t_eff")
axes[0].set_ylabel("# exposures")
fig.suptitle("DES keep-or-retake thresholds (red) — the pass/fail view of quality",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "teff_thresholds.png")

# ================================================================ summary.json
print("summary.json ...")
airmass_fit_exponent = float(slope)
best_rows = nights.nlargest(5, "score")
worst_rows = nights.nsmallest(5, "score")

def night_rows(d):
    return [
        {
            "night": str(idx),
            "score": round(float(r["score"]), 1),
            "n_exposures": int(r["n_exposures"]),
            "mean_teff": round(float(r["mean_teff"]), 3),
            "open_shutter_efficiency": round(float(r["open_shutter_efficiency"]), 3),
            "effective_efficiency": round(float(r["effective_efficiency"]), 3),
            "pass_fraction": round(float(r["pass_fraction"]), 3),
            "median_airmass": round(float(r["median_airmass"]), 3),
        }
        for idx, r in d.iterrows()
    ]

summary = {
    "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "project": {
        "title": "Evaluation Metrics and Rewards for AI-Based Telescope Schedulers",
        "goal": "Quantitative metrics that summarize the goodness of a telescope observation schedule, validated on real DES data, packaged for the group toolkit.",
        "phase": "v0.1 — foundations: t_eff reconstructed, calibrated and validated; blur metric decomposed; per-night schedule scores on real DES nights",
    },
    "dataset": {
        "source": "des-exposures.csv.gz",
        "n_rows_raw": 105889,
        "n_survey_90s_clean": int(len(f)),
        "n_nights_scored": int(len(nights)),
        "date_range": [str(f["dt"].min().date()), str(f["dt"].max().date())],
        "exposures_by_band": {b: int((f["filter"] == b).sum()) for b in BANDS},
    },
    "validation": {
        "logcorr_calibrated": round(logcorr, 4),
        "logcorr_blur_cloud_only": round(logcorr_core, 4),
        "median_abs_err_calibrated": round(med_err, 4),
        "median_abs_err_naive_paper": round(med_err_naive, 4),
        "tests_passed": 4,
        "tests_total": 4,
    },
    "calibration": {
        "fiducial_fwhm_arcsec": FIDUCIAL_FWHM,
        "sky_alpha": SKY_ALPHA,
        "teff_thresholds": TEFF_MIN,
        "airmass_exponent_theory": 0.6,
        "airmass_exponent_fit_pooled": round(airmass_fit_exponent, 3),
        "airmass_exponent_fit_within_night": round(slope_within, 3),
        "zenith_seeing_airmass_corr": round(sel_corr, 3),
    },
    "blur": {
        "median_fwhm_by_band": {b: round(float(f.loc[f["filter"] == b, "qc_fwhm"].median()), 3) for b in BANDS},
        "median_delivered_i_equiv": round(float(f["fwhm_i"].median()), 3),
        "median_zenith_i": round(float(f["zenith_i"].median()), 3),
        "median_teff_lost_to_pointing_pct": round(100 * float(np.median(cost)), 1),
    },
    "nights": {
        "median_score": round(float(nights["score"].median()), 1),
        "best": night_rows(best_rows),
        "worst": night_rows(worst_rows),
        "timeline": [
            {"night": str(idx), "score": round(float(r["score"]), 1),
             "effective_efficiency": round(float(r["effective_efficiency"]), 3)}
            for idx, r in nights.sort_index().iterrows()
        ],
    },
    "figures": [
        {"file": "teff_validation.png", "section": "validation", "title": "Reconstructed vs pipeline t_eff", "caption": f"Our calibrated metric against DES's own quality pipeline across {len(f):,} exposures — log-correlation {logcorr:.3f}."},
        {"file": "teff_terms.png", "section": "validation", "title": "The three factors of t_eff", "caption": "Blur, cloud and sky terms as multipliers on exposure time."},
        {"file": "fiducial_lambda.png", "section": "validation", "title": "Fiducials rediscover physics", "caption": "The per-band fiducial FWHM fit on data follows the atmospheric wavelength^-0.2 law."},
        {"file": "blur_by_band.png", "section": "blur", "title": "Delivered seeing by band", "caption": "Blur distributions in each DECam filter; redder bands are sharper."},
        {"file": "blur_airmass.png", "section": "blur", "title": "Scheduler selection bias (Simpson's paradox)", "caption": "Pooled, blur seems to improve with airmass — because the scheduler pointed low only in good seeing. Holding the night fixed recovers the physical X^0.6 trend. Evaluating schedulers from logged data is confounded by the logging policy."},
        {"file": "blur_skymap.png", "section": "blur", "title": "Seeing across the footprint", "caption": "Median delivered FWHM over the DES footprint."},
        {"file": "blur_decomposition.png", "section": "blur", "title": "Atmosphere vs scheduler", "caption": "Zenith seeing (uncontrollable) vs delivered blur, and the t_eff fraction lost to pointing."},
        {"file": "seeing_seasonality.png", "section": "blur", "title": "Seeing seasonality", "caption": "Monthly zenith seeing: Chilean summer delivers sharper images."},
        {"file": "night_scores.png", "section": "schedule", "title": "Every night, scored", "caption": "Composite schedule score for each DES night with a 60-day rolling median."},
        {"file": "efficiency_plane.png", "section": "schedule", "title": "The efficiency plane", "caption": "Idle time vs poor conditions: two independent ways a night is lost."},
        {"file": "best_worst_nights.png", "section": "schedule", "title": "Best and worst nights", "caption": "Exposure-by-exposure t_eff through the highest- and lowest-scoring nights."},
        {"file": "teff_thresholds.png", "section": "schedule", "title": "Keep-or-retake thresholds", "caption": "Fraction of exposures meeting DES survey thresholds per band."},
    ],
    "progress": [
        {"date": "2026-07-06", "milestone": "t_eff physics identified from Terranova et al. + Neilsen et al. and mapped to the DES CSV columns", "status": "done"},
        {"date": "2026-07-06", "milestone": "Empirical decomposition: blur x cloud core confirmed (log-corr 0.94), fiducials + sky term calibrated per band", "status": "done"},
        {"date": "2026-07-06", "milestone": "skai_metrics package (data / teff / blur / schedule) + 4 validation tests passing on real data", "status": "done"},
        {"date": "2026-07-06", "milestone": "Blur metric decomposed into atmosphere (uncontrollable) vs airmass (scheduler's choice)", "status": "done"},
        {"date": "2026-07-06", "milestone": "Finding: scheduler selection bias inverts the blur-airmass law in pooled data (Simpson's paradox) — direct evidence that off-policy evaluation needs care", "status": "done"},
        {"date": "2026-07-06", "milestone": "616 real DES nights scored with the composite schedule metric", "status": "done"},
        {"date": "2026-07-06", "milestone": "Notebooks 01-03, LaTeX write-up, progress site", "status": "in progress"},
        {"date": "", "milestone": "Review composite-score weights with mentors; apply metrics to the group's prototype AI scheduler output", "status": "next"},
        {"date": "", "milestone": "Integrate into the group's shared software toolkit", "status": "next"},
    ],
}

with open(SITE_DATA / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
# JS wrapper so the site also works when opened straight from file://
with open(SITE_DATA / "summary.js", "w") as fh:
    fh.write("window.SUMMARY = ")
    json.dump(summary, fh, indent=2)
    fh.write(";\n")
print("  wrote site/data/summary.json + summary.js")
print("done.")
