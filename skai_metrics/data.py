"""Loading and cleaning of the DES exposure table."""

from pathlib import Path

import numpy as np
import pandas as pd

# Default location of the raw table relative to this repo
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "des-exposures.csv"


def load_exposures(path=None, survey_only=False, quality_cuts=True):
    """Load des-exposures.csv with derived columns and optional cleaning.

    Parameters
    ----------
    path : str or Path, optional
        Location of the CSV. Defaults to ``data/des-exposures.csv``.
    survey_only : bool
        Keep only ``program == 'survey'`` 90 s wide-survey exposures
        (drops supernova fields, GW follow-up, engineering).
    quality_cuts : bool
        Drop rows with unphysical quality values (negative/zero FWHM,
        missing t_eff, FWHM > 4", sky excess <= -1). These are a small
        fraction of rows and are sensor/pipeline artifacts, not observations.

    Returns
    -------
    pandas.DataFrame
        With extra columns:
        ``ra_wrapped`` (RA in [-180, 180]), ``dt`` (parsed datetime),
        ``night`` (local observing night: date of the evening the night started,
        computed by shifting Chile-local time back 12 h so one night isn't
        split at midnight).
    """
    csv = Path(path) if path is not None else DEFAULT_CSV
    df = pd.read_csv(csv)

    df["ra_wrapped"] = df["ra"] - 360 * (df["ra"] > 180)
    df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
    # DES timestamps are UTC; CTIO local ~ UTC-4/-5. Shifting back 12 h maps every
    # exposure of one dark period onto the calendar date the night began.
    df["night"] = (df["dt"] - pd.Timedelta(hours=12)).dt.date

    if survey_only:
        df = df[(df["program"] == "survey") & (df["exptime"] == 90.0)]

    if quality_cuts:
        good = (
            df["teff"].notna()
            & (df["teff"] > 0)
            & (df["qc_fwhm"] > 0.4)
            & (df["qc_fwhm"] < 4.0)
            & (df["qc_sky"] > -0.9)
            # a handful of rows carry corrupt timestamps (epoch-zero / unparsable)
            & df["dt"].notna()
            & (df["dt"].dt.year >= 2012)
        )
        df = df[good]

    return df.copy()
