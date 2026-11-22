"""Download and load ECB euro-area yield-curve spot rates (percent, continuous compounding)."""
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
MATURITIES = [1, 2, 5, 10, 20, 30]
# G_N_A = AAA government bonds only; G_N_C = all euro-area issuers
CURVES = {"AAA": "G_N_A", "ALL": "G_N_C"}
URL = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.{c}.SV_C_YM.{m}"


BBK_URL = "https://api.statistiken.bundesbank.de/rest/data/BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R{m:02d}XX.R.A.A._Z._Z.A"


def _load_bundesbank(refresh):
    """Bundesbank Svensson zero-coupon curve for German federal securities, daily, percent.
    Daily values start 1997-08 (30Y: 2000-08), so only days with all six maturities are kept."""
    cols = {}
    for n in MATURITIES:
        path = RAW / f"bbk_{n}y.csv"
        if refresh or not path.exists():
            r = requests.get(BBK_URL.format(m=n), params={"format": "csv", "startPeriod": "1997-01-01"}, timeout=60)
            r.raise_for_status()
            RAW.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
        raw = pd.read_csv(path, sep=";", header=None, usecols=[0, 1], dtype=str, encoding="utf-8-sig")
        d = raw[raw[0].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]      # skip metadata rows; decimal comma, '.' = no value
        cols[f"{n}Y"] = pd.Series(pd.to_numeric(d[1].str.replace(",", "."), errors="coerce").values, index=pd.to_datetime(d[0].values))
    df = pd.DataFrame(cols).dropna().sort_index()
    df.index.name = "date"
    return df


GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"


def _load_gsw(refresh):
    """Federal Reserve (Gurkaynak-Sack-Wright) US Treasury zero-coupon curve, daily, percent, continuously compounded.
    The 30Y starts 1985-11, so only days with all six maturities are kept (holidays are all-NaN rows)."""
    path = RAW / "gsw_feds200628.csv"
    if refresh or not path.exists():
        r = requests.get(GSW_URL, timeout=120)
        r.raise_for_status()
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    with open(path, encoding="utf-8-sig") as f:
        header = next(i for i, line in enumerate(f) if line.startswith("Date,"))      # skip the note and legend rows
    cols = [f"SVENY{n:02d}" for n in MATURITIES]
    df = pd.read_csv(path, skiprows=header, usecols=["Date"] + cols, na_values=["NA"], parse_dates=["Date"], index_col="Date")
    df.columns = [f"{n}Y" for n in MATURITIES]
    df.index.name = "date"
    return df.dropna().sort_index()


def load_curve(curve="AAA", refresh=False):
    """Date-indexed DataFrame, columns '1Y'..'30Y', rates in percent. Cached in data/raw.
    curve: 'AAA' / 'ALL' (ECB euro area, from 2004), 'DE' (Bundesbank German zero curve, from 2000)
    or 'US' (Federal Reserve GSW Treasury zero curve, from 1985)."""
    if curve == "DE":
        return _load_bundesbank(refresh)
    if curve == "US":
        return _load_gsw(refresh)
    path = RAW / f"ecb_{curve.lower()}.csv"
    if refresh or not path.exists():
        m = "+".join(f"SR_{n}Y" for n in MATURITIES)
        r = requests.get(
            URL.format(c=CURVES[curve], m=m),
            params={"format": "csvdata", "detail": "dataonly"},
            timeout=60,
        )
        r.raise_for_status()
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    raw = pd.read_csv(path, usecols=["DATA_TYPE_FM", "TIME_PERIOD", "OBS_VALUE"])
    df = raw.pivot(index="TIME_PERIOD", columns="DATA_TYPE_FM", values="OBS_VALUE")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df[[f"SR_{n}Y" for n in MATURITIES]]
    df.columns = [f"{n}Y" for n in MATURITIES]
    return df.sort_index()
