"""Download and load ECB euro-area yield-curve spot rates (percent, continuous compounding)."""
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
MATURITIES = [1, 2, 5, 10, 20, 30]
# G_N_A = AAA government bonds only; G_N_C = all euro-area issuers
CURVES = {"AAA": "G_N_A", "ALL": "G_N_C"}
URL = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.{c}.SV_C_YM.{m}"


def load_curve(curve="AAA", refresh=False):
    """Date-indexed DataFrame, columns '1Y'..'30Y', rates in percent. Cached in data/raw."""
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
