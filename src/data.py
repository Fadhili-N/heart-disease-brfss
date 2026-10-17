"""Loading the raw BRFSS file and defining the classification target."""
from pathlib import Path

import pandas as pd

SEED = 42
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "LLCP2023.XPT"

# BRFSS answer codes: 1 = yes, 2 = no, 7 = don't know / not sure, 9 = refused.
YES, NO = 1, 2
TARGET_SOURCES = ["CVDINFR4", "CVDCRHD4"]  # heart attack, coronary heart disease
TARGET = "heart_disease"


CACHE_PATH = RAW_PATH.with_name("brfss2023_raw.pkl")


def load_raw(path: Path = RAW_PATH, use_cache: bool = True) -> pd.DataFrame:
    """Read the SAS transport file exactly as the CDC publishes it.

    Parsing takes about 100 s, so the first call stores a pickle next to the raw
    file (git-ignored) and later calls read that instead. The content is identical.
    """
    if use_cache and CACHE_PATH.exists():
        return pd.read_pickle(CACHE_PATH)
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    if use_cache:
        df.to_pickle(CACHE_PATH)
    return df


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target and drop rows where it cannot be determined.

    Positive: answered yes to a heart attack or to coronary heart disease.
    Negative: answered no to both.
    Dropped: anything else (don't know, refused, missing) because the true
    label is unknown and guessing it would put noise into the target.
    """
    mi, chd = df["CVDINFR4"], df["CVDCRHD4"]
    positive = (mi == YES) | (chd == YES)
    negative = (mi == NO) & (chd == NO)
    out = df.loc[positive | negative].copy()
    out[TARGET] = positive.loc[out.index].astype("int8")
    return out
