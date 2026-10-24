"""Loading the Criteo Uplift dataset.

The scikit-uplift download mirror for the pre-made 10% sample is offline, so we
download Criteo's official full file (v2.1, ~14M rows) from Criteo's Hugging Face
page and draw a seeded 10% simple random sample ourselves. The sample is cached
as parquet so later runs load in seconds.
"""
from pathlib import Path
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_URL = ("https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
           "criteo-research-uplift-v2.1.csv.gz")
RAW_PATH = ROOT / "data" / "raw" / "criteo-research-uplift-v2.1.csv.gz"

FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ["visit", "conversion"]
SEED = 42


def download_raw(path: Path = RAW_PATH) -> Path:
    """Download the full Criteo CSV (~311 MB) if it isn't already on disk."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, path)
    return path


def load_criteo(frac: float = 0.10, seed: int = SEED) -> pd.DataFrame:
    """Return a seeded random sample of the Criteo uplift data.

    Columns: f0..f11 (float32), treatment, exposure, visit, conversion (int8).
    """
    cache = ROOT / "data" / "processed" / f"criteo_{int(frac * 100)}pct_seed{seed}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    dtypes = {f: "float32" for f in FEATURES}
    dtypes.update({c: "int8" for c in ["treatment", "exposure", *OUTCOMES]})
    full = pd.read_csv(download_raw(), dtype=dtypes)

    df = full.sample(frac=frac, random_state=seed).reset_index(drop=True)
    df = df[FEATURES + ["treatment", "exposure", *OUTCOMES]]

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df
