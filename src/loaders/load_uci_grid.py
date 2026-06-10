"""UCI Electrical Grid Stability Simulated Data loader.

Source: https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data
Reference: Schafer et al., "Taming instabilities in power grid networks
           by decentralized reactive power control", Eur. Phys. J. Special
           Topics 225, 569 (2016).

10,000 instances, 14 features:
    - tau1..tau4   : reaction time of producer/consumers (real-valued)
    - p1..p4       : nominal power consumed/produced (real-valued)
    - g1..g4       : coefficient (gamma) proportional to price elasticity
    - stab         : maximum real part of characteristic equation (target)
    - stabf        : 'stable' / 'unstable' (binary label)

Used for: anomaly detection + predictive stability classifier training.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/471/"
    "electrical+grid+stability+simulated+data.zip"
)


def load_uci_grid(
    out_dir: Optional[Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download and load the UCI Grid Stability dataset.

    Args:
        out_dir: Cache directory. Default: data/raw/uci_grid/
        force_download: Re-download even if cached.

    Returns:
        DataFrame with 14 columns (12 features + stab + stabf).
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "uci_grid"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "Data_for_UCI_named.csv"

    if not csv_path.exists() or force_download:
        print(f"[UCI Grid] Downloading from {UCI_ZIP_URL} ...")
        # UCI's CDN drops non-streamed requests intermittently, so stream + chunk.
        headers = {"User-Agent": "Mozilla/5.0 (microgrid-dt ingest)"}
        r = requests.get(UCI_ZIP_URL, headers=headers, stream=True, timeout=300)
        r.raise_for_status()
        buf = bytearray()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                buf.extend(chunk)
        with zipfile.ZipFile(io.BytesIO(bytes(buf))) as zf:
            zf.extractall(out_dir)
        print(f"[UCI Grid] Extracted to {out_dir} ({len(buf):,} bytes)")

    if not csv_path.exists():
        # ZIP may unpack a different filename — pick the first CSV
        csvs = list(out_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {out_dir} after extraction")
        csv_path = csvs[0]

    df = pd.read_csv(csv_path)
    print(f"[UCI Grid] Loaded {len(df)} rows, {df.shape[1]} columns from {csv_path.name}")
    return df


if __name__ == "__main__":
    df = load_uci_grid()
    print(df.head())
    print(f"\nClass distribution:\n{df['stabf'].value_counts() if 'stabf' in df else 'N/A'}")
