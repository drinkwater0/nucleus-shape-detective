#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smíchá normalSource.csv + mutantsSource.csv do jednoho labels.csv,
kde:
    - filename = relativní cesta k obrázku (např. "bleb/6.png")
    - label    = 0 (normal) nebo 1 (mutant/bleb)
    - quality  = číselné skóre z tvého CSV
    - flags    = 'foreign', 'part', ...  (text)
"""

import pandas as pd
from pathlib import Path

BASE = Path("./data")               # kořen dat
OUT  = BASE / "annotations/labels.csv"

def load_source(csv_path: Path, subdir: str, label: int) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        names=[
            "id", "url", "article", "protein",
            "blank", "quality", "flags", "address"
        ],
        dtype={"id": int, "quality": int, "flags": str},
    )
    df["filename"] = df["id"].astype(str) + ".png"
    df["filename"] = subdir + "/" + df["filename"]
    df["label"] = label
    return df[["filename", "label", "quality", "flags"]]

def main() -> None:
    normal  = load_source(BASE / "normalSource.csv",  "normal", 0)
    mutants = load_source(BASE / "mutantsSource.csv", "bleb",   1)
    combined = pd.concat([normal, mutants], ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT, index=False)
    print(f"✓ Wrote {len(combined)} rows → {OUT}")

if __name__ == "__main__":
    main()
