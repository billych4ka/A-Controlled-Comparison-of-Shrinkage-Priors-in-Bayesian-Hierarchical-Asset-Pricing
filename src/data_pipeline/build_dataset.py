"""
src/data_pipeline/build_dataset.py

Final orchestration step: builds the per-asset (F_i, r_i) panels for a
universe via features.py, then persists them to data/processed/ so the
model-training code (gibbs/, nuts/) can load a ready-made dataset without
re-running the full load -> clean -> features pipeline each time.

Each universe is saved as a single pickle file containing a dict
{asset_name: (F_i, r_i)}, plus a small metadata dict (K, T, asset names,
date range, universe, drop_columns used, build timestamp) saved alongside
it as JSON for quick, human-readable inspection.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from src.data_pipeline.features import build_features

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

def save_dataset(
    universe: str,
    panels: dict,
    drop_columns: list[str] | None,
) -> tuple[Path, Path]:
    """
    Persist panels (dict {asset_name: (F_i, r_i)}) for one universe to
    data/processed/. Returns (pickle_path, metadata_path).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pickle_path = PROCESSED_DIR / f"{universe}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(panels, f)

    sample_f_i, _ = next(iter(panels.values()))
    metadata = {
        "universe": universe,
        "n_assets": len(panels),
        "asset_names": list(panels.keys()),
        "K": sample_f_i.shape[1],
        "T": sample_f_i.shape[0],
        "predictor_columns": list(sample_f_i.columns),
        "date_range": [str(sample_f_i.index.min()), str(sample_f_i.index.max())],
        "drop_columns": drop_columns,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = PROCESSED_DIR / f"{universe}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return pickle_path, metadata_path

def build_dataset(universe: str, drop_columns: list[str] | None = None) -> dict:
    """
    Full pipeline for one universe: load -> clean -> features -> save.
    Returns the panels dict (also written to disk).
    """
    panels = build_features(universe, drop_columns=drop_columns)
    pickle_path, metadata_path = save_dataset(universe, panels, drop_columns)
    print(f"Saved {universe}: {pickle_path.name}, {metadata_path.name}")
    return panels


def load_dataset(universe: str) -> dict:
    """Load a previously-saved panels dict for a universe."""
    pickle_path = PROCESSED_DIR / f"{universe}.pkl"
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    universes = ["size_op_25", "size_inv_25"]

    for universe in universes:
        panels = build_dataset(universe)

        reloaded = load_dataset(universe)
        sample_asset = list(panels.keys())[0]
        f_i_original, r_i_original = panels[sample_asset]
        f_i_reloaded, r_i_reloaded = reloaded[sample_asset]

        print(f"round-trip check on '{universe}' / '{sample_asset}': "
              f"F_i matches={f_i_original.equals(f_i_reloaded)}, "
              f"r_i matches={r_i_original.equals(r_i_reloaded)}\n")
