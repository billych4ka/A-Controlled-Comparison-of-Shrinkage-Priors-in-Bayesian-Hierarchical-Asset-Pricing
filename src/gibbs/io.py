"""
src/gibbs/io.py

Saving and loading Gibbs output. Deliberately generic: it serialises any
dataclass whose fields are numpy arrays, plus a JSON-able `meta` dict, so
the Gaussian baseline and the Bayesian LASSO can share it without either
importing from the other. That matters for the dissertation's framing:
the four models are meant to be independent implementations differing only
in the prior, and a shared save/load helper keeps that true in the code.

Format is .npz (compressed) for the arrays plus a sidecar .json for meta.
Two reasons not to pickle: a pickle silently breaks if the dataclass
definition later changes, and .npz needs no third-party reader, the same
lesson as the size_bm_25.pkl / pyarrow problem earlier in this project.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np


def save_draws(draws, path: str | Path) -> Path:
    """
    Save a GibbsDraws-like dataclass to <path>.npz plus <path>_meta.json.

    Fields that are None (e.g. Delta_b when store_Delta_b_full=False) are
    skipped and recorded in the metadata, so load_draws can restore them
    as None rather than guessing.

    draws : any dataclass of numpy arrays with a `meta` dict
    path  : destination, with or without the .npz suffix
    -> the Path actually written
    """
    if not is_dataclass(draws):
        raise TypeError(f"expected a dataclass, got {type(draws).__name__}")

    path = Path(path)
    if path.suffix == ".npz":
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays, absent = {}, []
    for f in fields(draws):
        if f.name == "meta":
            continue
        value = getattr(draws, f.name)
        if value is None:
            absent.append(f.name)
        else:
            arrays[f.name] = np.asarray(value)

    np.savez_compressed(path.with_suffix(".npz"), **arrays)

    meta = dict(getattr(draws, "meta", {}))
    meta["_class"] = type(draws).__name__
    meta["_absent_fields"] = absent
    meta["_shapes"] = {k: list(v.shape) for k, v in arrays.items()}
    with open(f"{path}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    return path.with_suffix(".npz")


def load_draws(path: str | Path, cls):
    """
    Inverse of save_draws. `cls` is the dataclass to rebuild (e.g.
    GibbsDraws), passed explicitly rather than looked up by name, so that
    loading never depends on importing every model's module.

    Returns an instance of cls with arrays restored and meta re-attached.
    """
    path = Path(path)
    if path.suffix == ".npz":
        path = path.with_suffix("")

    with np.load(path.with_suffix(".npz")) as npz:
        arrays = {k: npz[k] for k in npz.files}

    with open(f"{path}_meta.json") as fh:
        meta = json.load(fh)

    kwargs = {f.name: arrays.get(f.name) for f in fields(cls) if f.name != "meta"}
    kwargs["meta"] = {k: v for k, v in meta.items() if not k.startswith("_")}
    return cls(**kwargs)
