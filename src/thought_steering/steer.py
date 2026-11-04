"""The steering vectors added to the residual stream: a fit's direction scaled to a fraction of a typical activation's
norm."""

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from .graph import BLOCKS, WIDTH

BLOCK = 12  # one-based


def steering(terms, blocks=(BLOCK,)):
    """The additions [blocks, hidden] of fits at strengths, terms = [(fit folder, strength)]: strength * R[l] *
    v[l] / |v[l]| at each chosen one-based block, summed over fits; None when every strength is 0. Each fit's term is
    added in double precision and the sum rounded to float32, as the page's Float32Array does."""
    if not blocks or len(set(blocks)) != len(blocks) or not all(1 <= b <= BLOCKS for b in blocks):
        raise ValueError(f"choose distinct blocks from 1 to {BLOCKS}")
    if all(strength == 0 for _, strength in terms):
        return None
    rows = np.zeros((BLOCKS, WIDTH), np.float32)
    for folder, strength in terms:
        fit = load_file(Path(folder) / "fit.safetensors")
        scale = fit["reference"].astype(np.float64) / fit["norms"]
        vectors = (fit["directions"] * scale[:, None]).astype(np.float32)
        for b in blocks:
            rows[b - 1] = rows[b - 1] + strength * vectors[b - 1].astype(np.float64)
    return rows
