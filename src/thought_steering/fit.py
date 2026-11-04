"""The steering direction of a contrast data file, fitted on the steered graph's own activations.

Per block l:
    h(text)  = block l's output at the text's last token, the text read as the reply to the row's context
    h(side)  = h(text) for a text, else the mean of h over the list's elements
    d(row)   = h(positive) - h(negative)
    v[l]     = the mean over situations of the mean d of their rows
    R[l]     = the lower median, over distinct contexts, of the norm of block l's output at the context's last token
Steering at strength s adds s * R[l] * v[l] / |v[l]| (chat.py), so s is a fraction of a typical activation's norm.
"""

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from . import graph
from .prompt import MODEL, REVISION, Tokenizer


def leaves(side):
    """The texts of a side, in order."""
    return [side] if isinstance(side, str) else [text for s in side for text in leaves(s)]


def side_mean(side, h):
    return h(side) if isinstance(side, str) else sum(side_mean(s, h) for s in side) / len(side)


def estimate(rows, tokenizer, runner):
    """{directions [blocks, hidden], norms [blocks], reference [blocks]} (float32), the number of texts read and of
    distinct contexts."""
    differences, context_norms, forwards = {}, {}, 0
    for row in rows:
        prefix = tokenizer.prompt(row["context"])
        _, context_state, state = runner.step(prefix, runner.empty())
        texts = list(dict.fromkeys(leaves(row["negative"]) + leaves(row["positive"])))  # each text once
        # Each text is decoded after the same context state, unsteered.
        h = {text: runner.step(tokenizer.encode(text), state)[1] for text in texts}
        forwards += len(texts)
        context_norms.setdefault(tuple(prefix), np.linalg.norm(context_state, axis=-1))
        differences.setdefault(row["group_id"], []).append(
            side_mean(row["positive"], h.__getitem__) - side_mean(row["negative"], h.__getitem__))
    groups = differences.values()
    directions = np.asarray(sum(sum(d) / len(d) for d in groups) / len(groups), np.float32)
    norms = np.linalg.norm(directions, axis=-1)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError("a block's direction is zero or not finite")
    ordered = np.sort(np.stack(list(context_norms.values())), axis=0)
    reference = ordered[(len(ordered) - 1) // 2]
    return {"directions": directions, "norms": norms, "reference": reference}, forwards, len(context_norms)


def fit(data, out, variant="q4f16"):
    """Fits the data file's direction and writes out/fit.safetensors and out/fit.json."""
    data = Path(data).read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    tokenizer = Tokenizer()
    tensors, forwards, contexts = estimate(rows, tokenizer, graph.Runner(variant))
    # The live fits' manifest: what the fit read, and the counts.
    fingerprint = {"schema": 3, "data_sha256": hashlib.sha256(data).hexdigest(), "model": MODEL, "identity": REVISION,
                   **tokenizer.sha256, "enable_thinking": False, "site": "decoder block output, before the final norm",
                   "extraction": "final candidate token; calibration at final prefix token",
                   "graph": f"{graph.REPO}@{graph.REVISION}", "variant": variant, "surgery": 5,
                   "runtime": f"onnxruntime {version('onnxruntime')} CPU"}
    manifest = {"fingerprint": fingerprint, "groups": len({r["group_id"] for r in rows}), "rows": len(rows),
                "forwards": forwards, "context_prefixes": contexts}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, out / "fit.safetensors.tmp")
    os.replace(out / "fit.safetensors.tmp", out / "fit.safetensors")
    (out / "fit.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return manifest
